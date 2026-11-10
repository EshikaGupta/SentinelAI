import threading
import time
from typing import Optional

from langsmith import traceable
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import hashlib
import logging
from .monitoring import record_stage
from .vector_store import Chroma, get_vector_store

logger = logging.getLogger(__name__)

MIN_RERANK_SCORE = 0.0
DENSE_K = 10
SPARSE_K = 10
RERANK_TOP_K = 5
RRF_CONSTANT = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
_reranker_lock = threading.Lock()

_bm25_retriever = None
_bm25_corpus = []
_bm25_fingerprint = None
_bm25_lock = threading.RLock()

def _corpus_fingerprint(corpus: list[Document]) -> str:
    """Calculate a fingerprint when the corpus is refreshed."""
    hasher = hashlib.sha256()

    for document in corpus:
        hasher.update(
            str(document.metadata.get("source", "unknown")).encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(document.page_content.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()

def refresh_bm25_cache(vector_store: Optional[Chroma] = None) -> int:
    """
    Refresh the in-memory BM25 corpus from the persistent Chroma collection.
    Call only after indexing or deleting documents, or during startup.
    """
    global _bm25_retriever, _bm25_corpus, _bm25_fingerprint

    vector_store = vector_store or get_vector_store()

    stored = vector_store.get(
        include=["documents", "metadatas"]
    )

    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    corpus = [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
        for text, metadata in zip(texts, metadatas)
        if text and text.strip()
    ]

    fingerprint = _corpus_fingerprint(corpus)

    with _bm25_lock:
        if fingerprint == _bm25_fingerprint:
            return len(_bm25_corpus)

        _bm25_corpus = corpus
        _bm25_fingerprint = fingerprint

        _bm25_retriever = (
            BM25Retriever.from_documents(corpus)
            if corpus
            else None
        )

        if _bm25_retriever is not None:
            _bm25_retriever.k = SPARSE_K

        logger.info(
            "BM25 cache refreshed: chunks=%d",
            len(corpus),
        )

    return len(corpus)

def get_cached_chunks() -> list[Document]:
    """Return the in-memory corpus without reading source files."""
    with _bm25_lock:
        if _bm25_fingerprint is None:
            raise RuntimeError(
                "BM25 cache has not been initialized. "
                "Call refresh_bm25_cache() during application startup."
            )

        return list(_bm25_corpus)

def get_bm25_retriever(
    corpus: list[Document],
    k: int = SPARSE_K,
) -> Optional[BM25Retriever]:
    """
    Return the cached retriever.
    The corpus is rebuilt only by refresh_bm25_cache().
    """
    global _bm25_retriever

    with _bm25_lock:
        if _bm25_fingerprint is None:
            raise RuntimeError(
                "BM25 cache is not initialized. "
                "Call refresh_bm25_cache() first."
            )

        if not _bm25_corpus:
            return None

        _bm25_retriever.k = k
        return _bm25_retriever


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once per process."""
    global _reranker

    with _reranker_lock:
        if _reranker is None:
            _reranker = CrossEncoder(
                RERANK_MODEL,
                device="cpu",
            )

    return _reranker


def document_key(document: Document) -> str:
    """Create a stable key for deduplicating retrieved chunks."""
    source = str(document.metadata.get("source", "unknown"))
    content = document.page_content
    raw_key = f"{source}\n{content}"

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

@traceable(name="BM25 Retrieval",run_type="retriever")
def run_bm25(bm25, query):
    return bm25.invoke(query)

@traceable(name="Dense Retrieval",run_type="retriever")
def run_dense(vector_store, query, k, filter):
    return vector_store.similarity_search(query, k=k, filter=filter)


@traceable(name="Reciprocal Rank Fusion")
def fuse_rankings(
    dense_results: list[Document],
    sparse_results: list[Document],
) -> list[Document]:
    """Combine dense and sparse rankings using RRF."""
    fused_scores = {}
    documents_by_key = {}

    for results in (dense_results, sparse_results):
        for rank, document in enumerate(results, start=1):
            key = document_key(document)
            documents_by_key[key] = document

            fused_scores[key] = (
                fused_scores.get(key, 0.0)
                + 1.0 / (RRF_CONSTANT + rank)
            )

    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    return [documents_by_key[key] for key in ranked_keys]


def hybrid_retrieve(
    query: str,
    chunks: list[Document],
    vector_store: Chroma,
    source: Optional[str] = None,
    dense_k: int = DENSE_K,
    sparse_k: int = SPARSE_K,
) -> list[Document]:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    if dense_k < 1 or sparse_k < 1:
        raise ValueError("Retrieval limits must be positive.")

    if not chunks:
        return []

    query = query.strip()

    # Filter the cached corpus if a specific source is requested.
    corpus = [
        doc for doc in chunks
        if source is None or doc.metadata.get("source") == source
    ]

    if not corpus:
        return []

    # 1. Dense retrieval from Chroma.
    dense_results = run_dense(
        vector_store,
        query,
        dense_k,
        {"source": source} if source else None,
    )

    # 2. Sparse retrieval from the cached BM25 corpus.
    bm25 = get_bm25_retriever(corpus, k=sparse_k)

    sparse_results = (
        run_bm25(bm25, query)
        if bm25 is not None
        else []
    )

    # 3. Reciprocal Rank Fusion.
    candidates = fuse_rankings(
        dense_results=dense_results,
        sparse_results=sparse_results,
    )

    logger.info(
        "Hybrid retrieval completed: dense=%d sparse=%d unique=%d",
        len(dense_results),
        len(sparse_results),
        len(candidates),
    )

    return candidates

@traceable(name="Crossencoder Reranking")
def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = RERANK_TOP_K,
) -> list[Document]:
    """Rerank candidate chunks using query-document relevance scores."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (query.strip(), document.page_content)
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for document, score in ranked[:top_k]:
        # Return the score with the chunk for debugging and evaluation.
        result = Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "rerank_score": float(score),
            },
        )
        results.append(result)

    return results

@traceable(name="Hybrid Retrieval Pipeline")
def retrieve_and_rerank(
    query: str,
    chunks: list[Document],
    vector_store: Chroma,
    source: Optional[str] = None,
    top_k: int = RERANK_TOP_K,
) -> list[Document]:
    start = time.perf_counter()

    candidates = hybrid_retrieve(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        source=source,
    )
    retrieval_ms = (time.perf_counter() - start) * 1000
    record_stage(
        "hybrid_retrieval",
        retrieval_ms,
        candidate_count=len(candidates),
    )

    
    rerank_start = time.perf_counter()
    results = rerank_documents(
        query=query,
        documents=candidates,
        top_k=top_k,
    )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000
    record_stage(
        "reranking",
        rerank_ms,
        candidate_count=len(candidates),
        returned_count=len(results),
    )
    print(
        f"Retrieval: {retrieval_ms:.2f} ms | "
        f"Reranking: {rerank_ms:.2f} ms | "
        f"Candidates: {len(candidates)} | "
        f"Returned: {len(results)}"
    )

    return results