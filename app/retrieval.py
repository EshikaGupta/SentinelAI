
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

DENSE_K = 10
SPARSE_K = 10
RERANK_TOP_K = 5
RRF_CONSTANT = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
_reranker_lock = threading.Lock()

_bm25_corpus: list[Document] = []
_bm25_fingerprint = None

# Separate cached retrievers for different source filters and k values.
_bm25_retrievers: dict[tuple, BM25Retriever] = {}
_bm25_lock = threading.RLock()

def _corpus_fingerprint(corpus: list[Document]) -> str:
    """Calculate a fingerprint for the indexed corpus."""
    hasher = hashlib.sha256()

    for document in corpus:
        hasher.update(
            str(document.metadata.get("source", "unknown")).encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(document.page_content.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def _copy_document(
    document: Document,
    extra_metadata: Optional[dict] = None,
) -> Document:
    """Create a document copy without mutating the original metadata."""
    metadata = dict(document.metadata)

    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


def refresh_bm25_cache(
    vector_store: Optional[Chroma] = None,
) -> int:
    """
    Refresh the in-memory BM25 corpus from Chroma.

    Call after indexing/deleting documents and during application startup.
    """
    global _bm25_corpus, _bm25_fingerprint, _bm25_retrievers

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

        # All cached retrievers must be rebuilt after corpus changes.
        _bm25_retrievers.clear()

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
    source: Optional[str] = None,
) -> Optional[BM25Retriever]:
    """
    Return a cached BM25 retriever for the requested corpus/source.

    Retrievers are cached independently to avoid mutating shared k values.
    """
    if k < 1:
        raise ValueError("k must be positive.")

    if not corpus:
        return None

    cache_key = (source or "__all__", k)

    with _bm25_lock:
        if _bm25_fingerprint is None:
            raise RuntimeError(
                "BM25 cache is not initialized. "
                "Call refresh_bm25_cache() first."
            )

        if cache_key not in _bm25_retrievers:
            _bm25_retrievers[cache_key] = (
                BM25Retriever.from_documents(
                    corpus,
                    k=k,
                )
            )

            logger.info(
                "Created BM25 retriever: source=%s chunks=%d k=%d",
                source or "all",
                len(corpus),
                k,
            )

        return _bm25_retrievers[cache_key]


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
    """
    Retrieve dense results with relevance scores.

    Chroma relevance scores are retained as retrieval signals.
    They should not be interpreted as calibrated probabilities.
    """
    return vector_store.similarity_search_with_relevance_scores(
        query,
        k=k,
        filter=filter,
    )


@traceable(name="Reciprocal Rank Fusion")
def fuse_rankings(
    dense_results: list[Document],
    sparse_results: list[Document],
) -> list[Document]:
    """Combine dense and sparse rankings using RRF."""
    fused_scores = {}
    documents_by_key = {}

    for results, ranking_type in (
        (dense_results, "dense"),
        (sparse_results, "sparse"),
    ):
        for rank, document in enumerate(results, start=1):
            key = document_key(document)

            existing = documents_by_key.get(key)

            if existing:
                metadata = dict(existing.metadata)
                metadata.update(document.metadata)

                documents_by_key[key] = Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            else:
                documents_by_key[key] = _copy_document(document)

            if ranking_type == "dense":
                documents_by_key[key].metadata["dense_rank"] = rank
            else:
                documents_by_key[key].metadata["sparse_rank"] = rank

            fused_scores[key] = (
                fused_scores.get(key, 0.0)
                + 1.0 / (RRF_CONSTANT + rank)
            )

    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    results = []

    for rank, key in enumerate(ranked_keys, start=1):
        document = documents_by_key[key]

        results.append(
            _copy_document(
                document,
                {
                    "rrf_score": float(fused_scores[key]),
                    "rrf_rank": rank,
                },
            )
        )

    return results


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

    # Apply the same source filter to both retrieval methods.
    corpus = [
        doc for doc in chunks
        if source is None or doc.metadata.get("source") == source
    ]

    if not corpus:
        return []

    # 1. Dense retrieval.
    dense_scored_results = run_dense(
        vector_store,
        query,
        dense_k,
        {"source": source} if source else None,
    )

    dense_results = []

    for rank, (document, score) in enumerate(
        dense_scored_results,
        start=1,
    ):
        dense_results.append(
            _copy_document(
                document,
                {
                    "dense_score": float(score),
                    "dense_rank": rank,
                },
            )
        )

    # 2. Sparse retrieval using the matching cached corpus.
    bm25 = get_bm25_retriever(
        corpus=corpus,
        k=sparse_k,
        source=source,
    )

    sparse_results = (
        run_bm25(bm25, query)
        if bm25 is not None
        else []
    )

    sparse_results = [
        _copy_document(
            document,
            {"sparse_rank": rank},
        )
        for rank, document in enumerate(
            sparse_results,
            start=1,
        )
    ]

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
    """Rerank candidate chunks and retain confidence-related signals."""
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

    all_scores = [float(score) for _, score in ranked]

    top_score = all_scores[0]
    second_score = (
        all_scores[1]
        if len(all_scores) > 1
        else None
    )

    score_gap = (
        top_score - second_score
        if second_score is not None
        else None
    )

    score_mean = sum(all_scores) / len(all_scores)

    score_variance = sum(
        (score - score_mean) ** 2
        for score in all_scores
    ) / len(all_scores)

    score_std = score_variance ** 0.5

    results = []

    for rank, (document, score) in enumerate(
        ranked[:top_k],
        start=1,
    ):
        results.append(
            _copy_document(
                document,
                {
                    "rerank_score": float(score),
                    "rerank_rank": rank,
                    "rerank_top_score": top_score,
                    "rerank_score_gap": score_gap,
                    "rerank_score_mean": score_mean,
                    "rerank_score_std": score_std,
                },
            )
        )

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

    logger.info(
        "Retrieval: %.2f ms | Reranking: %.2f ms | "
        "Candidates: %d | Returned: %d",
        retrieval_ms,
        rerank_ms,
        len(candidates),
        len(results),
    )

    return results