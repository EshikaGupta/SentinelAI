
import hashlib
import logging
import threading
from typing import Optional
from langsmith import traceable
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from app.vector_store import Chroma

logger = logging.getLogger(__name__)

DENSE_K = 10
SPARSE_K = 10
RERANK_TOP_K = 5
RRF_CONSTANT = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
_reranker_lock = threading.Lock()

_bm25_retriever = None
_bm25_fingerprint = None
_bm25_lock = threading.Lock()


def get_bm25_retriever(
    corpus: list[Document],
    k: int = SPARSE_K,
) -> BM25Retriever:
    """Reuse BM25 while the corpus remains unchanged."""
    global _bm25_retriever, _bm25_fingerprint

    # Include the source and text of every chunk in the fingerprint.
    hasher = hashlib.sha256()

    for document in corpus:
        hasher.update(
            str(document.metadata.get("source", "unknown")).encode("utf-8")
        )
        hasher.update(b"\0")
        hasher.update(document.page_content.encode("utf-8"))
        hasher.update(b"\0")

    fingerprint = hasher.hexdigest()

    with _bm25_lock:
        if (
            _bm25_retriever is None
            or _bm25_fingerprint != fingerprint
        ):
            _bm25_retriever = BM25Retriever.from_documents(corpus)
            _bm25_fingerprint = fingerprint

            logger.info(
                "BM25 index rebuilt: chunks=%d",
                len(corpus),
            )

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

from langsmith import traceable

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
    """
    Retrieve candidates using dense and sparse search,
    then combine their rankings with Reciprocal Rank Fusion.

    `chunks` should represent the current indexed corpus.
    `source`, if provided, must match the stored source metadata.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    if dense_k < 1 or sparse_k < 1:
        raise ValueError("Retrieval limits must be positive.")

    if not chunks:
        return []

    query = query.strip()

    # Apply the same source restriction to both retrievers.
    corpus = []

    for index, doc in enumerate(chunks):
        if source is not None and doc.metadata.get("source") != source:
            continue

        # Copy the document and preserve existing metadata.
        metadata = dict(doc.metadata)

        # Add the index if it isn't already present.
        metadata.setdefault("chunk_index", index)

        corpus.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    if not corpus:
        return []

    # 1. Dense retrieval: semantic similarity from Chroma.

    dense_results = run_dense(vector_store, query, dense_k, {"source": source} if source else None)

    # 2. Sparse retrieval: lexical keyword matching with BM25.
    
    bm25 = get_bm25_retriever(corpus, k=sparse_k)
    sparse_results = run_bm25(bm25, query)

    # 3. Reciprocal Rank Fusion.
    # Each result contributes 1 / (constant + rank).
    # fused_scores = {}
    # documents_by_key = {}

    # for results in (dense_results, sparse_results):
    #     for rank, document in enumerate(results, start=1):
    #         key = document_key(document)
    #         documents_by_key[key] = document
    #         fused_scores[key] = (
    #             fused_scores.get(key, 0.0)
    #             + 1.0 / (RRF_CONSTANT + rank)
    #         )

    # ranked_keys = sorted(
    #     fused_scores,
    #     key=fused_scores.get,
    #     reverse=True,
    # )
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
    """Run the complete hybrid retrieval and reranking pipeline."""
    candidates = hybrid_retrieve(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        source=source,
    )
    # print("AFTER HYBRID RETRIEVAL:")
    # for doc in candidates:
    #     print(doc.metadata)

    results= rerank_documents(
        query=query,
        documents=candidates,
        top_k=top_k,
    )
    # print("AFTER RERANKING:")
    # for doc in results:
    #     print(doc.metadata)
    return results
