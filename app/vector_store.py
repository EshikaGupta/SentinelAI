
import hashlib
import logging
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = Path("chroma_db")
COLLECTION_NAME = "sentinel_documents"

logger = logging.getLogger(__name__)

_embeddings = None
_embeddings_lock = threading.Lock()
_index_lock = threading.RLock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embedding model once per process."""
    global _embeddings

    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    return _embeddings


def get_vector_store() -> Chroma:
    """Open the persistent collection without re-indexing documents."""
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_source(source_path: str | Path) -> str:
    """Return a stable absolute path for identifying a source document."""
    return str(Path(source_path).resolve())


def _document_id(
    source_id: str,
    content_hash: str,
    chunk_index: int,
) -> str:
    """Generate a deterministic ID for a chunk."""
    identity = f"{source_id}:{content_hash}:{chunk_index}"
    return _sha256(identity)


def index_document(
    chunks: list[Document],
    source_path: str | Path,
    document_text: str,
) -> dict:
    """
    Index one document idempotently.

    - Unchanged content reuses existing chunk IDs.
    - Updated content receives new chunk IDs.
    - Stale chunks for this source are removed after new chunks are indexed.
    """
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("Document text must not be empty.")

    if not chunks:
        raise ValueError("Cannot index a document with no chunks.")

    source = _canonical_source(source_path)
    source_id = _sha256(source)
    content_hash = _sha256(document_text)

    prepared_chunks = []

    for index, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            continue

        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "source": source,
                "source_id": source_id,
                "content_hash": content_hash,
                "chunk_index": index,
            }
        )

        prepared_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    if not prepared_chunks:
        raise ValueError("Document contains no non-empty chunks.")

    ids = [
        _document_id(source_id, content_hash, chunk_index)
        for chunk_index, chunk in enumerate(prepared_chunks)
    ]

    with _index_lock:
        store = get_vector_store()

        # Capture the existing IDs for this source before modifying it.
        existing = store.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        old_ids = set(existing["ids"])
        new_ids = set(ids)

        # Chroma upserts records with matching IDs.
        store.add_documents(
            documents=prepared_chunks,
            ids=ids,
        )

        # Remove obsolete chunks only after the new chunks have been written.
        stale_ids = old_ids - new_ids
        if stale_ids:
            store.delete(ids=list(stale_ids))

    logger.info(
        "Document indexed: source_id=%s chunks=%d stale_removed=%d",
        source_id,
        len(prepared_chunks),
        len(stale_ids),
    )

    return {
        "source": source,
        "content_hash": content_hash,
        "chunks_indexed": len(prepared_chunks),
        "stale_chunks_removed": len(stale_ids),
    }


def delete_document(source_path: str | Path) -> int:
    """Delete all indexed chunks associated with a source path."""
    source = _canonical_source(source_path)
    source_id = _sha256(source)

    with _index_lock:
        store = get_vector_store()
        existing = store.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        ids = existing["ids"]

        if ids:
            store.delete(ids=ids)

    logger.info(
        "Document deleted: source_id=%s chunks_removed=%d",
        source_id,
        len(ids),
    )

    return len(ids)


def search_documents(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> list[Document]:
    """Retrieve the top-k semantically similar chunks."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    return vector_store.similarity_search(query.strip(), k=k)


if __name__ == "__main__":
    from app.ingestion import (
        DOCUMENT_PATH,
        load_document,
        split_document,
    )

    text = load_document(DOCUMENT_PATH)
    chunks = split_document(
        text,
        source=str(DOCUMENT_PATH.resolve()),
    )

    result = index_document(
        chunks=chunks,
        source_path=DOCUMENT_PATH,
        document_text=text,
    )

    print("Indexing complete:")
    print(result)
