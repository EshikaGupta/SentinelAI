import hashlib
import logging
import threading
from pathlib import Path
from .ingestion import load_document, split_document
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "sample_docs"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "sentinel_documents"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

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
    """Open the persistent Chroma collection without indexing."""
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_source(source_path: str | Path) -> str:
    return str(Path(source_path).resolve())


def _document_id(source_id: str, content_hash: str, chunk_index: int) -> str:
    return _sha256(f"{source_id}:{content_hash}:{chunk_index}")


def index_document(
    chunks: list[Document],
    source_path: str | Path,
    document_text: str,
) -> dict:
    """Idempotently index one file and remove its stale chunks."""
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("Document text must not be empty.")

    if not chunks:
        raise ValueError("Cannot index a document with no chunks.")

    source = _canonical_source(source_path)
    source_id = _sha256(source)
    content_hash = _sha256(document_text)
    prepared = []

    for index, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            continue

        metadata = dict(chunk.metadata)
        metadata.update({
            "source": source,
            "source_id": source_id,
            "content_hash": content_hash,
            "chunk_index": index,
        })
        prepared.append(Document(page_content=chunk.page_content, metadata=metadata))

    if not prepared:
        raise ValueError("Document contains no non-empty chunks.")

    ids = [
        _document_id(source_id, content_hash, i)
        for i, _ in enumerate(prepared)
    ]

    with _index_lock:
        store = get_vector_store()
        existing = store.get(where={"source_id": source_id}, include=["metadatas"])
        old_ids = set(existing["ids"])
        store.add_documents(documents=prepared, ids=ids)
        stale_ids = old_ids - set(ids)
        if stale_ids:
            store.delete(ids=list(stale_ids))

    logger.info(
        "Indexed %s: chunks=%d stale_removed=%d",
        source, len(prepared), len(stale_ids),
    )

    return {
        "source": source,
        "content_hash": content_hash,
        "chunks_indexed": len(prepared),
        "stale_chunks_removed": len(stale_ids),
    }

    
def index_directory(directory: str | Path = DOCUMENTS_DIRECTORY) -> list[dict]:
    """Index every supported document in a directory (recursively)."""

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Document directory not found: {directory}")

    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported documents found in {directory}")

    results = []
    for path in paths:
        text = load_document(path)
        chunks = split_document(text, source=str(path))
        results.append(index_document(chunks, path, text))
    return results


def delete_document(source_path: str | Path) -> int:
    source_id = _sha256(_canonical_source(source_path))
    with _index_lock:
        store = get_vector_store()
        existing = store.get(where={"source_id": source_id}, include=["metadatas"])
        ids = existing["ids"]

        if ids:
            store.delete(ids=ids)
    return len(ids)


def search_documents(vector_store: Chroma, query: str, k: int = 5) -> list[Document]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    return vector_store.similarity_search(query.strip(), k=k)


if __name__ == "__main__":
    from pprint import pprint
    pprint(index_directory())
