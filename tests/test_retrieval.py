from pathlib import Path

from app.ingestion import load_document, split_document
from app.retrieval import retrieve_and_rerank
from app import vector_store as vector_store_module


def test_multi_topic_retrieval_returns_both_sources(tmp_path, monkeypatch):
    # Isolate this test from the application's persistent Chroma database.
    monkeypatch.setattr(
        vector_store_module,
        "PERSIST_DIRECTORY",
        tmp_path / "chroma_db",
    )

    docs_dir = vector_store_module.DOCUMENTS_DIRECTORY
    filenames = ["03_climate.pdf", "04_data_literacy.pdf"]

    all_chunks = []

    for filename in filenames:
        path = docs_dir / filename
        assert path.is_file(), f"Test document not found: {path}"

        text = load_document(path)
        chunks = split_document(text, source=str(path.resolve()))

        vector_store_module.index_document(
            chunks=chunks,
            source_path=path,
            document_text=text,
        )
        all_chunks.extend(chunks)

    store = vector_store_module.get_vector_store()

    results = retrieve_and_rerank(
        query="What is Earth's climate influenced by, and what is data literacy?",
        chunks=all_chunks,
        vector_store=store,
        top_k=5,
    )

    returned_sources = {
        Path(doc.metadata["source"]).name
        for doc in results
    }

    assert "03_climate.pdf" in returned_sources
    assert "04_data_literacy.pdf" in returned_sources