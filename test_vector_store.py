
from app.ingestion import DOCUMENT_PATH, load_document, split_document
from app.vector_store import (
    get_vector_store,
    index_document,
    search_documents,
)

def main():
    text = load_document(DOCUMENT_PATH)
    chunks = split_document(text)

    result = index_document(
        chunks=chunks,
        source_path=DOCUMENT_PATH,
        document_text=text,
    )
    print("Index result:", result)

    store = get_vector_store()
    results = search_documents(
        store,
        "What is the default timeout for API requests?",
        k=2,
    )

    for document in results:
        print("\nSource:", document.metadata.get("source"))
        print("Chunk:", document.metadata.get("chunk_index"))
        print("Content:", document.page_content)


if __name__ == "__main__":
    main()
