
from app.ingestion import DOCUMENT_PATH, load_document, split_document
from app.vector_store import index_document, get_vector_store
from app.retrieval import retrieve_and_rerank


def main():
    text = load_document(DOCUMENT_PATH)
    chunks = split_document(text)

    index_document(
        chunks=chunks,
        source_path=DOCUMENT_PATH,
        document_text=text,
    )

    vector_store = get_vector_store()

    queries = [
        "What does error 429 mean?",
        "What is the default API timeout?",
        "How should authentication credentials be handled in logs?",
    ]

    for query in queries:
        print(f"\nQUERY: {query}")

        results = retrieve_and_rerank(
            query=query,
            chunks=chunks,
            vector_store=vector_store,
            top_k=3,
        )

        for result in results:
            print(
                f"\nRerank score: "
                f"{result.metadata['rerank_score']:.4f}"
            )
            print(f"Source: {result.metadata.get('source')}")
            print(result.page_content)


if __name__ == "__main__":
    main()
