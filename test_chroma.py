
from app.vector_store import get_vector_store

store = get_vector_store()

# Get every record in the collection
data = store.get(include=["documents", "metadatas", "embeddings"])

print("Total records:", len(data["ids"]))
print("Collection name:", store._collection.name)

# Show available fields
print("\nFields:", list(data.keys()))

# Display each record
for i, record_id in enumerate(data["ids"]):
    print(f"\n{'=' * 60}")
    print(f"Row: {i + 1}")
    print(f"ID: {record_id}")
    print(f"Document/chunk:\n{data['documents'][i]}")
    print(f"Metadata: {data['metadatas'][i]}")

    embedding = data["embeddings"][i]
    if embedding is not None:
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
