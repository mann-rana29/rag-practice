import chromadb
from sentence_transformers import SentenceTransformer

chunks = [
    "Python is used for programming.",
    "React is used for frontend interfaces.",
    "RAG retrieves relevant context for an LLM.",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_practice")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True   
).tolist()

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents= chunks,
    embeddings=embeddings,
    metadatas=[
        {"source" : "sample.txt", "chunk_index" : i} for i in range(len(chunks))
    ]
)

query = "What retrieves context for an LLM?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)[0].tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    include=["documents","metadatas","distances"]
)

for doc, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print("\nDocument:", doc)
    print("Metadata:",metadata)
    print("Distance:",distance)