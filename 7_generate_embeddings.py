from sentence_transformers import SentenceTransformer

documents = [
    "Python is a programming language",
    "React is a frontend library",
    "Rag retrieves relevant context"
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Number of embeddings : ", len(embeddings))
print("Embedding shape : ", embeddings.shape)
print(embeddings)