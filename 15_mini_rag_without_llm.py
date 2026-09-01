import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

def load_document(path):
    with open(path, "r", encoding= 'utf-8') as file:
        return file.read()

def preprocess(text):
    return re.sub(r"\s+"," ", text.lower()).strip()

def chunk_text(text, chunk_size = 250, overlap=40):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    step = chunk_size-overlap

    return [text[i : i+chunk_size] for i in range(0,len(text),step) if text[i:i+chunk_size]]

text = preprocess(load_document("data/sample.txt"))
chunks = chunk_text(text)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunk_embeddings = model.encode(
    chunks,
    convert_to_tensor=True,
    normalize_embeddings=True
)

query = input("Ask a question : ")

query_embedding = model.encode(
    query,
    convert_to_tensor=True,
    normalize_embeddings=True
)

hits = semantic_search(
    query_embedding,
    chunk_embeddings,
    top_k=min(3,len(chunks))
)[0]

print("\n Retrieved context : ")
for hit in hits:
    print(f"\nScore: {hit['score']:.4f}" )
    print(chunks[hit["corpus_id"]])