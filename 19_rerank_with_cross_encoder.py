from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import semantic_search

chunks = [
"RAG retrieves documents and supplies them to an LLM.",
"React is a frontend UI library.",
"Embeddings represent semantic meaning as vectors.",
"A vector database stores and retrieves embeddings.",
]

bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

query = "How does RAG retrieve information for a model?"

embeddings = bi_encoder.encode(
    chunks,
    convert_to_tensor=True,
    normalize_embeddings=True
)

query_embedding = bi_encoder.encode(
    query,
    convert_to_tensor=True,
    normalize_embeddings=True
)

hits = semantic_search(
    query_embedding,
    embeddings,
    top_k=min(3,len(chunks))
)[0]

candidates = [chunks[h["corpus_id"]] for h in hits]

pairs = [[query,text] for text in candidates]
scores = cross_encoder.predict(pairs)

ranked = sorted(
    zip(candidates, scores),
    key=lambda x : float(x[1]),
    reverse=True
)


for text , score in ranked:
    print(f"{float(score) :.4f} | {text}")