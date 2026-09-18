import numpy as np
from fastembed import TextEmbedding

class InMemoryRAGEngine:
    def __init__(self, model_name : str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model = TextEmbedding(model_name=model_name)
        self.documents = []

        self.embeddings = None

    def ingest_documents(self, docs : list[str]):
        self.documents = docs

        embedding_generator = self.embedding_model.embed(docs)
        self.embeddings = np.array(list(embedding_generator))

    def cosize_similarity(self, query_vec : np.ndarray, docs_vecs : np.ndarray) -> np.ndarray:
        dot_product = np.dot(docs_vecs,query_vec)

        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(docs_vecs,axis=1)
 
        return dot_product/ (query_norm * doc_norms +1e-10)

    def retrieve(self, query: str, top_k : int = 2) -> list[tuple[str,float]]:
        query_embedding = list(self.embedding_model.embed([query]))[0]

        scores = self.cosize_similarity(query_embedding, self.embeddings)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.documents[i], float(scores[i])) for i in top_indices]

        return results


engine = InMemoryRAGEngine()
engine.ingest_documents([
"FastEmbed provides high throughput embedding generation.",
"Vector databases allow semantic similarity search in memory.",
"Python makes implementing modular RAG systems straightforward."
])
print(engine.retrieve("Volleyball", top_k=1))