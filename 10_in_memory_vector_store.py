import numpy as np

class VectorStore:
    def __init__(self):
        self.records = []

    def add(self, id, text, embedding, metadata = None):
        self.records.append({
            "id" : id,
            "text" : text,
            "embedding" : np.asarray(embedding, dtype=float),
            "metadata" : metadata or {}
        })

    def search(self, query_embedding, k):
        if not self.records:
            return []

        query = np.asarray(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("query embedding cannot be zero")

        scores = []

        for record in self.records:
            emb = record["embedding"]
            emb_norm = np.linalg.norm(emb)

            if emb_norm == 0:
                continue

            score = float(np.dot(query,emb) /(query_norm * emb_norm))
            scores.append({
                "id" : record["id"],
                "text" : record["text"],
                "metadata" : record["metadata"],
                "score" : score
            })

        scores.sort(key = lambda x : x["score"], reverse=True)
        return scores[:k]

    def count(self, ):
        return len(self.records)

if __name__ == "__main__":

    store = VectorStore()

    store.add(
        "doc1",
        "Python is a programming language.",
        [1.0, 0.0, 0.0],
        {"source": "python.txt"}
    )

    store.add(
        "doc2",
        "React is a frontend library.",
        [0.0, 1.0, 0.0],
        {"source": "react.txt"}
    )

    store.add(
        "doc3",
        "RAG retrieves relevant context.",
        [0.9, 0.1, 0.0],
        {"source": "rag.txt"}
    )

    print("Number of documents:", store.count())

    query = [1.0, 0.0, 0.0]

    results = store.search(query, k=2)

    for result in results:
        print(result)