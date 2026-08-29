from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if hasattr(self.model, "encode_document"):
            self.embeddings = self.model.encode_document(
                chunks,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        else:
            self.embeddings = self.model.encode(
                chunks,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

    def search(self,query,k=3):
        if hasattr(self.model, "encode_query"):
            query_embedding = self.model.encode_query(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        else:
            query_embedding = self.model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        hits = semantic_search(
            query_embedding,
            self.embeddings,
            top_k=min(k,len(self.chunks))
        )[0]

        return [
            {
                "text" : self.chunks[hit["corpus_id"]],
                "score" : float(hit["score"]),
                "index" : hit["corpus_id"]
            }
            for hit in hits
        ]

if __name__ == "__main__":
    chunks = [
    "Python is commonly used for backend programming.",
    "React is used to build user interfaces.",
    "RAG retrieves relevant information before generation.",
    ]
    retriever = Retriever(chunks)
    print(retriever.search("Frontend", 2))