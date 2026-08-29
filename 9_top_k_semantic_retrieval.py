import numpy as np

def top_k_retrieve(query_embedding, document_embeddings, documents, k = 3):
    if k <= 0: 
        return []

    query = np.asarray(query_embedding,dtype=float)
    docs = np.asarray(document_embeddings, dtype=float)

    if len(documents) != len(docs):
        raise ValueError("documents and embeddings length must match")

    query_norm = np.linalg.norm(query)
    docs_norms = np.linalg.norm(docs, axis=1) #axis=1 means calculate norm of each doc

    if query_norm == 0 or np.any(docs_norms == 0):
        raise ValueError("Zero vectors are not supported")

    scores = (docs @ query) / (docs_norms * query_norm) # @ means dot product between doc1 . query , doc2 . query , etc...

    k = min(k, len(documents))
    indices = np.argsort(scores)[::-1][:k] #-1 reverses it from largest to smallest.

    return [
        {
            "index" : int(i),
            "text" : documents[i],
            "score" : float(scores[i])
        }
        for i in indices
    ]


if __name__ == "__main__":
    documents = [
        "Python is a programming language.",
        "React is a frontend library.",
        "RAG retrieves relevant context.",
        "Vector databases store embeddings."
    ]

    document_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.2, 0.8, 0.0]
    ])

    query_embedding = np.array([1.0, 0.0, 0.0])

    results = top_k_retrieve(
        query_embedding,
        document_embeddings,
        documents,
        k=2
    )

    for result in results:
        print(result)