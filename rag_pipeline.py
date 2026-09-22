import numpy as np
from typing import List
from fastembed import TextEmbedding

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
documents = []
vectors = []

def load_document(filepath : str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    docs = content.split("\n\n")
    return [doc.strip() for doc in docs if doc.strip()]

def preprocess_docs(docs : List[str]) -> List[str]:
    return [doc.lower().strip() for doc in docs]

def split_documents(docs : List[str], chunk_size : int, overlap : int) -> List[str]:
    chunks = []

    for doc in docs:
        words = doc.split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start,end])
            chunks.append(chunk)

            start = end - overlap

    return chunks

def embed_documents(docs : List[str]) -> List[List[float]]:
    embeddings = embedding_model.embed(docs)

    return [embedding.tolist() for embedding in embeddings]

def embed_query(query : str) -> List[float]:
    embedding = next(embedding_model.embed([query]))

    return embedding.tolist()

def add_documents(docs : List[str], vs : List[List[float]]):
    documents.extend(docs)
    vectors.extend(vs)

def similarity_search(query_vector, k):
    query = np.array(query_vector, dtype=np.float32)
    query = (query/np.linalg.norm(query))

    document_vectors = np.array(vectors,dtype=np.float32)
    document_vectors = (vectors/ np.linalg.norm(document_vectors, axis=1, keepdims=True))

    scores = np.dot(query,document_vectors)

    indices = np.argsort(scores)[::-1]

    return [documents[i] for i in indices[:k]]



