from fastembed import TextEmbedding

class FastEmbedIndexer:
    def __init__(self, model_name : str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)
        self.store = []

    def add_documents(self, documents : list[str]):
        embeddings = list(self.model.embed(documents))

        for doc, emb in zip(documents, embeddings):
            self.store.append({
                "text" : doc,
                "embedding" : emb
            })

indexer = FastEmbedIndexer()

indexer.add_documents(["RAG enhances LLM outputs.", "FastEmbed is lightweight and fast."])
print(indexer.store)