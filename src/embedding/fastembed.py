from .base import Embedder
from fastembed import TextEmbedding


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name):
        self.model = TextEmbedding(model_name)

    def embed_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        return list(self.model.embed(texts))

    def embed_query(self, query):
        return list(self.model.embed(query))[0]
