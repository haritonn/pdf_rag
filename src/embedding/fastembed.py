from .base import EmbeddingClass
from fastembed import TextEmbedding


class FastEmbedEmbedder(EmbeddingClass):
    def __init__(self, model_name):
        self.model = TextEmbedding(model_name)

    def embed_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        return list(self.model.embed(texts))
