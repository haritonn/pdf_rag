from .base import Embedder
from fastembed import TextEmbedding, SparseTextEmbedding


class FastEmbedEmbedder(Embedder):
    def __init__(self, dense_model, sparse_model):
        self.dense = TextEmbedding(dense_model)
        self.sparse = SparseTextEmbedding(sparse_model)

    def embed_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        return list(zip(self.dense.embed(texts), self.sparse.embed(texts)))

    def embed_query(self, query):
        return (
            list(self.dense.embed([query])[0]),
            list(self.sparse.embed([query])[0]),
        )
