from .base import Embedder
from fastembed import TextEmbedding, SparseTextEmbedding


class FastEmbedEmbedder(Embedder):
    def __init__(
        self,
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
    ):
        self.dense = TextEmbedding(dense_model)
        self.sparse = SparseTextEmbedding(sparse_model)

    def embed_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        return list(zip(dense, sparse))

    def embed_query(self, query: str):
        dense = list(self.dense.embed([query]))[0]
        sparse = list(self.sparse.embed([query]))[0]
        return dense, sparse
