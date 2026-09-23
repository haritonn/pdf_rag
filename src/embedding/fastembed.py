from collections.abc import Sequence
from typing import Any, cast

from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_core.documents import Document

from .base import Embedder, Embedding


class FastEmbedEmbedder(Embedder):
    def __init__(
        self,
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        providers=None,
    ):
        self.providers = providers or ["CPUExecutionProvider"]
        self.dense = TextEmbedding(dense_model, providers=self.providers)
        self.sparse = SparseTextEmbedding(sparse_model, providers=self.providers)

        self._check_providers()

    def _check_providers(self) -> None:
        requested = set(self.providers)
        if "CUDAExecutionProvider" not in requested:
            return

        dense_model = cast(Any, self.dense)
        sparse_model = cast(Any, self.sparse)

        sessions = (
            dense_model.model.model,
            sparse_model.model.model,
        )
        actual = [set(session.get_providers()) for session in sessions]
        if any("CUDAExecutionProvider" not in providers for providers in actual):
            raise RuntimeError(
                "CUDA was requested for FastEmbed, but the CUDA execution "
                "provider was not activated by one of the ONNX sessions. "
                "Check nvidia-smi and install a CUDA-compatible "
                "onnxruntime-gpu build."
            )

    def embed_chunks(self, chunks: Sequence[Document]) -> list[Embedding]:
        texts = [c.page_content for c in chunks]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        return cast(list[Embedding], list(zip(dense, sparse)))

    def embed_query(self, query: str) -> Embedding:
        dense = list(self.dense.embed([query]))[0]
        sparse = list(self.sparse.embed([query]))[0]
        return cast(Embedding, (dense, sparse))
