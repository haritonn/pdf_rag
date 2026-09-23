from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias

from langchain_core.documents import Document as LangChainDocument


class DenseVector(Protocol):
    def tolist(self) -> Any: ...

class SparseVector(Protocol):
    indices: Any
    values: Any

Embedding: TypeAlias = tuple[DenseVector, SparseVector]

class Embedder(Protocol):
    def embed_chunks(
        self, chunks: Sequence[LangChainDocument]
    ) -> list[Embedding]: ...

    def embed_query(self, query: str) -> Embedding: ...
