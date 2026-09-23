from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document


class Reranker(Protocol):
    def rerank( self, query: str, documents: Sequence[Document], top_k: int) -> list[Document]: ...
