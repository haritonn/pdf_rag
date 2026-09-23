from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document as LangChainDocument

from ..embedding.base import Embedding


class VectorStore(Protocol):
    def add_documents(
        self, chunks: Sequence[LangChainDocument], embeddings: Sequence[Embedding]
    ) -> None: ...

    def search_up(
        self, query_vector: Embedding, top_k: int, paper_id: str | None = None
    ) -> list[LangChainDocument]: ...

    def collection_exists(self, collection_name: str) -> bool: ...
