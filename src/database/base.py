from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document as LangChainDocument


class VectorStore(ABC):
    @abstractmethod
    def add_documents(
        self, chunks: List[LangChainDocument], embeddings: List[List[float]]
    ) -> None: ...

    @abstractmethod
    def search_up(
        self, query_vector: List[float], top_k: int
    ) -> List[LangChainDocument]: ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool: ...
