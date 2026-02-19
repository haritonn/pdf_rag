from abc import ABC, abstractmethod
from langchain_document.core import Document as LangChainDocument
from collections import List


class VectorStore(ABC):
    @abstractmethod
    def add_documents(
        self, chunks: List[LangChainDocument], ebmeddings: List[List[float]]
    ) -> None: ...

    @abstractmethod
    def search_up(
        self, query_vector: List[float], top_k: int
    ) -> List[LangChainDocument]: ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool: ...
