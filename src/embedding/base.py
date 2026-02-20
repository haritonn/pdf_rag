from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document as LangChainDocument


class Embedder(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[LangChainDocument]) -> List[List[float]]: ...
