from abc import ABC, abstractmethod
from collections import List
from langchain_core.document import Document as LangChainDocument


class EmbeddingClass(ABC):
    @abstractmethod
    def embed_chunks(self, chunks: List[LangChainDocument]) -> List[List[float]]:
        pass
