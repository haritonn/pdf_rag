from abc import ABC, abstractmethod
from ..models.document import Document
from langchain_core.documents import Document as LangChainDocument
from typing import List


class TextChunker(ABC):
    @abstractmethod
    def chunk_document(self, doc: Document) -> List[LangChainDocument]: ...
