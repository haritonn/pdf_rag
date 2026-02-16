from abc import ABC, abstractmethod
from ..models.document import Chunk, Document
from typing import List


class TextChunker(ABC):
    @abstractmethod
    def chunk_document(self, doc: Document) -> List[Chunk]:
        pass
