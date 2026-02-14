from abc import ABC, abstractmethod
from ..models.document import Chunk, Document
from typing import List


class TextChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: Document) -> List[Chunk]:
        pass
