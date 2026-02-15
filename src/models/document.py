from dataclasses import dataclass
from typing import List, Dict, Literal, Optional


@dataclass
class Element:
    """Element of content with specified metadata"""

    type: Literal["text", "code", "table", "formula", "image"]
    content: str
    metadata: Dict


@dataclass
class Document:
    """Parsed document"""

    elements: List[Element]
    metadata: Dict

    def get_content(self):
        return ("\n\n".join(elem.content) for elem in self.elements)

    def get_content_by_type(self, type):
        return [elem for elem in self.elements if elem.type == type]


@dataclass
class Chunk:
    """Chunk of document"""

    text: str
    metadata: Dict
    chunk: Optional[str]
