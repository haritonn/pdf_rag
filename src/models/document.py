from dataclasses import dataclass
from typing import List, Dict, Literal


@dataclass
class Element:
    """Element of content with specified metadata"""

    type: Literal["text", "code", "table", "formula"]
    content: str
    metadata: Dict


@dataclass
class Document:
    """Parsed document"""

    elements: List[Element]
    metadata: Dict


@dataclass
class Chunk:
    """Chunk of document"""

    text: str
    metadata: Dict
