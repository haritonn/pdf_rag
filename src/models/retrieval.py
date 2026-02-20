from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document as LangChainDocument


@dataclass
class RetrievalResult:
    answer: str
    source_chunks: List[LangChainDocument]
