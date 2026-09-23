from dataclasses import dataclass

from langchain_core.documents import Document as LangChainDocument


@dataclass
class RetrievalResult:
    answer: str
    source_chunks: list[LangChainDocument]
