from langchain_core.documents import Document as LangChainDocument
from typing import List

from ..database.base import VectorStore
from ..embedding.base import Embedder
from ..llm.base import LLMProvider
from ..models.retrieval import RetrievalResult


class RetrievalPipeline:
    """Pipeline from chunks to LLM provided answer"""

    def __init__(
        self, embed: Embedder, store: VectorStore, llm: LLMProvider, top_k: int
    ):
        self._embed = embed
        self._vector_store = store
        self._llm = llm
        self.top_k = top_k

    def _get_context_docs(self, query: str) -> List[LangChainDocument]:
        query_vector = self._embed.embed_query(query)
        return self._vector_store.search_up(query_vector, self.top_k)

    def query(self, question: str) -> RetrievalResult:
        chunks = self._get_context_docs(question)
        text_contents = [c.page_content for c in chunks]
        answer = self._llm.generate(question, text_contents)
        return RetrievalResult(answer=answer, source_chunks=chunks)

    def stream(self, question: str):
        chunks = self._get_context_docs(question)
        text_contents = [c.page_content for c in chunks]
        return self._llm.stream(question, text_contents), chunks
