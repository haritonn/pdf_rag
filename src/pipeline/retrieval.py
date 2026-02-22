from ..database.base import VectorStore
from ..embedding.base import Embedder
from ..llm.base import LLMProvider
from ..models.retrieval import RetrievalResult

MAX_CONTEXT_CHARS = 6000


class RetrievalPipeline:
    """Pipeline from chunks to LLM provided answer"""

    def __init__(
        self, embed: Embedder, store: VectorStore, llm: LLMProvider, top_k: int
    ):
        self._embed = embed
        self._vector_store = store
        self._llm = llm
        self.top_k = top_k

    def _get_context_docs(self, query):
        query_vector = self._embed.embed_query(query)
        return self._vector_store.search_up(query_vector, self.top_k)

    def _trim_context(self, docs):
        result, total = [], 0
        for doc in docs:
            if total + len(doc.page_content) > MAX_CONTEXT_CHARS:
                break
            result.append(doc)
            total += len(doc.page_content)
        return result

    def stream(self, question):
        chunks = self._trim_context(self._get_context_docs(question))
        text_contents = [c.page_content for c in chunks]
        return self._llm.stream(question, text_contents), chunks

    def query(self, question):
        stream, chunks = self.stream(question)
        answer = "".join(stream)
        return RetrievalResult(answer=answer, source_chunks=chunks)
