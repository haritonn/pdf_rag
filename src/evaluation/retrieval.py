from collections.abc import Iterator
from typing import Protocol

from ..database.base import VectorStore
from ..embedding.base import Embedder
from ..models.retrieval import RetrievalResult
from ..reranking.base import Reranker

MAX_CONTEXT_CHARS = 6000


class StreamingGenerator(Protocol):
    def stream(self, prompt: str, context_docs: list[str]) -> Iterator[str]: ...


class RetrievalPipeline:
    """Pipeline from chunks to an LLM-provided answer."""

    def __init__(
        self,
        embed: Embedder,
        store: VectorStore,
        llm: StreamingGenerator,
        top_k: int,
        reranker: Reranker | None = None,
        candidate_k: int | None = None,
    ):
        self._embed = embed
        self._vector_store = store
        self._llm = llm
        self.top_k = top_k
        self._reranker = reranker
        self.candidate_k = candidate_k or max(50, top_k * 5)

    def _get_context_docs(self, query, paper_id=None):
        query_vector = self._embed.embed_query(query)
        if paper_id is None:
            documents = self._vector_store.search_up(query_vector, self.candidate_k)
        else:
            documents = self._vector_store.search_up(
                query_vector,
                self.candidate_k,
                paper_id=paper_id,
            )

        if self._reranker is not None:
            return self._reranker.rerank(query, documents, self.top_k)
        return documents[: self.top_k]

    def _trim_context(self, docs):
        result, total = [], 0
        for doc in docs:
            if total + len(doc.page_content) > MAX_CONTEXT_CHARS:
                break
            result.append(doc)
            total += len(doc.page_content)
        return result

    def stream(self, question, paper_id=None):
        chunks = self._trim_context(self._get_context_docs(question, paper_id))
        text_contents = [chunk.page_content for chunk in chunks]
        return self._llm.stream(question, text_contents), chunks

    def query(self, question, paper_id=None):
        stream, chunks = self.stream(question, paper_id)
        answer = "".join(stream)
        return RetrievalResult(answer=answer, source_chunks=chunks)
