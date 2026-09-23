
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document

from src.evaluation.retrieval import (
    RetrievalPipeline,
)


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, query: str) -> Any:
        self.queries.append(query)
        return 'query-vector'

    def embed_chunks(self, chunks: Any) -> list[Any]:
        return []


class FakeStore:
    def __init__(self):
        self.calls = []

    def search_up(
        self,
        query_vector: Any,
        top_k: int,
        paper_id: str | None = None,
    ) -> list[Document]:
        self.calls.append((query_vector, top_k))

        return [
            Document(page_content="first context"),
            Document(page_content="second context"),
        ]

    def add_documents(self, chunks: Any, embeddings: Any) -> None:
        pass

    def collection_exists(self, collection_name: str) -> bool:
        return True


class FakeLLM:
    def __init__(self):
        self.calls = []

    def stream(self, prompt, context_docs) -> Iterator[str]:
        self.calls.append((prompt, context_docs))

        yield "generated "
        yield "answer"

def test_pipe_got_query():
    embedder = FakeEmbedder()
    store = FakeStore()
    llm = FakeLLM()
    pipe = RetrievalPipeline(embedder, store, llm, top_k=2)
    result = pipe.query('мне не очень нравится писать тесты')
    assert result.answer == 'generated answer'
    assert embedder.queries == ['мне не очень нравится писать тесты']
    assert store.calls == [
              ("query-vector", 2)
          ]

    assert llm.calls == [
        (
            "мне не очень нравится писать тесты",
            ["first context", "second context"],
        )
    ]
    assert len(result.source_chunks) == 2
