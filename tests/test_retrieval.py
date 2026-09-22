import pytest

from langchain_core.documents import Document
from src.evaluation.retrieval import (
    MAX_CONTEXT_CHARS,
    RetrievalPipeline,
)

class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return 'query-vector'


class FakeStore:
    def __init__(self):
        self.calls = []

    def search_up(self, query_vector, top_k):
        self.calls.append((query_vector, top_k))

        return [
            Document(page_content="first context"),
            Document(page_content="second context"),
        ]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def stream(self, prompt, context_docs):
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
