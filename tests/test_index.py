import pytest

from src.evaluation.index import build_index, paper_to_documents
from src.evaluation.qasper import parse_qasper_paper

def test_paper_to_documents(raw_qasper_paper):
    paper = parse_qasper_paper(raw_qasper_paper)
    documents = paper_to_documents(paper)

    assert len(documents) == 2
    assert documents[0].page_content == "First paragraph."
    assert documents[0].metadata == {
        "paper_id": "paper-1",
        "section": "Introduction",
        "section_idx": 0,
        "paragraph_idx": 0,
    }

def test_empty_paragraphs_skipping(raw_qasper_paper):
    raw_qasper_paper['full_text']['paragraphs'][0].append("")
    paper = parse_qasper_paper(raw_qasper_paper)
    documents = paper_to_documents(paper)

    assert len(documents) == 2


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_chunks(self, chunks):
        self.calls.append(chunks)
        return [
            (FakeDense(), FakeSparse())
            for _ in chunks
        ]

class FakeStore:
    def __init__(self):
        self.calls = []

    def add_documents(self, chunks, embeddings):
        self.calls.append((chunks, embeddings))

class FakeDense:
    def tolist(self):
        return [0.1, 0.2]

class FakeArray(list):
    def tolist(self):
        return list(self)

class FakeSparse:
    indices = [1]
    values = [0.5]

    def __init__(self):
        self.indices = FakeArray(self.indices)
        self.values = FakeArray(self.values)


def test_build_index_batches(raw_qasper_paper):
    from src.evaluation.qasper import parse_qasper_paper

    paper = parse_qasper_paper(raw_qasper_paper)
    embedder = FakeEmbedder()
    store = FakeStore()

    count = build_index(
        [paper],
        embedder,
        store,
        batch_size=1,
    )

    assert count == 2
    assert len(store.calls) == 2
    assert len(embedder.calls) == 2
