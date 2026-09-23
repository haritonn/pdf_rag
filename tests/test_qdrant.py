import numpy as np
import pytest
from langchain_core.documents import Document

from src.database.qdrant import QdrantVectorStore


class FakeSparseVector:
    def __init__(self, indices, values):
        self.indices = np.asarray(indices)
        self.values = np.asarray(values)

@pytest.mark.integration
def test_integration_qdrant(tmp_path):
    client = QdrantVectorStore(
        collection_name="qdrant_test",
        vector_size=2,
        path=str(tmp_path / "qdrant"),
    )

    dense = np.array([1, 3])
    sparse = FakeSparseVector(indices=[1, 3], values=[1.0, 0.5])
    document = Document(page_content="retrieval doc", metadata={"paper_id": "paper-1", "section": "intro"})

    try:
        client.add_documents(
            [document],
            [(dense, sparse)],
        )
        results = client.search_up(
            (dense, sparse),
            top_k=1
        )
        assert len(results) == 1
        assert results[0].page_content == "retrieval doc"
        assert results[0].metadata["paper_id"] == "paper-1"
        assert results[0].metadata["section"] == "intro"
        assert isinstance(results[0].metadata["_score"], float)

    finally:
        client.client.close()
