import uuid

from langchain_core.documents import Document as LangChainDocument
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self, collection_name, vector_size, path=".qdrant_db", distance=Distance.COSINE
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

    def collection_exists(self, collection_name):
        existing_cols = [c.name for c in self.client.get_collections().collections]
        return collection_name in existing_cols

    def ensure_collection(self, collection_name: str) -> None:
        """Creating collection if it doesnt exist"""
        if not self.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size, distance=self.distance
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams()),
                },
            )

    def add_documents(self, chunks, embeddings):
        self.ensure_collection(self.collection_name)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense.tolist(),
                    "sparse": SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload={"data": chunk.page_content, **chunk.metadata},
            )
            for chunk, (dense, sparse) in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_up(self, query_vector, top_k, paper_id = None):
        query_filter = None
        if paper_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            )
        dense_vec, sparse_emb = query_vector
        hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
                Prefetch(query=dense_vec.tolist(), using="dense", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
        ).points

        result = []
        for hit in hits:
            if hit.payload is None: continue
            payload = hit.payload
            data = payload.get('data')
            if not isinstance(data, str): continue
            metadata = {
                k: v for k, v in payload.items()
                if k != "data"
            }
            metadata["_score"] = float(hit.score)
            page_content = hit.payload["data"]
            result.append(LangChainDocument(
                page_content=page_content,
                metadata=metadata,
            ))

        return result
