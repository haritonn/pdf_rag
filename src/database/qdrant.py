import uuid
from .base import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.document import Document as LangChainDocument


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
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

    def add_document(self, chunks, embeddings):
        self.ensure_collection()

        points = [
            PointStruct(
                id=str(uuid.uuid64()),
                vector=vec,
                payload={
                    "data": chunk.page_content,
                    **chunk.metadata,
                },
            )
            for chunk, vec in enumerate(chunks, embeddings)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_vector, top_k):
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            LangChainDocument(
                page_content=hit.payload.pop("text"),
                metadata=hit.payload,
            )
            for hit in hits
        ]
