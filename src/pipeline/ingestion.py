from pathlib import Path
from typing import List, Dict
from ..chunking.base import TextChunker
from ..parsing.base import DocumentParser
from ..database.base import VectorStore
from ..embedding.base import Embedder


class IngestionPipeline:
    """Pipeline from document(-s) to the embeddings (with vector db)"""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: TextChunker,
        embed: Embedder,
        vector_store: VectorStore,
    ):
        self._parser = parser
        self._chunker = chunker
        self._embed = embed
        self._vector_store = vector_store

    def process_file(self, file_path: Path) -> int:
        document = self._parser.parse_file(file_path)
        chunks = self._chunker.chunk_document(document)
        embeddings = self._embed.embed_chunks(chunks)
        self._vector_storage.add_documents(chunks, embeddings)
        return len(chunks)

    def process_batch(self, file_paths: List[Path]) -> Dict[str, int]:
        return {str(p): self.ingest_file(p) for p in file_paths}
