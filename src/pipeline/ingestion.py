from ..models.document import Chunk
from pathlib import Path
from typing import List
from ..chunking.base import TextChunker
from ..parsing.base import DocumentParser


class IngestionPipeline:
    """Pipeline from document(-s) to the list of chunks"""

    def __init__(self, parser: DocumentParser, chunker: TextChunker):
        self._parser = parser
        self._chunker = chunker

    def process_file(self, file_path: Path) -> List[Chunk]:
        document = self._parser.parse_file(file_path)
        chunks = self._chunker.chunk_text(document)

        return chunks

    def preprocess_batch(self, file_paths: List[Path]) -> List[Chunk]:
        all_chunks = []
        for path in file_paths:
            chunks = self.process_file(path)

            all_chunks.extend(chunks)

        return all_chunks
