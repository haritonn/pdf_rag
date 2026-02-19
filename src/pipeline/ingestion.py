from pathlib import Path
from typing import List
from ..chunking.base import TextChunker
from ..parsing.base import DocumentParser
from langchain_core.documents import Document as LangChainDocument


class IngestionPipeline:
    """Pipeline from document(-s) to the list of chunks"""

    def __init__(self, parser: DocumentParser, chunker: TextChunker):
        self._parser = parser
        self._chunker = chunker

    def process_file(self, file_path: Path) -> List[LangChainDocument]:
        document = self._parser.parse_file(file_path)
        return self._chunker.chunk_document(document)

    def preprocess_batch(self, file_paths: List[Path]) -> List[LangChainDocument]:
        all_chunks = []
        for path in file_paths:
            all_chunks.extend(self.process_file(path))

        return all_chunks
