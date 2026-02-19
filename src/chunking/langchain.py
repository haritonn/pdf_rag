from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models.document import Document
from .base import TextChunker


class LangChainChunker(TextChunker):

    def __init__(self, chunk_size, chunk_overlap):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
        )

    def chunk_document(self, doc: Document) -> list[LangChainDocument]:
        chunks = []
        chunk_id = 0

        for elem in doc.elements:
            if not elem.content.strip():
                continue

            base_meta = {
                **doc.metadata,
                **elem.metadata,
                "element_type": elem.type,
                "chunk_id": chunk_id,
            }

            if elem.type in ("table", "formula", "code"):
                chunks.append(LangChainDocument(page_content=elem.content, metadata=base_meta))
                chunk_id += 1
                continue

            for text in self.splitter.split_text(elem.content):
                chunks.append(LangChainDocument(
                    page_content=text,
                    metadata={**base_meta, "chunk_id": chunk_id},
                ))
                chunk_id += 1

        return chunks

