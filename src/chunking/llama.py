from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from ..models.document import Chunk
from .base import TextChunker


class LlamaIndexChunker(TextChunker):
    def __init__(self, chunk_size, chunk_overlap):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, doc):
        full_text = "\n\n".join(elem.content for elem in doc.elements)

        llama_doc = LlamaDocument(
            text=full_text,
            metadata=doc.metadata,
        )

        nodes = self.splitter.get_nodes_from_documents([llama_doc])

        chunks = []
        for i, node in enumerate(nodes):
            chunks.append(
                Chunk(
                    text=node.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "node_id": node.node_id,
                    },
                    chunk=str(i),
                )
            )

        return chunks
