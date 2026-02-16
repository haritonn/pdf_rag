import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.parsing.docling import DoclingParser
from src.chunking.llama import LlamaIndexChunker
from src.pipeline.ingestion import IngestionPipeline


def test_pipeline():
    """Test on sample .pdf file"""
    parser = DoclingParser(extract_images=False, extract_tables=True)
    chunker = LlamaIndexChunker(chunk_size=1024, chunk_overlap=100)
    pipeline = IngestionPipeline(parser=parser, chunker=chunker)

    file_path = Path("test.pdf")
    chunks = pipeline.process_file(file_path)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk.text[:200]}...")


if __name__ == "__main__":
    test_pipeline()
