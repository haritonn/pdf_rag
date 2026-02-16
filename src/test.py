from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.parsing.docling import DoclingParser
from src.chunking.llama import LlamaIndexChunker


parser = DoclingParser(extract_images=False, extract_tables=True)
doc = parser.parse_file(Path("test.pdf"))

print("PARSED ELEMENTS:")

for i, elem in enumerate(doc.elements):
    print(f"\nElement {i}:")
    print(f"  Type: {elem.type}")
    print(f"  Metadata: {elem.metadata}")
    print(f"  Content:\n{elem.content[:500]}")
    print("-" * 60)

chunker = LlamaIndexChunker(chunk_size=1024, chunk_overlap=100)
chunks = chunker.chunk_document(doc)

print("\n" + "=" * 60)
print("CHUNKS:")
print("=" * 60)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Text:\n{chunk.text}")
    print("-" * 60)
