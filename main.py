import sys
from pathlib import Path
from typing import Optional

from src.parsing.docling import DoclingParser
from src.chunking.langchain import LangChainChunker
from src.embedding.fastembed import FastEmbedEmbedder
from src.database.qdrant import QdrantVectorStore
from src.llm.ollama import OllamaProvider
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.retrieval import RetrievalPipeline
from src.utils.autodetect_ram import auto_select_model

# ── Конфиг ────────────────────────────────────────────────
COLLECTION_NAME = "pdf_rag"
VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
DB_PATH = ".qdrant_db"
# ──────────────────────────────────────────────────────────


def build_pipelines():
    model = auto_select_model()
    print(f"  модель: {model}")

    parser = DoclingParser()
    chunker = LangChainChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    embedder = FastEmbedEmbedder()
    store = QdrantVectorStore(COLLECTION_NAME, VECTOR_SIZE, path=DB_PATH)
    llm = OllamaProvider(model_name=model)

    ingestion = IngestionPipeline(parser, chunker, embedder, store)
    retrieval = RetrievalPipeline(embedder, store, llm, TOP_K)
    return ingestion, retrieval


def cmd_ingest(ingestion: IngestionPipeline):
    raw = input("Путь к файлу или папке: ").strip()
    path = Path(raw)

    if not path.exists():
        print(f"[!] Не найдено: {path}")
        return

    if path.is_file() and path.suffix.lower() == ".pdf":
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("*.pdf"))
        if not files:
            print("[!] PDF-файлы не найдены в папке")
            return
    else:
        print("[!] Укажите PDF-файл или папку")
        return

    print(f"  найдено файлов: {len(files)}")
    total_chunks = 0
    for i, f in enumerate(files, 1):
        try:
            n = ingestion.process_file(f)
            total_chunks += n
            print(f"  [{i}/{len(files)}] {f.name} → {n} чанков")
        except Exception as e:
            print(f"  [{i}/{len(files)}] {f.name} → ошибка: {e}")

    print(f"\n  готово. всего чанков: {total_chunks}")


def cmd_query(retrieval: RetrievalPipeline):
    print("Режим вопросов (exit — выход)\n")
    while True:
        try:
            question = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        print()
        try:
            stream, chunks = retrieval.stream(question)
            for token in stream:
                print(token, end="", flush=True)
            print("\n")

            if chunks:
                print("─" * 40)
                print("Источники:")
                seen = set()
                for i, chunk in enumerate(chunks, 1):
                    src = chunk.metadata.get("source", "—")
                    page = chunk.metadata.get("page", "?")
                    key = (src, page)
                    if key not in seen:
                        seen.add(key)
                        print(f"  [{i}] {Path(src).name}, стр. {page}")
                print()

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[!] Ошибка: {e}\n")


def print_menu():
    print("\n┌─ PDF RAG ───────────────────────┐")
    print("│  1. Индексировать PDF           │")
    print("│  2. Задать вопрос               │")
    print("│  0. Выход                       │")
    print("└─────────────────────────────────┘")


def main():
    print("Инициализация...")
    try:
        ingestion, retrieval = build_pipelines()
    except Exception as e:
        print(f"[!] Ошибка инициализации: {e}")
        sys.exit(1)

    while True:
        print_menu()
        try:
            choice = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if choice == "1":
            cmd_ingest(ingestion)
        elif choice == "2":
            cmd_query(retrieval)
        elif choice == "0":
            print("Пока!")
            break
        else:
            print("[!] Неизвестная команда")


if __name__ == "__main__":
    main()
