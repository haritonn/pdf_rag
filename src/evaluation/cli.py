"""Command-line entry point for QASPER retrieval experiments."""

import argparse
from collections.abc import Sequence

from ..database.qdrant import QdrantVectorStore
from ..embedding.fastembed import FastEmbedEmbedder
from .generation import OllamaProvider
from .index import build_index
from .qasper import load_qasper, parse_qasper_paper
from .retrieval import RetrievalPipeline


DEFAULT_COLLECTION = "qasper_hybrid"
DEFAULT_DB_PATH = ".qdrant_db"
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def _add_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--vector-size", type=int, default=384)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run QASPER hybrid retrieval with Qdrant RRF and Ollama."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-index", help="Build the QASPER index.")
    _add_index_arguments(build)
    build.add_argument("--limit", type=int, help="Index only the first N papers.")
    build.add_argument("--batch-size", type=int, default=64)
    build.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    query = commands.add_parser("query", help="Answer a question using the index.")
    _add_index_arguments(query)
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    query.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    query.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    query.add_argument(
        "--show-sources",
        action="store_true",
        help="Print metadata of retrieved chunks.",
    )
    return parser


def _make_embedder(args: argparse.Namespace) -> FastEmbedEmbedder:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    return FastEmbedEmbedder(
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        providers=providers,
    )


def _make_store(args: argparse.Namespace) -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=args.collection,
        vector_size=args.vector_size,
        path=args.db_path,
    )

def run_build_index(args: argparse.Namespace) -> None:
    rows = load_qasper()
    if args.limit is not None:
        rows = rows.select(range(min(args.limit, len(rows))))

    papers = (parse_qasper_paper(row) for row in rows)
    count = build_index(
        papers,
        _make_embedder(args),
        _make_store(args),
        total_papers=len(rows),
        batch_size=args.batch_size,
    )
    print(f"Indexed chunks: {count}")


def run_query(args: argparse.Namespace) -> None:
    embedder = _make_embedder(args)
    store = _make_store(args)
    llm = OllamaProvider(
        model_name=args.ollama_model,
        host=args.ollama_host,
    )
    pipeline = RetrievalPipeline(
        embed=embedder,
        store=store,
        llm=llm,
        top_k=args.top_k,
    )

    result = pipeline.query(args.question)
    print(result.answer)

    if args.show_sources:
        print("\nSources:")
        for chunk in result.source_chunks:
            print(chunk.metadata)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "build-index":
        run_build_index(args)
    elif args.command == "query":
        run_query(args)


if __name__ == "__main__":
    main()
