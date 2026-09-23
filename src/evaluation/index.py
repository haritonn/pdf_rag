"""Conversion of QASPER papers into indexed retrieval documents."""

from collections.abc import Iterable

from langchain_core.documents import Document
from tqdm.auto import tqdm

from ..database.base import VectorStore
from ..embedding.base import Embedder
from ..models.qasper import QasperPaper


def paper_to_documents(paper: QasperPaper) -> list[Document]:
    documents = []

    for section_idx, section in enumerate(paper.sections):
        for paragraph_idx, paragraph in enumerate(section.paragraphs):
            if not paragraph.strip():
                continue

            documents.append(
                Document(
                    page_content=paragraph,
                    metadata={
                        "paper_id": paper.paper_id,
                        "section": section.name,
                        "section_idx": section_idx,
                        "paragraph_idx": paragraph_idx,
                        "doc_id": (
                            f"{paper.paper_id}:"
                            f"{section_idx}:"
                            f"{paragraph_idx}"
                        )
                    },
                )
            )

    return documents


def build_index(
    papers: Iterable[QasperPaper],
    embedder: Embedder,
    store: VectorStore,
    *,
    total_papers: int | None = None,
    batch_size: int = 64,
) -> int:
    pending: list[Document] = []
    indexed_chunks = 0

    with tqdm(
        papers,
        total=total_papers,
        desc="Indexing papers",
        unit="paper",
    ) as progress:
        for paper in progress:
            pending.extend(paper_to_documents(paper))

            while len(pending) >= batch_size:
                batch = pending[:batch_size]
                del pending[:batch_size]
                store.add_documents(batch, embedder.embed_chunks(batch))
                indexed_chunks += len(batch)
                progress.set_postfix(chunks=indexed_chunks)

    if pending:
        store.add_documents(pending, embedder.embed_chunks(pending))
        indexed_chunks += len(pending)

    return indexed_chunks
