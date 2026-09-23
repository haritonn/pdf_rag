"""Retrieval evaluation helpers for QASPER and ranx."""

from collections.abc import Iterable

from ranx import Qrels, Run, evaluate

from ..database.base import VectorStore
from ..embedding.base import Embedder
from ..models.qasper import QasperPaper, QasperQuestion
from .index import paper_to_documents


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def query_id(paper_id: str, question_idx: int) -> str:
    return f"{paper_id}:q{question_idx}"


def evidence_matches(evidence: str, paragraph: str) -> bool:
    evidence = normalize_text(evidence)
    paragraph = normalize_text(paragraph)
    return bool(evidence) and (evidence in paragraph or paragraph in evidence)


def question_qrels(
    paper: QasperPaper,
    question_idx: int,
    question: QasperQuestion,
) -> tuple[str, dict[str, int]]:
    documents = paper_to_documents(paper)
    evidence = {
        item
        for answer in question.answers
        for item in answer.evidence
        if not item.startswith("FLOAT SELECTED")
    }

    relevant = {
        document.metadata["doc_id"]: 1
        for document in documents
        if any(
            evidence_matches(gold, document.page_content)
            for gold in evidence
        )
    }
    return query_id(paper.paper_id, question_idx), relevant


def build_qrels(papers: Iterable[QasperPaper]) -> dict[str, dict[str, int]]:
    qrels = {}
    for paper in papers:
        for question_idx, question in enumerate(paper.questions):
            qid, relevant = question_qrels(paper, question_idx, question)
            if relevant:
                qrels[qid] = relevant
    return qrels


def build_run(
    papers: Iterable[QasperPaper],
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
) -> dict[str, dict[str, float]]:
    run = {}
    for paper in papers:
        for question_idx, question in enumerate(paper.questions):
            qid = query_id(paper.paper_id, question_idx)
            query_vector = embedder.embed_query(question.text)
            documents = store.search_up(
                query_vector,
                top_k,
                paper_id=paper.paper_id,
            )
            run[qid] = {
                document.metadata["doc_id"]: float(
                    document.metadata["_score"]
                )
                for document in documents
            }
    return run


def evaluate_retrieval(
    papers: Iterable[QasperPaper],
    embedder: Embedder,
    store: VectorStore,
    top_k: int = 10,
) -> dict[str, float]:
    papers = tuple(papers)
    qrels_dict = build_qrels(papers)
    run_dict = build_run(papers, embedder, store, top_k)

    if not qrels_dict:
        return {
            "total_queries": float(sum(len(paper.questions) for paper in papers)),
            "evaluated_queries": 0.0,
        }

    qrels = Qrels(qrels_dict)
    run = Run({qid: run_dict.get(qid, {}) for qid in qrels_dict})
    raw_scores = evaluate(
        qrels,
        run,
        metrics=["mrr@10", "ndcg@10", "recall@10"],
    )

    if not isinstance(raw_scores, dict):
        raise TypeError("ranx.evaluate returned a scalar")

    scores = {
        str(key): float(value)
        for key, value in raw_scores.items()
    }

    scores["total_queries"] = float(
        sum(len(paper.questions) for paper in papers)
    )
    scores["evaluated_queries"] = float(len(qrels_dict))

    return scores
