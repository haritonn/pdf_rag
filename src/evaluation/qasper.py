import json
import unicodedata
from collections import Counter
from datasets import load_dataset
from typing import Mapping, Any

from src.models.qasper import (
    QasperPaper, QasperSection, QasperQuestion, QasperAnswer, AnswerType
)


BASE_URL = (
    "https://huggingface.co/datasets/allenai/qasper/"
    "resolve/refs%2Fconvert%2Fparquet/qasper"
)


def load_qasper():
    url = f"{BASE_URL}/validation/0000.parquet"
    return load_dataset("parquet", data_files={"validation": url}, split="validation")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def get_answer_type(answer):
    if answer["unanswerable"]: return "unanswerable"
    if answer["extractive_spans"]: return "extractive"
    if answer["free_form_answer"].strip(): return "abstractive"
    if answer["yes_no"] is not None: return "boolean"
    return "unknown"

def parse_qasper_answer(raw: Mapping[str, Any]) -> QasperAnswer:
      return QasperAnswer(
          unanswerable=bool(raw["unanswerable"]),
          extractive_spans=tuple(raw["extractive_spans"]),
          free_form_answer=raw["free_form_answer"],
          yes_no=raw["yes_no"],
          evidence=tuple(raw["evidence"]),
      )

def parse_qasper_paper(raw: Mapping[str, Any]) -> QasperPaper:
    full_text = raw['full_text']
    qas = raw['qas']
    sections = tuple(
        QasperSection(
            name=section_name,
            paragraphs=tuple(paragraph for paragraph in paragraphs)
        ) for section_name, paragraphs in zip(full_text['section_name'], full_text['paragraphs'], strict=True)
    )

    questions = tuple(
        QasperQuestion(
            text=question,
            answers=tuple(parse_qasper_answer(answer) for answer in answer_group['answer'])
        ) for question, answer_group in zip(qas['question'], qas['answers'], strict=True)
    )

    return QasperPaper(
        paper_id=raw['id'],
        title=raw['title'],
        abstract=raw['abstract'],
        sections=sections,
        questions=questions,
    )

if __name__ == "__main__":
    dataset = load_qasper()
    paper = parse_qasper_paper(dataset[0])
    assert paper.paper_id
    assert paper.title
    assert paper.abstract
    assert all(
        isinstance(answer.answer_type, AnswerType)
        for question in paper.questions
        for answer in question.answers
    )
