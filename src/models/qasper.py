from dataclasses import dataclass
from enum import StrEnum

class AnswerType(StrEnum):
    UNANSWERABLE = "unanswerable"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class QasperAnswer:
    unanswerable: bool
    extractive_spans: tuple[str, ...]
    free_form_answer: str
    yes_no: bool | None
    evidence: tuple[str, ...]

    @property
    def answer_type(self) -> AnswerType:
        if self.unanswerable: return AnswerType.UNANSWERABLE
        if self.extractive_spans: return AnswerType.EXTRACTIVE
        if self.free_form_answer.strip(): return AnswerType.ABSTRACTIVE
        if self.yes_no is not None: return AnswerType.BOOLEAN
        return AnswerType.UNKNOWN

@dataclass(frozen=True, slots=True)
class QasperQuestion:
    text: str
    answers: tuple[QasperAnswer, ...]

@dataclass(frozen=True, slots=True)
class QasperSection:
    name: str
    paragraphs: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class QasperPaper:
    paper_id: str
    title: str
    abstract: str
    sections: tuple[QasperSection, ...]
    questions: tuple[QasperQuestion, ...]
