import pytest

from src.evaluation.qasper import parse_qasper_paper
from src.models.qasper import AnswerType


def test_parse_qasper_paper(raw_qasper_paper):
    paper = parse_qasper_paper(raw_qasper_paper)
    assert paper.paper_id == "paper-1"
    assert paper.title == "Test paper"
    assert len(paper.sections) == 1
    assert paper.sections[0].paragraphs == (
        "First paragraph.",
        "Second paragraph.",
    )
    assert paper.questions[0].text == "Is this useful?"

def test_parse_answer_type(raw_qasper_paper):
    paper = parse_qasper_paper(raw_qasper_paper)
    answer = paper.questions[0].answers[0]
    assert answer.answer_type == AnswerType.BOOLEAN
    assert answer.yes_no is True

def test_parser_rejects_mismatched_sections(raw_qasper_paper):
      raw_qasper_paper["full_text"]["section_name"].append("Extra")

      with pytest.raises(ValueError):
          parse_qasper_paper(raw_qasper_paper)
