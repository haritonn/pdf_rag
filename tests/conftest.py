import pytest


@pytest.fixture
def raw_qasper_paper():
    return {
              "id": "paper-1",
              "title": "Test paper",
              "abstract": "Test abstract",
              "full_text": {
                  "section_name": ["Introduction"],
                  "paragraphs": [["First paragraph.", "Second paragraph."]],
              },
              "qas": {
                  "question": ["Is this useful?"],
                  "answers": [
                      {
                          "answer": [
                              {
                                  "unanswerable": False,
                                  "extractive_spans": [],
                                  "free_form_answer": "",
                                  "yes_no": True,
                                  "evidence": ["First paragraph."],
                              }
                          ]
                      }
                  ],
              },
          }
