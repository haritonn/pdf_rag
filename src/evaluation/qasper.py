import json
import unicodedata
from collections import Counter
from datasets import load_dataset


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


def summarize_qasper(dataset):
    stats = Counter(papers=len(dataset))
    answer_types = Counter()
    unique_evidence = set()

    for paper in dataset:
        full_text = paper["full_text"]
        paragraphs = [
            text
            for section in full_text["paragraphs"] for text in section
            if text.strip()
        ]
        paragraph_set = {normalize_text(text) for text in paragraphs}
        stats["sections"] += len(full_text["section_name"])
        stats["paragraphs"] += len(paragraphs)
        stats["questions"] += len(paper["qas"]["question"])

        for answer_group in paper["qas"]["answers"]:
            answers = answer_group["answer"]
            stats["annotations"] += len(answers)
            unanswerable_flags = [answer["unanswerable"] for answer in answers]

            if unanswerable_flags and all(unanswerable_flags):
                stats["unanswerable_questions"] += 1
            elif any(unanswerable_flags):
                stats["mixed_questions"] += 1
            else:
                stats["answerable_questions"] += 1

            for answer in answers:
                answer_types[get_answer_type(answer)] += 1
                for evidence in answer["evidence"]:
                    if evidence.startswith("FLOAT SELECTED"):
                        stats["float_evidence_items"] += 1
                        continue

                    stats["text_evidence_items"] += 1
                    evidence = normalize_text(evidence)
                    unique_evidence.add(evidence)
                    if evidence in paragraph_set:
                        stats["matched_text_evidence_items"] += 1
                    else:
                        stats["unmatched_text_evidence_items"] += 1

    return {
        **stats,
        "answer_types": dict(answer_types),
        "unique_text_evidence_items": len(unique_evidence),
    }


if __name__ == "__main__":
    summary = summarize_qasper(load_qasper())
    print(json.dumps(summary, indent=2))
