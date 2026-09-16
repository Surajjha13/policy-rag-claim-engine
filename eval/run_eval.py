"""End-to-end evaluation: runs every public + custom case through the real
pipeline and reports decision accuracy, citation grounding, retrieval
recall@k (on the hand-labeled subset), and abstention counts.

Usage: python eval/run_eval.py   (run from the project root, with the
policy index already built via `python -m src.ingestion.build_index`.)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.pipeline import run_pipeline  # noqa: E402
from src.schemas.case import ClaimCase  # noqa: E402

PUBLIC_CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "candidate_cases" / "public_test_cases.json"
CUSTOM_CASES_PATH = Path(__file__).parent / "custom_cases.json"
EXPECTED_PATH = Path(__file__).parent / "expected_outcomes.json"
GOLD_EVIDENCE_PATH = Path(__file__).parent / "gold_evidence.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_cases() -> list[dict]:
    public = json.loads(PUBLIC_CASES_PATH.read_text())
    custom = json.loads(CUSTOM_CASES_PATH.read_text())
    return public + custom


def citation_hit_rate(response) -> float:
    if not response.citations:
        return 0.0
    grounded = sum(1 for c in response.citations if c.page > 0 and c.chunk_id)
    return grounded / len(response.citations)


def recall_at_k(case_id: str, response, gold: dict) -> float | None:
    entry = gold.get(case_id)
    if not entry:
        return None
    retrieved_ids = {c.chunk_id for c in response.citations}
    gold_ids = set(entry["gold_chunk_ids"])
    if not gold_ids:
        return None
    return len(retrieved_ids & gold_ids) / len(gold_ids)


def main():
    cases = load_cases()
    expected = json.loads(EXPECTED_PATH.read_text())
    gold = json.loads(GOLD_EVIDENCE_PATH.read_text())

    results = []
    correct = 0
    needs_review_count = 0
    for raw in cases:
        case = ClaimCase.model_validate(raw)
        response = run_pipeline(case)
        exp = expected.get(case.case_id, {})
        is_correct = exp.get("expected_decision") == response.decision
        correct += int(is_correct)
        needs_review_count += int(response.decision == "NEEDS_REVIEW")
        results.append(
            {
                "case_id": case.case_id,
                "decision": response.decision,
                "expected_decision": exp.get("expected_decision"),
                "correct": is_correct,
                "confidence": response.confidence,
                "citation_hit_rate": citation_hit_rate(response),
                "recall_at_k": recall_at_k(case.case_id, response, gold),
                "validation_status": response.validation.status,
            }
        )

    citation_rates = [r["citation_hit_rate"] for r in results]
    summary = {
        "total_cases": len(cases),
        "accuracy": correct / len(cases),
        "needs_review_count": needs_review_count,
        "avg_citation_hit_rate": sum(citation_rates) / len(citation_rates) if citation_rates else 0.0,
    }
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "cases": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
