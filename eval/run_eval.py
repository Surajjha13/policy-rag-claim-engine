"""End-to-end evaluation: runs every public + custom case through the real
pipeline and reports decision accuracy, citation grounding, retrieval
recall@k (on the hand-labeled subset), and abstention counts.

Usage: python eval/run_eval.py   (run from the project root, with the
policy index already built via `python -m src.ingestion.build_index`.)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.pipeline import run_pipeline  # noqa: E402
from src.schemas.case import ClaimCase  # noqa: E402
from src.schemas.decision import DecisionResponse, ValidationResult  # noqa: E402

# Free-tier LLM providers (e.g. Groq) enforce a strict tokens-per-minute
# budget; running 18 cases back-to-back with no pacing hit that limit on
# the very first real eval run (see docs/architecture_note.md). A short
# pause between cases keeps this script usable on a free tier without
# needing every individual LLM call's retry/backoff (src/llm/client.py) to
# absorb the whole batch's load.
INTER_CASE_DELAY_SECONDS = 20

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
    for i, raw in enumerate(cases):
        case = ClaimCase.model_validate(raw)
        print(f"[{i + 1}/{len(cases)}] running {case.case_id}...", flush=True)
        try:
            response = run_pipeline(case)
        except Exception as e:
            print(f"  -> pipeline error on {case.case_id}: {type(e).__name__}: {e}", flush=True)
            response = DecisionResponse(
                case_id=case.case_id,
                decision="NEEDS_REVIEW",
                confidence=0.0,
                key_findings=[],
                applicable_limits=[],
                missing_evidence=[f"Eval run pipeline error: {type(e).__name__}"],
                citations=[],
                validation=ValidationResult(status="FAIL", unsupported_claims=[]),
                trace=[],
            )
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
        if i < len(cases) - 1:
            time.sleep(INTER_CASE_DELAY_SECONDS)

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
