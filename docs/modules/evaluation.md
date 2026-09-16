# Evaluation module (`eval/`)

## Files and what each one is for

- **`custom_cases.json`** - 6 candidate-authored cases (exceeds the required
  minimum of 5), never mixed into or replacing the supplied
  `public_test_cases.json`. Each targets a specific reliability scenario
  not fully covered, or covered differently, by the 12 supplied cases:

  | Case | Targets |
  |---|---|
  | CUST-001 | Portability where the *previous* Sum Insured is lower than the current one - tests the "reduction applies only to the extent of the previous Sum Insured" clause, producing `PARTIALLY_ADMISSIBLE` |
  | CUST-002 | Irrelevant attributes (`occupation`, `parking_available`) that must not affect a clean `ADMISSIBLE` decision |
  | CUST-003 | An undiagnosed condition - tests "the system cannot confidently establish a condition required by the policy" |
  | CUST-004 | A hospital that *affirmatively* fails the bed-count criterion (evidence present and unfavorable, contrast with PUB-011 where evidence is simply absent) |
  | CUST-005 | Domiciliary treatment where *neither* qualifying condition is true (contrast with PUB-004 where one is true) |
  | CUST-006 | The ICU sub-limit (2% of SI) specifically, distinct from the normal room sub-limit (1%) already exercised by PUB-001/PUB-007 |

- **`expected_outcomes.json`** - the hand-adjudicated gold label for all 18
  cases (12 public + 6 custom). Every entry's `rationale` and
  `policy_reference` were derived by extracting the actual policy PDF text
  (all 17 pages) and reading it, not by assuming generic health-insurance
  conventions - see the "corrected assumptions" note below for a case where
  that mattered.
- **`gold_evidence.json`** - for 7 cases with a single clean clause answer,
  the real `chunk_id`(s) (from `index_store/chunks.json`, produced by
  `build_index.py`) that should be retrieved. Used for the recall@k metric.
- **`run_eval.py`** - loads both case files, runs every case through the
  real `run_pipeline`, and reports: decision accuracy against
  `expected_outcomes.json`, citation hit rate (citations with a real
  page+chunk_id), recall@k on the `gold_evidence.json` subset, and the
  count of `NEEDS_REVIEW` outcomes. Writes `eval/results.json` and prints a
  summary. One command (`python eval/run_eval.py`), no manual steps.

## How "expected outcome" was established (per the assignment's requirement)

Not by assumption. The actual `USGIC-CSCIndividualHealthInsurance_2017-2018.pdf`
was extracted and read end-to-end before writing `expected_outcomes.json`.
This surfaced real, specific facts that a generic-template answer would have
gotten wrong:

- The 30-day and pre-existing-disease (48-month) waiting periods are
  numbered items **inside** "WHAT WE EXCLUDE," not a standalone "Waiting
  Periods" section.
- Room-rent, doctor/surgeon-fee, and medicine/OT sub-limits (1% / 25% / 40%
  of Sum Insured respectively) are itemized **inside** "SCOPE OF COVER."
  The ICU sub-limit is 2%, not the same as the normal room cap - a detail
  CUST-006 specifically exercises.
- **The policy *defines* "Unproven/Experimental Treatment" (Definitions,
  page 6) but never actually excludes it anywhere in the 21-item "WHAT WE
  EXCLUDE" list.** PUB-012's task ("Determine whether the treatment is
  excluded because it is experimental or unproven") reads as if it expects
  a confident exclusion, but the textual evidence for one doesn't exist in
  the supplied document. This project's expected label for PUB-012 is
  `NEEDS_REVIEW`, not `NOT_ADMISSIBLE` - concluding exclusion from the
  definition alone would be exactly the kind of unsupported inference the
  assignment says to avoid. This is flagged as a documented, deliberate
  design decision, not a gap - see `docs/architecture_note.md`.

This yields 4 `NEEDS_REVIEW` cases across the 18 (PUB-006, PUB-011, PUB-012,
CUST-003), exceeding the required minimum of 2, and covers three distinct
*reasons* to abstain: missing documents/unresolved evidence (PUB-006),
evidence present but genuinely inconclusive about a defined term (PUB-011),
and a defined-but-not-operatively-excluded concept (PUB-012, CUST-003).
