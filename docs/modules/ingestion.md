# Ingestion module (`src/ingestion/`)

Turns the policy PDF into the two indexes the retriever needs. Runs once,
offline, via `python -m src.ingestion.build_index` (Task 4's script) - not
on every API request.

## `pdf_parser.py` - `extract_pages(pdf_path) -> list[PageText]`

Uses PyMuPDF to pull raw text per page, 1-indexed. This is the *only* place
page numbers enter the system, which is what lets every downstream citation
say `"page": 7` truthfully.

**Why not read the PDF fresh on every request?** It's ~17 pages and parses
in well under a second, but the embeddings/BM25 index built from it is what's
expensive to rebuild, so `extract_pages` is only ever called from
`build_index.py`, not from the request path.

## `policy_sections.py` - `HEADING_MARKERS`

This is a hand-verified list of the literal heading strings that actually
appear in `USGIC-CSCIndividualHealthInsurance_2017-2018.pdf`, found by
extracting and reading all 17 pages of the real document (not guessed):

```python
HEADING_MARKERS = [
    ("DEFINITIONS", "Definitions"),
    ("SCOPE OF COVER", "Scope of Cover"),
    ("WHAT WE EXCLUDE", "Exclusions"),
    ("EXTENSIONS", "Extensions"),
    ("CLAIMS PROCEDURE", "Claims Procedure"),
    ("STANDARD TERMS AND CONDITIONS", "Standard Terms and Conditions"),
]
```

**Why this instead of a page-range map** (the original plan): the first draft
of this project assumed a generic layout with a standalone "Waiting Periods"
section and a standalone "Sub-limits" section, and planned to hard-code page
ranges per section. Reading the actual text showed that assumption was
**wrong** for this policy:

- The 30-day and pre-existing-disease waiting periods are numbered items
  **inside** "WHAT WE EXCLUDE" (items 1 and 2), not a separate section.
- Room-rent, doctor-fee, and ICU sub-limits are itemized **inside**
  "SCOPE OF COVER", not a separate section.
- Headings land **mid-page** (e.g. page 7 contains the tail end of the
  Critical Illness definitions *and* the start of "SCOPE OF COVER" in the
  same page's text).

A page-range map would have mislabeled every chunk on a heading-boundary
page. Detecting the literal heading strings in the extracted text and
carrying the "current section" forward across pages is robust to exactly
that problem, at the cost of being specific to this document's actual
heading text (a genuinely different policy PDF would need its own list -
see the Known Limitations in `docs/architecture.md`).

## `chunker.py` - `chunk_pages(pages) -> list[Chunk]`

Two-level splitting:

1. **Clause splitting** (`_split_clauses_with_offsets`): regex
   `^\s*(\d{1,2}(?:\.\d{1,2}){0,2})[.)]\s+` finds numbered clause markers
   (`"1. Pre-existing diseases"`, `"2. 30 days Waiting Period"`,
   `"a) Normal Room expenses: 1.0%..."` would NOT match - only digit-led
   markers do, which matches how this policy's operative clauses are
   actually numbered). Text before the first match on a page is kept as its
   own leading piece (see the bug note below) rather than discarded.
2. **Length capping** (`_split_long`): any clause over 900 characters is
   split on sentence boundaries only, never mid-sentence, so a long clause
   (e.g. the pre-existing-disease exclusion with its portability carve-out)
   still becomes multiple retrievable, independently citable chunks.

Each chunk is then tagged with a section via `_headings_in_page`, which
finds every `HEADING_MARKERS` string in that page's raw text and, walking
clauses in reading order, updates `current_section` whenever a heading's
character offset is passed. `current_section` is a variable in the
`chunk_pages` loop, not reset per page, so a heading found on page 7 keeps
applying to page 8's opening clauses until a new heading appears.

**A real bug this caught:** the first implementation only kept clause text
from the position of the first numbered-clause regex match onward, silently
dropping any prose before it (e.g. the Critical Illness prose before
"SCOPE OF COVER" on page 7, or the exclusions preamble before item 1 on
page 8). `test_chunker_assigns_section_by_nearest_preceding_heading_and_carries_across_pages`
caught this - the fix was adding the `leading = text[: matches[0].start()]`
piece in `_split_clauses_with_offsets`. This is documented as failure case
in `docs/architecture_note.md`.

## `build_index.py` - `build_index() -> list[Chunk]`

Orchestrates: `extract_pages` -> `chunk_pages` -> embed every chunk with
`sentence-transformers` -> build a FAISS `IndexFlatIP` over the embeddings
plus an index-aligned chunk-metadata pickle (FAISS itself only stores
vectors) -> build a `BM25Okapi` index over the same chunks -> pickle it
alongside the raw chunk list (so `SparseIndex` never needs the FAISS index
to answer a query). Also writes `index_store/chunks.json` - a plain, human-
readable dump of every chunk, useful for manually spot-checking citations
during development (this is how the gold `chunk_id`s in
`eval/gold_evidence.json` and `eval/expected_outcomes.json` were found).
