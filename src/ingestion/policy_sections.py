"""Heading markers found in the actual supplied policy PDF (verified by
extracting all 17 pages of text and reading them - see docs/modules/ingestion.md
for the excerpt this was built from).

This policy's real structure does NOT match a generic "Waiting Periods" /
"Sub-limits" section layout: the 30-day and pre-existing-disease waiting
periods are numbered items INSIDE "WHAT WE EXCLUDE" (items 1-3), and the
room-rent/procedure sub-limits are itemized inside "SCOPE OF COVER". Rather
than guess page ranges, the chunker detects these literal heading strings
in the extracted text and assigns every clause to the nearest heading that
precedes it (carried across page boundaries), which is robust to the
heading landing mid-page.
"""

HEADING_MARKERS: list[tuple[str, str]] = [
    ("DEFINITIONS", "Definitions"),
    ("SCOPE OF COVER", "Scope of Cover"),
    ("WHAT WE EXCLUDE", "Exclusions"),
    ("EXTENSIONS", "Extensions"),
    ("CLAIMS PROCEDURE", "Claims Procedure"),
    ("STANDARD TERMS AND CONDITIONS", "Standard Terms and Conditions"),
]

DEFAULT_SECTION = "Preamble"
