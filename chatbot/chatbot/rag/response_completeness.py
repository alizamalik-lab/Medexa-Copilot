"""Pre-send completeness gate: every CPT + every requested sub-question covered."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CompletenessReport:
    ok: bool
    missing_codes: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.missing_codes:
            parts.append("missing CPTs: " + ", ".join(self.missing_codes))
        if self.missing_topics:
            parts.append("missing topics: " + ", ".join(self.missing_topics))
        return "; ".join(parts) if parts else "complete"


_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "medicare": ("medicare", "cms units", "medicare units"),
    "ama": ("ama", "ama units", "rule of eight"),
    "compare": ("medicare", "ama", "cms", "| rule |"),
    "mue": ("mue",),
    "ncci": ("billed together", "ncci", "with "),
    "modifier": ("modifier 59", "modifier"),
    "units": ("units", "calculation", "billing units"),
    "timed": ("timed", "untimed"),
    "summary": ("description",),
    "category": ("category", "billing rule"),
    "addon": ("add-on", "addon", "aoc"),
    "icd": ("icd",),
}


def _code_mentioned(answer: str, code: str) -> bool:
    return bool(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", answer, re.I)
    )


def _topic_covered(answer: str, topic: str) -> bool:
    markers = _TOPIC_ALIASES.get(topic, (topic,))
    lowered = answer.lower()
    return any(marker in lowered for marker in markers)


def check_response_completeness(
    *,
    answer: str,
    expected_codes: list[str] | None = None,
    expected_topics: set[str] | None = None,
    answered_topics: list[str] | set[str] | None = None,
) -> CompletenessReport:
    """
    Verify:
      ✓ Every CPT processed
      ✓ Every user question answered
      ✓ No CPT ignored
      ✓ No billing category / topic skipped
    """
    codes = expected_codes or []
    topics = set(expected_topics or ())
    answered = set(answered_topics or ())

    missing_codes = [c for c in codes if not _code_mentioned(answer, c)]
    missing_topics: list[str] = []
    for topic in sorted(topics):
        if topic in answered:
            continue
        if _topic_covered(answer, topic):
            continue
        missing_topics.append(topic)

    notes: list[str] = []
    if not missing_codes:
        notes.append("every_cpt_processed")
    if not missing_topics:
        notes.append("every_topic_answered")
    if missing_codes:
        notes.append("cpt_ignored")
    if missing_topics:
        notes.append("topic_skipped")

    return CompletenessReport(
        ok=not missing_codes and not missing_topics,
        missing_codes=missing_codes,
        missing_topics=missing_topics,
        notes=notes,
    )


def append_completeness_gap_notice(answer: str, report: CompletenessReport) -> str:
    """Surface any remaining gaps (should be rare after orchestrator fill)."""
    if report.ok:
        return answer
    lines = [answer.rstrip(), "", "**Completeness note**"]
    if report.missing_codes:
        lines.append(
            "- Still need coverage for: "
            + ", ".join(f"**{c}**" for c in report.missing_codes)
        )
    if report.missing_topics:
        lines.append(
            "- Still need answers for: " + ", ".join(report.missing_topics)
        )
    return "\n".join(lines)
