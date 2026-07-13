"""Deterministic timed-CPT unit calculation for AMA and CMS methodologies."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CptTimeEntry:
    cpt_code: str
    minutes: int


_COMPARE_HINTS = (
    "compare",
    "comparison",
    "difference between",
    " vs ",
    "versus",
)

_UNIT_CALC_HINTS = (
    "calculate unit",
    "calcute unit",
    "calcute",
    "calculate",
    "how many unit",
    "units under",
    "unit calculation",
    "billable unit",
    "8-minute",
    "8 minute",
    "minute rule",
    "rule of eight",
    "ama rule",
    "cms rule",
)

_MINUTES_OF_CPT = re.compile(
    r"(\d{1,3})\s*minutes?\s+of\s+(?:cpt\s*)?(\d{5})\b",
    re.IGNORECASE,
)
_CPT_EQUALS_MINUTES = re.compile(
    r"\b(\d{5})\s*[=:]\s*(\d{1,3})\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_CPT_THEN_MINUTES = re.compile(
    r"\b(\d{5})\b[^.\d=]{0,40}?(\d{1,3})\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_CPT_CODE = re.compile(r"\b(\d{5})\b")
_TIME_VALUE = re.compile(r"\b(\d{1,3})\s*min(?:ute)?s?\b", re.IGNORECASE)


def is_rule_comparison_question(question: str) -> bool:
    lowered = question.lower()
    if not any(hint in lowered for hint in _COMPARE_HINTS):
        return False
    mentions_cms = any(
        term in lowered
        for term in ("medicare", "cms", "8-minute", "8 minute", "8 min rule")
    )
    mentions_ama = any(
        term in lowered for term in ("ama", "rule of eight")
    )
    return mentions_cms and mentions_ama


def is_unit_calculation_question(question: str) -> bool:
    lowered = question.lower()
    if is_rule_comparison_question(question):
        return bool(_CPT_CODE.search(question)) and bool(
            _TIME_VALUE.search(question) or _CPT_EQUALS_MINUTES.search(question)
        )
    if not any(hint in lowered for hint in _UNIT_CALC_HINTS):
        return False
    has_codes = bool(_CPT_CODE.search(question))
    has_times = bool(_TIME_VALUE.search(question) or _CPT_EQUALS_MINUTES.search(question))
    return has_codes and has_times


def detect_rule_methodology(question: str) -> str:
    lowered = question.lower()
    if "ama" in lowered or "rule of eight" in lowered:
        return "AMA"
    if any(
        term in lowered
        for term in ("medicare", "cms", "8-minute", "8 minute", "8 min rule")
    ):
        return "CMS"
    return "CMS"


def parse_cpt_time_entries(question: str) -> list[CptTimeEntry]:
    found: dict[str, int] = {}

    for match in _CPT_EQUALS_MINUTES.finditer(question):
        code = match.group(1)
        minutes = int(match.group(2))
        found[code] = minutes

    for match in _MINUTES_OF_CPT.finditer(question):
        minutes = int(match.group(1))
        code = match.group(2)
        found[code] = minutes

    for match in _CPT_THEN_MINUTES.finditer(question):
        code = match.group(1)
        minutes = int(match.group(2))
        if code not in found:
            found[code] = minutes

    return [CptTimeEntry(cpt_code=code, minutes=minutes) for code, minutes in found.items()]


def timed_units_for_minutes(minutes: int) -> int:
    """15-minute timed units using the CMS/AMA 8-minute remainder rule."""
    if minutes < 8:
        return 0
    full_units = minutes // 15
    remainder = minutes % 15
    if remainder >= 8:
        full_units += 1
    return full_units


def cms_conversion_table_text() -> str:
    return (
        "• 8–22 minutes = **1 unit**\n"
        "• 23–37 minutes = **2 units**\n"
        "• 38–52 minutes = **3 units**\n"
        "• Continue per the CMS unit table for additional minutes"
    )


def format_cms_ama_conceptual_comparison() -> str:
    """Conceptual CMS vs AMA comparison table (no patient minutes required)."""
    lines = [
        "| Topic | CMS 8-Minute Rule | AMA Rule of Eight |",
        "|-------|-------------------|-------------------|",
        "| Used By | Medicare | Commonly used by many commercial payers |",
        "| Calculation Method | Pool all timed CPT minutes | Calculate each CPT code separately |",
        "| Unit Assignment | Based on total pooled minutes | Each CPT must independently qualify |",
        "| Typical Use | Medicare billing | Commercial payer billing (when adopted) |",
        "",
        "Always verify the payer's billing guidelines, as commercial insurance policies may differ.",
    ]
    return "\n".join(lines)


def is_cms_ama_comparison_question(question: str) -> bool:
    lowered = question.lower()
    mentions_cms = any(
        term in lowered
        for term in ("medicare", "cms", "8-minute", "8 minute", "8 min rule")
    )
    mentions_ama = any(term in lowered for term in ("ama", "rule of eight"))
    if not (mentions_cms and mentions_ama):
        return False
    return any(hint in lowered for hint in _COMPARE_HINTS) or (
        "difference" in lowered or "compare" in lowered
    )


def try_answer_rule_comparison_explanation(question: str) -> str | None:
    """Return a conceptual CMS vs AMA comparison when no timed minutes are provided."""
    if not is_cms_ama_comparison_question(question):
        return None
    if parse_cpt_time_entries(question):
        return None
    return format_cms_ama_conceptual_comparison()


def calculate_ama_units(entries: list[CptTimeEntry]) -> dict[str, int]:
    return {entry.cpt_code: timed_units_for_minutes(entry.minutes) for entry in entries}


def calculate_cms_pooled_units(entries: list[CptTimeEntry]) -> dict[str, int]:
    total_minutes = sum(entry.minutes for entry in entries)
    total_units = timed_units_for_minutes(total_minutes)
    if not entries:
        return {}
    if len(entries) == 1:
        return {entries[0].cpt_code: total_units}

    # CMS pools minutes across timed codes; assign units to the line with the most time.
    primary = max(entries, key=lambda entry: entry.minutes)
    result = {entry.cpt_code: 0 for entry in entries}
    result[primary.cpt_code] = total_units
    return result


def _format_unit_line(code: str, minutes: int, units: int) -> str:
    unit_label = "unit" if units == 1 else "units"
    return f"**{code}**: {units} {unit_label} ({minutes} minutes)"


def _format_ama_answer(entries: list[CptTimeEntry]) -> list[str]:
    units_by_code = calculate_ama_units(entries)
    lines = [
        "### AMA Rule of Eight (each CPT calculated separately)",
    ]
    for entry in entries:
        lines.append(
            f"- {_format_unit_line(entry.cpt_code, entry.minutes, units_by_code[entry.cpt_code])}"
        )
    ama_total = sum(units_by_code.values())
    lines.append(
        f"- **Total billable units: {ama_total}**"
    )
    return lines


def _format_cms_answer(entries: list[CptTimeEntry]) -> list[str]:
    total_minutes = sum(entry.minutes for entry in entries)
    total_units = timed_units_for_minutes(total_minutes)
    minute_parts = " + ".join(str(entry.minutes) for entry in entries)
    lines = [
        "### Medicare CMS 8-Minute Rule (minutes pooled first)",
        f"- Pooled minutes: {minute_parts} = **{total_minutes} min**",
    ]
    for entry in entries:
        lines.append(f"- {entry.cpt_code}: {entry.minutes} min (contributes to pool)")
    lines.append(
        f"- **Total billable units: {total_units}** from pooled time"
    )
    return lines


def _format_comparison_answer(entries: list[CptTimeEntry]) -> str:
    cms_total = timed_units_for_minutes(sum(entry.minutes for entry in entries))
    ama_units = calculate_ama_units(entries)
    ama_total = sum(ama_units.values())
    total_minutes = sum(entry.minutes for entry in entries)
    minute_parts = " + ".join(str(entry.minutes) for entry in entries)

    lines = [
        format_cms_ama_conceptual_comparison(),
        "",
        f"### Example with **{total_minutes} total timed minutes** ({minute_parts})",
        "",
        "| Rule | Billable Units |",
        "|------|----------------|",
        f"| **CMS 8-Minute Rule** (pooled minutes) | **{cms_total}** |",
        f"| **AMA Rule of Eight** (each CPT separate) | **{ama_total}** |",
    ]
    if cms_total != ama_total:
        lines.append(
            f"\nFor this example, pooled Medicare/CMS time yields **{cms_total} unit{'s' if cms_total != 1 else ''}**, "
            f"while AMA yields **{ama_total} unit{'s' if ama_total != 1 else ''}**."
        )
    return "\n".join(lines)


def try_answer_unit_calculation(question: str) -> str | None:
    payload = try_unit_calculation_payload(question)
    if payload is None:
        return None
    return payload["answer"]


def try_unit_calculation_payload(question: str) -> dict | None:
    entries = parse_cpt_time_entries(question)
    if not entries:
        return None
    if not (
        is_unit_calculation_question(question)
        or _has_implicit_time_documentation(question)
    ):
        return None

    answer = _build_unit_calculation_answer(entries, question)
    return {
        "type": "unit_calculation",
        "entries": [
            {"cpt_code": entry.cpt_code, "minutes": entry.minutes}
            for entry in entries
        ],
        "answer": answer,
    }


def _has_implicit_time_documentation(question: str) -> bool:
    lowered = question.lower()
    if not _TIME_VALUE.search(question):
        return False
    return any(
        hint in lowered
        for hint in (
            "performed",
            "provided",
            "treated",
            "spent",
            "minutes of",
            "min of",
            "for ",
        )
    )


def _build_unit_calculation_answer(entries: list[CptTimeEntry], question: str) -> str:
    if is_rule_comparison_question(question):
        return _format_comparison_answer(entries)

    methodology = detect_rule_methodology(question)
    total_minutes = sum(entry.minutes for entry in entries)
    total_units = timed_units_for_minutes(total_minutes)

    if methodology == "CMS":
        table = cms_conversion_table_text()
        if len(entries) == 1:
            entry = entries[0]
            return "\n".join(
                [
                    f"**Step 1:** Total timed minutes = **{entry.minutes} minutes**",
                    (
                        "**Step 2:** Rule applied = **CMS 8-Minute Rule** for Medicare timed "
                        "therapy services. Use the CMS conversion table:\n" + table
                    ),
                    (
                        f"**Step 3:** Billable units = **{total_units} unit"
                        f"{'s' if total_units != 1 else ''}**"
                    ),
                    "",
                    f"**Final Answer:** **{total_units} unit{'s' if total_units != 1 else ''}**",
                ]
            )

        minute_parts = " + ".join(str(entry.minutes) for entry in entries)
        code_list = ", ".join(entry.cpt_code for entry in entries)
        return "\n".join(
            [
                f"**Step 1:** Total timed minutes = **{total_minutes} minutes** ({minute_parts})",
                (
                    f"**Step 2:** Rule applied = **CMS 8-Minute Rule**. Pool all timed therapy "
                    f"minutes across {code_list}, then apply the CMS conversion table:\n" + table
                ),
                (
                    f"**Step 3:** Billable units = **{total_units} unit"
                    f"{'s' if total_units != 1 else ''}** from pooled time"
                ),
                "",
                f"**Final Answer:** **{total_units} unit{'s' if total_units != 1 else ''}**",
            ]
        )

    if len(entries) == 1:
        entry = entries[0]
        units = timed_units_for_minutes(entry.minutes)
        return "\n".join(
            [
                f"**Step 1:** Total treatment time = **{entry.minutes} minutes**",
                "**Step 2:** Apply the **AMA Rule of Eight** to this CPT individually",
                (
                    f"**Step 3:** {entry.minutes} minutes qualifies for "
                    f"**{units} billable unit{'s' if units != 1 else ''}**"
                ),
                "",
                f"**Final Answer:** **{units} unit{'s' if units != 1 else ''}**",
            ]
        )

    return "\n".join(_format_ama_answer(entries))
