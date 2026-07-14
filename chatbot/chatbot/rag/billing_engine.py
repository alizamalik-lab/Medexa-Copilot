"""Deterministic CPT unit calculation using PT/OT/SLP billing categories."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.billing_categories import CodeBillingProfile, get_billing_category_store
from rag.query_router import CPT_PATTERN, HCPCS_PATTERN

EIGHT_MINUTE_RULE = "8_minute_rule"
FULL_BLOCK_REQUIRED = "full_block_required"
UNTIMED_RULES = frozenset(
    {
        "untimed_per_session",
        "untimed_per_encounter",
        "untimed_per_procedure",
        "untimed_per_day",
        "untimed_per_episode",
    }
)
AREA_BASED = "area_based"
TIME_BAND_SELECT = "time_band_select"


@dataclass(frozen=True)
class CptTimeEntry:
    cpt_code: str
    minutes: int


@dataclass(frozen=True)
class UnitCalculationResult:
    cpt_code: str
    minutes: int
    units: int
    billing_rule: str
    strategy: str
    notes: str = ""
    area_sq_cm: int | None = None
    calculated_units: int | None = None


_UNIT_GUIDE_HINTS = (
    "how to calculate",
    "how do i calculate",
    "how are units calculated",
    "how is it billed",
    "calculate billing unit",
    "billing unit calculation",
    "how many units can i bill",
)

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
    "billing unit",
    "billing units",
    "8-minute",
    "8 minute",
    "minute rule",
    "rule of eight",
    "ama rule",
    "cms rule",
)

_BILLING_CODE = r"(?:\d{5}|[A-VJ-Z]\d{4})"

_MINUTES_OF_CPT = re.compile(
    rf"(\d{{1,3}})\s*minutes?\s+of\s+(?:cpt\s*|hcpcs\s*)?({_BILLING_CODE})\b",
    re.IGNORECASE,
)
_CPT_EQUALS_MINUTES = re.compile(
    rf"\b({_BILLING_CODE})\s*[=:]\s*(\d{{1,3}})\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_CPT_THEN_MINUTES = re.compile(
    rf"\b({_BILLING_CODE})\b[^.]{{0,60}}?(\d{{1,3}})\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_CPT_FOR_MINUTES = re.compile(
    rf"\b({_BILLING_CODE})\b(?:\s+\w+){{0,4}}?\s+for\s+(\d{{1,3}})\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_CPT_CODE = re.compile(rf"\b({_BILLING_CODE})\b", re.IGNORECASE)
_TIME_VALUE = re.compile(r"\b(\d{1,3})\s*min(?:ute)?s?\b", re.IGNORECASE)
_AREA_SQ_CM = re.compile(
    r"(\d{1,4})\s*(?:sq\.?\s*cm|square\s*centimeters?)",
    re.IGNORECASE,
)
_OCCURRENCE_HINTS = re.compile(
    r"\b(twice|two times|2 times|thrice|three times|3 times|"
    r"once in the morning|once at night|morning and.{0,40}night|"
    r"same day|per day|calendar day)\b",
    re.IGNORECASE,
)

_CPT_DEFINITION_HINTS = re.compile(
    r"^\s*(?:what\s+is|what('s|s)|define|explain)\s+"
    r"(?:a\s+|an\s+|the\s+)?cpt(?:\s+code|\s+codes)?\s*\??\s*$",
    re.IGNORECASE,
)

CPT_DEFINITION_ANSWER = (
    "**CPT** stands for **Current Procedural Terminology**.\n\n"
    "• CPT codes are standardized 5-digit procedure codes maintained by the AMA.\n"
    "• They describe medical, surgical, and therapy services for billing and claims.\n"
    "• Related HCPCS Level II codes (for example, **G0329**) are often used with "
    "Medicare for similar service reporting.\n\n"
    "If you have a specific code (like **97110**), ask what it is, its MUE, "
    "add-on codes, or how units are calculated."
)


def is_rule_comparison_question(question: str) -> bool:
    lowered = question.lower()
    if not any(hint in lowered for hint in _COMPARE_HINTS):
        return False
    mentions_cms = any(
        term in lowered
        for term in ("medicare", "cms", "8-minute", "8 minute", "8 min rule")
    )
    mentions_ama = any(term in lowered for term in ("ama", "rule of eight"))
    return mentions_cms and mentions_ama


def is_unit_calculation_question(question: str) -> bool:
    lowered = question.lower()
    if is_rule_comparison_question(question):
        return bool(_CPT_CODE.search(question)) and bool(
            _TIME_VALUE.search(question) or _CPT_EQUALS_MINUTES.search(question)
        )
    if not any(hint in lowered for hint in _UNIT_CALC_HINTS):
        return False
    has_codes = bool(_CPT_CODE.search(question) or HCPCS_PATTERN.search(question))
    has_times = bool(
        _TIME_VALUE.search(question)
        or _CPT_EQUALS_MINUTES.search(question)
        or _AREA_SQ_CM.search(question)
    )
    return has_codes and has_times


def _extract_billing_codes(question: str) -> list[str]:
    """Extract every CPT/HCPCS code in appearance order (never drop later codes)."""
    hits: list[tuple[int, str]] = []
    for pattern in (CPT_PATTERN, HCPCS_PATTERN):
        for match in pattern.finditer(question):
            hits.append((match.start(), match.group(1).upper()))
    hits.sort(key=lambda item: item[0])
    seen: set[str] = set()
    codes: list[str] = []
    for _pos, code in hits:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def is_unit_calculation_guide_question(question: str) -> bool:
    lowered = question.lower()
    if not any(hint in lowered for hint in _UNIT_GUIDE_HINTS):
        return False
    if not _extract_billing_codes(question):
        return False
    if parse_cpt_time_entries(question) or _TIME_VALUE.search(question):
        return False
    return True


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
        code = match.group(1).upper()
        minutes = int(match.group(2))
        found[code] = minutes

    for match in _CPT_FOR_MINUTES.finditer(question):
        code = match.group(1).upper()
        minutes = int(match.group(2))
        found[code] = minutes

    for match in _MINUTES_OF_CPT.finditer(question):
        minutes = int(match.group(1))
        code = match.group(2).upper()
        found[code] = minutes

    for match in _CPT_THEN_MINUTES.finditer(question):
        code = match.group(1).upper()
        minutes = int(match.group(2))
        if code not in found:
            found[code] = minutes

    # Unit questions that name a code and total minutes, but not "CODE for N minutes"
    # e.g. "I performed G0329 twice today for 40 minutes"
    if not found:
        codes = _extract_billing_codes(question)
        time_match = _TIME_VALUE.search(question)
        if len(codes) == 1 and time_match:
            found[codes[0]] = int(time_match.group(1))

    return [
        CptTimeEntry(cpt_code=code, minutes=minutes) for code, minutes in found.items()
    ]


def try_cpt_definition_answer(question: str) -> str | None:
    if _CPT_DEFINITION_HINTS.match(question.strip()):
        return CPT_DEFINITION_ANSWER
    return None


def mentions_same_day_multiple_occurrences(question: str) -> bool:
    return bool(_OCCURRENCE_HINTS.search(question))


def parse_area_sq_cm(question: str) -> int | None:
    match = _AREA_SQ_CM.search(question)
    if not match:
        return None
    return int(match.group(1))


def eight_minute_units(
    minutes: int,
    *,
    segment_size_minutes: int = 15,
    unit_threshold_minutes: int = 8,
) -> int:
    if minutes < unit_threshold_minutes:
        return 0
    full_units = minutes // segment_size_minutes
    remainder = minutes % segment_size_minutes
    if remainder >= unit_threshold_minutes:
        full_units += 1
    return full_units


def timed_units_for_minutes(minutes: int) -> int:
    """Backward-compatible CMS/AMA 8-minute remainder rule on 15-minute segments."""
    return eight_minute_units(minutes)


def full_block_units(minutes: int, block_minutes: int) -> int:
    if block_minutes <= 0 or minutes < block_minutes:
        return 0
    return minutes // block_minutes


def is_session_limited_profile(profile: CodeBillingProfile) -> bool:
    return profile.max_units_cap == 1


def calculate_full_block_units(
    minutes: int,
    profile: CodeBillingProfile,
) -> tuple[int, int]:
    """Return (calculated_units, billable_units) for full-block CPTs."""
    block = profile.block_minutes or 15
    if is_session_limited_profile(profile):
        units = 1 if minutes >= block else 0
        return units, units

    calculated = full_block_units(minutes, block)
    final = apply_max_units_cap(calculated, profile)
    return calculated, final


def apply_max_units_cap(
    calculated_units: int,
    profile: CodeBillingProfile | None,
) -> int:
    if profile is None or profile.max_units_cap is None:
        return calculated_units
    return min(calculated_units, profile.max_units_cap)


def minutes_in_time_band(minutes: int, time_band: str) -> bool:
    band = time_band.strip().lower()
    range_match = re.match(r"(\d+)\s*-\s*(\d+)\s*minutes?", band)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return low <= minutes <= high
    plus_match = re.match(r"(\d+)\s*\+\s*minutes?", band)
    if plus_match:
        return minutes >= int(plus_match.group(1))
    return False


def cms_conversion_table_text(
    *,
    segment_size_minutes: int = 15,
    unit_threshold_minutes: int = 8,
) -> str:
    first_end = segment_size_minutes + unit_threshold_minutes - 1
    second_start = segment_size_minutes + unit_threshold_minutes
    second_end = (segment_size_minutes * 2) + unit_threshold_minutes - 1
    third_start = (segment_size_minutes * 2) + unit_threshold_minutes
    third_end = (segment_size_minutes * 3) + unit_threshold_minutes - 1
    return (
        f"• {unit_threshold_minutes}–{first_end} minutes = **1 unit**\n"
        f"• {second_start}–{second_end} minutes = **2 units**\n"
        f"• {third_start}–{third_end} minutes = **3 units**\n"
        f"• Continue per the {segment_size_minutes}-minute segment table"
    )


def format_cms_ama_conceptual_comparison() -> str:
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
    if not is_cms_ama_comparison_question(question):
        return None
    if parse_cpt_time_entries(question):
        return None
    return format_cms_ama_conceptual_comparison()


def calculate_ama_units(entries: list[CptTimeEntry]) -> dict[str, int]:
    store = get_billing_category_store()
    results: dict[str, int] = {}
    for entry in entries:
        profile = store.get_profile(entry.cpt_code)
        if profile and profile.billing_rule == EIGHT_MINUTE_RULE:
            results[entry.cpt_code] = eight_minute_units(
                entry.minutes,
                segment_size_minutes=profile.segment_size_minutes,
                unit_threshold_minutes=profile.unit_threshold_minutes,
            )
        else:
            results[entry.cpt_code] = timed_units_for_minutes(entry.minutes)
    return results


def calculate_cms_pooled_units(entries: list[CptTimeEntry]) -> dict[str, int]:
    store = get_billing_category_store()
    eight_minute_entries = [
        entry
        for entry in entries
        if (profile := store.get_profile(entry.cpt_code))
        and profile.billing_rule == EIGHT_MINUTE_RULE
    ]
    if not eight_minute_entries:
        eight_minute_entries = list(entries)

    profile = store.get_profile(eight_minute_entries[0].cpt_code)
    segment = profile.segment_size_minutes if profile else 15
    threshold = profile.unit_threshold_minutes if profile else 8

    total_minutes = sum(entry.minutes for entry in eight_minute_entries)
    total_units = eight_minute_units(
        total_minutes,
        segment_size_minutes=segment,
        unit_threshold_minutes=threshold,
    )
    if len(eight_minute_entries) == 1:
        return {eight_minute_entries[0].cpt_code: total_units}

    primary = max(eight_minute_entries, key=lambda entry: entry.minutes)
    result = {entry.cpt_code: 0 for entry in eight_minute_entries}
    result[primary.cpt_code] = total_units
    return result


def calculate_units_for_entry(
    entry: CptTimeEntry,
    profile: CodeBillingProfile | None,
    *,
    methodology: str = "CMS",
    area_sq_cm: int | None = None,
) -> UnitCalculationResult:
    store = get_billing_category_store()
    code = entry.cpt_code.upper()

    if profile is None:
        units = timed_units_for_minutes(entry.minutes)
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule="unknown",
            strategy="default_8_minute_fallback",
            notes=(
                f"CPT {code} was not found in pt_ot_slp_billing_categories.json. "
                "Applied generic 15-minute / 8-minute remainder logic."
            ),
        )

    rule = profile.billing_rule
    rule_text = store.rule_definition(rule)

    if rule == EIGHT_MINUTE_RULE:
        calculated = eight_minute_units(
            entry.minutes,
            segment_size_minutes=profile.segment_size_minutes,
            unit_threshold_minutes=profile.unit_threshold_minutes,
        )
        units = apply_max_units_cap(calculated, profile)
        strategy = (
            "ama_rule_of_eight_per_code"
            if methodology == "AMA"
            else "cms_8_minute_rule_per_code"
        )
        prefix = (
            "AMA Rule of Eight: calculate this CPT independently. "
            if methodology == "AMA"
            else (
                f"{profile.segment_size_minutes}-minute segments with "
                f"{profile.unit_threshold_minutes}-minute remainder threshold. "
            )
        )
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule=rule,
            strategy=strategy,
            notes=f"{prefix}{rule_text}",
            calculated_units=calculated,
        )

    if rule == FULL_BLOCK_REQUIRED:
        block = profile.block_minutes or 15
        calculated, units = calculate_full_block_units(entry.minutes, profile)
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule=rule,
            strategy="full_block_required",
            calculated_units=calculated,
            notes=(
                f"Each unit requires a full {block}-minute block. "
                f"No partial credit. {rule_text}"
            ),
        )

    if rule in UNTIMED_RULES:
        units = 1
        notes = (
            "This is an untimed CPT. Bill 1 unit when the service is documented; "
            f"treatment minutes do not drive unit count. {rule_text}"
        )
        if profile.max_units_cap == 1 or "1 per day" in (
            profile.max_units_allowed or ""
        ).lower():
            units = 1
            notes = (
                f"Category rule is `{rule}` ({profile.max_units_allowed or '1 per day'}). "
                "Bill **1 unit per calendar day**, even if the service is performed "
                "more than once or additional minutes are documented. "
                f"{rule_text}"
            )
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule=rule,
            strategy=rule,
            notes=notes,
            calculated_units=1,
        )

    if rule == TIME_BAND_SELECT:
        band = profile.time_band or profile.min_time_for_1_unit
        in_band = minutes_in_time_band(entry.minutes, band) if band else False
        units = 1 if in_band else 0
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule=rule,
            strategy="time_band_select",
            notes=(
                f"Select one code based on total time band ({band}). "
                f"{'Time qualifies for this code.' if in_band else 'Time does not qualify for this code band.'} "
                f"{rule_text}"
            ),
        )

    if rule == AREA_BASED:
        if area_sq_cm is None:
            return UnitCalculationResult(
                cpt_code=code,
                minutes=entry.minutes,
                units=0,
                billing_rule=rule,
                strategy="area_based",
                notes=(
                    "Area-based billing requires wound size in sq cm. "
                    f"{profile.area_unit_sq_cm or profile.min_time_for_1_unit or rule_text}"
                ),
            )
        unit_size = profile.area_unit_sq_cm or 20
        calculated = max(1, area_sq_cm // unit_size) if area_sq_cm >= unit_size else 0
        units = apply_max_units_cap(calculated, profile)
        return UnitCalculationResult(
            cpt_code=code,
            minutes=entry.minutes,
            units=units,
            billing_rule=rule,
            strategy="area_based",
            area_sq_cm=area_sq_cm,
            calculated_units=calculated,
            notes=(
                f"Area-based billing using {unit_size} sq cm per unit. "
                f"{rule_text}"
            ),
        )

    units = timed_units_for_minutes(entry.minutes)
    return UnitCalculationResult(
        cpt_code=code,
        minutes=entry.minutes,
        units=units,
        billing_rule=rule,
        strategy="generic_timed_fallback",
        notes=rule_text,
    )


def _all_eight_minute_rule(entries: list[CptTimeEntry]) -> bool:
    store = get_billing_category_store()
    profiles = [store.get_profile(entry.cpt_code) for entry in entries]
    if any(profile is None for profile in profiles):
        return False
    return all(profile.billing_rule == EIGHT_MINUTE_RULE for profile in profiles)


def _format_comparison_answer(entries: list[CptTimeEntry]) -> str:
    cms_units = calculate_cms_pooled_units(entries)
    cms_total = sum(cms_units.values())
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


def try_unit_calculation_guide(question: str) -> str | None:
    """Explain how to calculate units for a CPT/HCPCS when no minutes are given."""
    if not is_unit_calculation_guide_question(question):
        return None

    codes = _extract_billing_codes(question)
    if not codes:
        return None

    store = get_billing_category_store()
    code = codes[0]
    profile = store.get_profile(code)
    if profile is None:
        return None

    rule_label = profile.billing_rule.replace("_", " ")
    lines = [
        f"**Billing Unit Calculation — {code}**",
        "",
        f"**Billing Rule:** {rule_label}",
    ]

    if profile.billing_rule in UNTIMED_RULES:
        cap = profile.max_units_allowed or "1 when service is documented"
        lines.extend(
            [
                "**Unit Requirement:** Untimed — minutes do not determine units",
                f"**Calculation:** Bill **1 unit** per qualifying service ({cap})",
                f"**Max Units Allowed:** {cap}",
                "**Total Billing Units:** 1",
            ]
        )
        return "\n\n".join(lines)

    if profile.billing_rule == EIGHT_MINUTE_RULE:
        segment = profile.segment_size_minutes
        threshold = profile.unit_threshold_minutes
        cap = profile.max_units_allowed or "Multiple units per day"
        lines.extend(
            [
                (
                    f"**Unit Requirement:** {segment}-minute timed segment "
                    f"({threshold}-minute remainder rule)"
                ),
                (
                    f"**Calculation:** Add documented timed minutes, then apply the "
                    f"8-minute rule conversion table (e.g. 8–22 min = 1 unit)"
                ),
                f"**Max Units Allowed:** {cap}",
                "**Total Billing Units:** Based on documented minutes",
            ]
        )
        return "\n\n".join(lines)

    if profile.billing_rule == FULL_BLOCK_REQUIRED:
        block = profile.block_minutes or 15
        cap = profile.max_units_allowed or "Per category rules"
        if is_session_limited_profile(profile):
            lines.extend(
                [
                    f"**Unit Requirement:** {block} minutes (full block)",
                    (
                        f"**Calculation:** Bill **1 unit** when the full {block}-minute "
                        f"block is completed"
                    ),
                    f"**Max Units Allowed:** {cap}",
                    "**Total Billing Units:** 1",
                ]
            )
        else:
            lines.extend(
                [
                    f"**Unit Requirement:** {block} minutes (full block)",
                    (
                        f"**Calculation:** Documented minutes / {block} minutes per unit "
                        f"(full blocks only, no partial credit)"
                    ),
                    f"**Max Units Allowed:** {cap}",
                    "**Total Billing Units:** Based on documented minutes",
                ]
            )
        return "\n\n".join(lines)

    if profile.billing_rule == TIME_BAND_SELECT:
        band = profile.time_band or profile.min_time_for_1_unit
        lines.extend(
            [
                f"**Unit Requirement:** Time band — {band}",
                "**Calculation:** Select the one code that matches total documented time",
                f"**Max Units Allowed:** {profile.max_units_allowed or '1 (pick one code)'}",
                "**Total Billing Units:** 1 if time falls in the band, otherwise 0",
            ]
        )
        return "\n\n".join(lines)

    if profile.billing_rule == AREA_BASED:
        unit_size = profile.area_unit_sq_cm or "area-based"
        lines.extend(
            [
                f"**Unit Requirement:** {unit_size} sq cm per unit"
                if isinstance(unit_size, int)
                else "**Unit Requirement:** Area-based (sq cm)",
                "**Calculation:** Wound size in sq cm determines units — not time",
                f"**Max Units Allowed:** {profile.max_units_allowed or 'Per category rules'}",
                "**Total Billing Units:** Based on documented wound area",
            ]
        )
        return "\n\n".join(lines)

    lines.extend(
        [
            f"**Unit Requirement:** {rule_label}",
            "**Calculation:** Follow the billing rule for this CPT category",
            "**Total Billing Units:** Based on documented service details",
        ]
    )
    return "\n\n".join(lines)


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


def _format_result_lines(result: UnitCalculationResult) -> list[str]:
    unit_label = "unit" if result.units == 1 else "units"
    lines = [
        f"**CPT {result.cpt_code}** — **{result.units}** {unit_label}",
        f"- Billing rule: `{result.billing_rule}`",
    ]
    if result.billing_rule in UNTIMED_RULES:
        lines.append("- Input time does not change unit count for this CPT.")
    elif result.billing_rule == AREA_BASED and result.area_sq_cm is not None:
        lines.append(f"- Wound area: **{result.area_sq_cm} sq cm**")
    else:
        lines.append(f"- Documented time: **{result.minutes} minutes**")
    if result.notes:
        lines.append(f"- {result.notes}")
    return lines


def _format_standard_unit_answer(
    result: UnitCalculationResult,
    profile: CodeBillingProfile | None,
    *,
    include_title: bool = True,
    concise: bool = False,
) -> str:
    if concise:
        label = "Medicare Units" if result.billing_rule == EIGHT_MINUTE_RULE else "Units"
        if result.billing_rule in UNTIMED_RULES:
            return f"**{result.cpt_code}:** **{result.units}** unit (untimed)."
        return f"**{label}:** **{result.units}**."

    lines = []
    if include_title:
        lines.append("**Billing Unit Calculation**")
    else:
        lines.append(f"### CPT {result.cpt_code}")
    lines.append(f"**Documented Time:** {result.minutes} minutes")

    calculated = (
        result.calculated_units
        if result.calculated_units is not None
        else result.units
    )

    if result.billing_rule == FULL_BLOCK_REQUIRED:
        block = profile.block_minutes if profile and profile.block_minutes else 15
        session_limited = profile is not None and is_session_limited_profile(profile)
        if session_limited:
            if result.units == 1:
                calc_line = (
                    f"**Calculation:** {result.minutes} minutes meets full "
                    f"{block}-minute block = **1 unit**"
                )
            else:
                calc_line = (
                    f"**Calculation:** {result.minutes} minutes does not meet full "
                    f"{block}-minute block = **0 units**"
                )
            lines.extend(
                [
                    f"**Unit Requirement:** {block} minutes (full block)",
                    calc_line,
                ]
            )
            if profile and profile.max_units_allowed:
                lines.append(f"**Max Units Allowed:** {profile.max_units_allowed}")
            lines.append(f"**Total Billing Units:** {result.units}")
            return "\n\n".join(lines)

        lines.extend(
            [
                f"**Unit Requirement:** {block} minutes (full block)",
                (
                    f"**Calculation:** {result.minutes} minutes / {block} minutes per unit "
                    f"= {calculated} units"
                ),
            ]
        )
        if profile and profile.max_units_allowed and calculated != result.units:
            lines.append(f"**Max Units Allowed:** {profile.max_units_allowed}")
        lines.append(f"**Total Billing Units:** {result.units}")
        return "\n\n".join(lines)

    if result.billing_rule == EIGHT_MINUTE_RULE:
        segment = profile.segment_size_minutes if profile else 15
        threshold = profile.unit_threshold_minutes if profile else 8
        lines.extend(
            [
                (
                    f"**Unit Requirement:** {segment}-minute timed segment "
                    f"({threshold}-minute remainder rule)"
                ),
                (
                    f"**Calculation:** {result.minutes} minutes under 8-minute rule "
                    f"= {calculated} units"
                ),
            ]
        )
        if profile and profile.max_units_allowed and calculated != result.units:
            lines.append(f"**Max Units Allowed:** {profile.max_units_allowed}")
        lines.append(f"**Total Billing Units:** {result.units}")
        return "\n\n".join(lines)

    if result.billing_rule in UNTIMED_RULES:
        requirement = (
            profile.max_units_allowed.replace("_", " ")
            if profile and profile.max_units_allowed
            else "1 unit when service is documented"
        )
        if profile and (
            profile.max_units_cap == 1
            or "1 per day" in (profile.max_units_allowed or "").lower()
        ):
            calc_note = (
                f"**Calculation:** Untimed / {requirement} — minutes and number of "
                "same-day sessions do not increase units = **1 unit**"
            )
        else:
            calc_note = f"**Calculation:** Bill **1 unit** ({requirement})"
        lines.extend(
            [
                "**Unit Requirement:** Untimed — time does not determine units",
                calc_note,
            ]
        )
        if profile and profile.max_units_allowed:
            lines.append(f"**Max Units Allowed:** {requirement}")
        lines.append("**Total Billing Units:** 1")
        return "\n\n".join(lines)

    if result.billing_rule == TIME_BAND_SELECT:
        band = profile.time_band if profile else result.notes
        lines.extend(
            [
                f"**Unit Requirement:** Time band — {band}",
                (
                    f"**Calculation:** {result.minutes} minutes "
                    f"{'qualifies' if result.units else 'does not qualify'} "
                    f"= {result.units} units"
                ),
                f"**Total Billing Units:** {result.units}",
            ]
        )
        return "\n\n".join(lines)

    if result.billing_rule == AREA_BASED:
        if result.area_sq_cm is None:
            lines.extend(
                [
                    "**Unit Requirement:** Area-based (sq cm)",
                    "**Calculation:** Wound size in sq cm is required to calculate units",
                    "**Total Billing Units:** 0",
                ]
            )
        else:
            unit_size = profile.area_unit_sq_cm if profile and profile.area_unit_sq_cm else 20
            lines.extend(
                [
                    f"**Unit Requirement:** {unit_size} sq cm per unit",
                    (
                        f"**Calculation:** {result.area_sq_cm} sq cm / "
                        f"{unit_size} sq cm per unit = {calculated} units"
                    ),
                ]
            )
            if profile and profile.max_units_allowed and calculated != result.units:
                lines.append(f"**Max Units Allowed:** {profile.max_units_allowed}")
            lines.append(f"**Total Billing Units:** {result.units}")
        return "\n\n".join(lines)

    lines.extend(
        [
            f"**Unit Requirement:** {result.billing_rule}",
            f"**Calculation:** {result.minutes} minutes = {result.units} units",
            f"**Total Billing Units:** {result.units}",
        ]
    )
    return "\n\n".join(lines)


def _format_pooled_eight_minute_answer(
    entries: list[CptTimeEntry],
    profile: CodeBillingProfile | None,
    pooled_units: int,
) -> str:
    total_minutes = sum(entry.minutes for entry in entries)
    segment = profile.segment_size_minutes if profile else 15
    threshold = profile.unit_threshold_minutes if profile else 8
    minute_parts = " + ".join(str(entry.minutes) for entry in entries)
    code_list = ", ".join(entry.cpt_code for entry in entries)
    return "\n\n".join(
        [
            "**Billing Unit Calculation**",
            f"**Documented Time:** {total_minutes} minutes ({minute_parts})",
            (
                f"**Unit Requirement:** {segment}-minute timed segment "
                f"({threshold}-minute remainder rule)"
            ),
            (
                f"**Calculation:** CMS pooled minutes for {code_list} = "
                f"{total_minutes} minutes → {pooled_units} units"
            ),
            f"**Total Billing Units:** {pooled_units}",
        ]
    )


def _build_unit_calculation_answer(entries: list[CptTimeEntry], question: str) -> str:
    from rag.response_style import wants_calculation_steps, wants_concise

    store = get_billing_category_store()
    methodology = detect_rule_methodology(question)
    area_sq_cm = parse_area_sq_cm(question)
    concise = wants_concise(question) and not wants_calculation_steps(question)

    if is_rule_comparison_question(question) and _all_eight_minute_rule(entries):
        from rag.response_style import wants_calculation_steps as _steps
        from rag.response_style import wants_concise as _concise

        cms = calculate_cms_pooled_units(entries)
        ama = calculate_ama_units(entries)
        cms_total = sum(cms.values())
        ama_total = sum(ama.values())
        if _concise(question) and not _steps(question):
            why = (
                "CMS pools timed minutes; AMA calculates each CPT independently."
                if cms_total != ama_total
                else "Both methods yield the same units for this scenario."
            )
            return (
                "| Rule | Units |\n"
                "|------|------:|\n"
                f"| Medicare (CMS) | {cms_total} |\n"
                f"| AMA | {ama_total} |\n\n"
                f"{why}"
            )
        return _format_comparison_answer(entries)

    eight_minute_entries = [
        entry
        for entry in entries
        if (profile := store.get_profile(entry.cpt_code))
        and profile.billing_rule == EIGHT_MINUTE_RULE
    ]
    other_entries = [entry for entry in entries if entry not in eight_minute_entries]

    if methodology == "CMS" and len(eight_minute_entries) > 1 and not other_entries:
        profile = store.get_profile(eight_minute_entries[0].cpt_code)
        pooled_units = sum(calculate_cms_pooled_units(eight_minute_entries).values())
        if concise:
            return f"**Medicare Units:** **{pooled_units}**."
        return _format_pooled_eight_minute_answer(
            eight_minute_entries, profile, pooled_units
        )

    if len(entries) == 1:
        entry = entries[0]
        profile = store.get_profile(entry.cpt_code)
        result = calculate_units_for_entry(
            entry,
            profile,
            methodology=methodology,
            area_sq_cm=area_sq_cm,
        )
        answer = _format_standard_unit_answer(result, profile, concise=concise)
        if (
            not concise
            and mentions_same_day_multiple_occurrences(question)
            and profile
            and (
                profile.max_units_cap == 1
                or "1 per day" in (profile.max_units_allowed or "").lower()
            )
        ):
            answer += (
                "\n\nEven if performed more than once the same day, "
                "billable units remain **1** for this code."
            )
        return answer

    results: list[UnitCalculationResult] = []

    if methodology == "CMS" and len(eight_minute_entries) > 1:
        pooled = calculate_cms_pooled_units(eight_minute_entries)
        primary_code = max(pooled, key=pooled.get)
        total_minutes = sum(entry.minutes for entry in eight_minute_entries)
        pooled_units = pooled[primary_code]
        results.append(
            UnitCalculationResult(
                cpt_code=primary_code,
                minutes=total_minutes,
                units=pooled_units,
                billing_rule=EIGHT_MINUTE_RULE,
                strategy="cms_8_minute_rule_pooled",
                notes=(
                    "CMS pooled "
                    + ", ".join(entry.cpt_code for entry in eight_minute_entries)
                    + f" = {total_minutes} minutes."
                ),
            )
        )
        pooled_codes = {entry.cpt_code for entry in eight_minute_entries}
        for entry in entries:
            if entry.cpt_code in pooled_codes:
                continue
            profile = store.get_profile(entry.cpt_code)
            results.append(
                calculate_units_for_entry(
                    entry,
                    profile,
                    methodology=methodology,
                    area_sq_cm=area_sq_cm,
                )
            )
    else:
        for entry in entries:
            profile = store.get_profile(entry.cpt_code)
            results.append(
                calculate_units_for_entry(
                    entry,
                    profile,
                    methodology=methodology,
                    area_sq_cm=area_sq_cm,
                )
            )

    total_units = sum(result.units for result in results)
    if concise:
        parts = [
            f"**{r.cpt_code}:** {r.units}" for r in results
        ]
        return f"{'; '.join(parts)}. **Total: {total_units}**."

    sections: list[str] = ["**Billing Unit Calculation**", ""]
    for result in results:
        profile = store.get_profile(result.cpt_code)
        sections.append(
            _format_standard_unit_answer(result, profile, include_title=False)
        )
        sections.append("")

    if mentions_same_day_multiple_occurrences(question):
        sections.append(
            "Same-day repeat sessions were considered. For codes limited to "
            "**1 per day**, additional morning/evening occurrences do not add units."
        )
        sections.append("")

    sections.append(
        f"**Combined Total Billing Units:** **{total_units}**"
    )
    return "\n".join(sections)
