"""Format billing tool output for LLM explanation prompts."""

from __future__ import annotations

import json
from typing import Any


def format_tool_result_for_llm(tool_name: str, result: dict[str, Any]) -> str:
    payload = dict(result)
    if tool_name == "lookup_icd" and isinstance(payload.get("valid_icd10"), list):
        codes = payload["valid_icd10"]
        payload["valid_icd10_count"] = len(codes)
        if len(codes) > 12:
            payload["valid_icd10_examples"] = codes[:12]
            del payload["valid_icd10"]
    if tool_name == "validate_icd10":
        payload.pop("total_mapped_codes", None)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_combined_billing_data(
    tool_results: list[tuple[str, dict[str, Any]]],
    unit_calculation: dict[str, Any] | None = None,
) -> str:
    sections: list[str] = []
    if unit_calculation:
        sections.append(
            "Medicare/CMS unit calculation:\n"
            + json.dumps(
                {
                    "entries": unit_calculation.get("entries", []),
                    "answer": unit_calculation.get("answer", ""),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    for tool_name, result in tool_results:
        sections.append(
            f"{_friendly_section_name(tool_name)}:\n"
            + format_tool_result_for_llm(tool_name, result)
        )
    return "\n\n".join(sections)


def _friendly_section_name(tool_name: str) -> str:
    labels = {
        "lookup_mue": "MUE limit",
        "check_ncci": "NCCI edit check",
        "lookup_icd": "ICD-10 mapping",
        "validate_icd10": "ICD-10 validation",
        "lookup_aoc": "Add-on code rules",
        "lookup_cpt": "CPT billing details",
        "summarize_ncci_restrictions": "NCCI restrictions summary",
        "explain_billing_rules": "Billing rules",
    }
    return labels.get(tool_name, "Billing data")
