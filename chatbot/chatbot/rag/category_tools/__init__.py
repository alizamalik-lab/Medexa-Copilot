"""Modular billing category tools (future Bedrock Agent compatible).

Each tool owns one billing-rule family from pt_ot_slp_billing_categories.json.
The LLM must never invent these rules — tools produce deterministic results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rag.billing_categories import CodeBillingProfile, get_billing_category_store


@dataclass
class CategoryToolResult:
    """Structured output shared by all category tools."""

    tool_name: str
    billing_rule: str
    answer: str
    structured: dict[str, Any] = field(default_factory=dict)
    clarification: str | None = None
    needs_clarification: bool = False
    sources: list[str] = field(default_factory=lambda: ["billing_category_engine"])

    @property
    def ok(self) -> bool:
        return not self.needs_clarification and bool(self.answer)


class CategoryTool(Protocol):
    name: str
    billing_rules: tuple[str, ...]

    def supports(self, billing_rule: str) -> bool: ...

    def run(
        self,
        *,
        question: str,
        codes: list[str],
        profiles: list[CodeBillingProfile],
        minutes_by_code: dict[str, int],
        area_sq_cm: int | None,
        methodology: str,
    ) -> CategoryToolResult: ...


def profile_payload(profile: CodeBillingProfile) -> dict[str, Any]:
    store = get_billing_category_store()
    return {
        "cpt_code": profile.cpt_code,
        "category_id": profile.category_id,
        "subcategory_id": profile.subcategory_id,
        "billing_rule": profile.billing_rule,
        "billing_rule_definition": store.rule_definition(profile.billing_rule),
        "min_requirement": profile.min_time_for_1_unit,
        "max_units_allowed": profile.max_units_allowed,
        "max_units_cap": profile.max_units_cap,
        "billing_time": profile.billing_time,
        "block_minutes": profile.block_minutes,
        "time_band": profile.time_band,
        "area_unit_sq_cm": profile.area_unit_sq_cm,
        "description": profile.description,
    }


ResultKind = Literal["answer", "clarification", "empty"]
