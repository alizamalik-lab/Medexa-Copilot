"""Full-block timed category tool (Category B)."""

from __future__ import annotations

from rag.billing_categories import CodeBillingProfile
from rag.billing_engine import (
    FULL_BLOCK_REQUIRED,
    CptTimeEntry,
    _format_standard_unit_answer,
    calculate_units_for_entry,
    mentions_same_day_multiple_occurrences,
)
from rag.category_tools import CategoryToolResult, profile_payload


class FullBlockBillingTool:
    name = "full_block_billing_tool"
    billing_rules = (FULL_BLOCK_REQUIRED,)

    def supports(self, billing_rule: str) -> bool:
        return billing_rule == FULL_BLOCK_REQUIRED

    def run(
        self,
        *,
        question: str,
        codes: list[str],
        profiles: list[CodeBillingProfile],
        minutes_by_code: dict[str, int],
        area_sq_cm: int | None,
        methodology: str,
    ) -> CategoryToolResult:
        profile = profiles[0] if profiles else None
        code = codes[0] if codes else (profile.cpt_code if profile else None)
        if profile is None or code is None:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=FULL_BLOCK_REQUIRED,
                answer="",
                needs_clarification=True,
                clarification="Which full-block CPT code are you billing?",
            )

        minutes = minutes_by_code.get(code)
        if minutes is None:
            block = profile.block_minutes or 15
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=FULL_BLOCK_REQUIRED,
                answer=(
                    f"**Billing Rule:** Category {profile.category_id} — "
                    f"`full_block_required`\n\n"
                    f"**Block Size:** {block} minutes\n\n"
                    f"**Max Units Allowed:** {profile.max_units_allowed}\n\n"
                    "Provide documented minutes to calculate units "
                    "(no partial-block credit)."
                ),
                structured={"tool": self.name, "profile": profile_payload(profile)},
            )

        entry = CptTimeEntry(cpt_code=code, minutes=minutes)
        result = calculate_units_for_entry(
            entry, profile, methodology=methodology, area_sq_cm=area_sq_cm
        )
        answer = _format_standard_unit_answer(result, profile)
        if mentions_same_day_multiple_occurrences(question) and profile.max_units_cap == 1:
            answer += (
                "\n\nEven with multiple same-day sessions, this category allows "
                f"**{profile.max_units_allowed}**."
            )
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=FULL_BLOCK_REQUIRED,
            answer=answer,
            structured={
                "tool": self.name,
                "profile": profile_payload(profile),
                "units": result.units,
                "calculated_units": result.calculated_units,
                "minutes": minutes,
            },
        )
