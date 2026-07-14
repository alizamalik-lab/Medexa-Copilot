"""Timed 8-minute-rule category tool (Category A)."""

from __future__ import annotations

from rag.billing_categories import CodeBillingProfile
from rag.billing_engine import (
    EIGHT_MINUTE_RULE,
    CptTimeEntry,
    _format_pooled_eight_minute_answer,
    _format_standard_unit_answer,
    apply_max_units_cap,
    calculate_cms_pooled_units,
    calculate_units_for_entry,
)
from rag.category_tools import CategoryToolResult, profile_payload


class TimedBillingTool:
    name = "timed_billing_tool"
    billing_rules = (EIGHT_MINUTE_RULE,)

    def supports(self, billing_rule: str) -> bool:
        return billing_rule == EIGHT_MINUTE_RULE

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
        entries = [
            CptTimeEntry(cpt_code=code, minutes=minutes_by_code[code])
            for code in codes
            if code in minutes_by_code
        ]
        if not entries:
            profile = profiles[0] if profiles else None
            if profile is None:
                return CategoryToolResult(
                    tool_name=self.name,
                    billing_rule=EIGHT_MINUTE_RULE,
                    answer="",
                    needs_clarification=True,
                    clarification=(
                        "This CPT uses the CMS 8-Minute Rule. "
                        "How many minutes were documented?"
                    ),
                )
            guide = (
                f"**Billing Rule:** Category {profile.category_id} — "
                f"`8_minute_rule`\n\n"
                f"**Min for 1 unit:** {profile.min_time_for_1_unit}\n\n"
                f"**Max Units Allowed:** {profile.max_units_allowed}\n\n"
                "Provide documented timed minutes to calculate billable units."
            )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=EIGHT_MINUTE_RULE,
                answer=guide,
                structured={
                    "tool": self.name,
                    "profiles": [profile_payload(p) for p in profiles],
                },
            )

        if methodology == "CMS" and len(entries) > 1:
            pooled = calculate_cms_pooled_units(entries)
            pooled_units = sum(pooled.values())
            primary = max(pooled, key=pooled.get)
            primary_profile = next(
                (p for p in profiles if p.cpt_code == primary), profiles[0]
            )
            capped = apply_max_units_cap(pooled_units, primary_profile)
            answer = _format_pooled_eight_minute_answer(
                entries, primary_profile, capped
            )
            if capped != pooled_units and primary_profile.max_units_allowed:
                answer += (
                    f"\n\n**Cap Applied:** Calculated {pooled_units} units, "
                    f"limited to **{capped}** per category "
                    f"({primary_profile.max_units_allowed})."
                )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=EIGHT_MINUTE_RULE,
                answer=answer,
                structured={
                    "tool": self.name,
                    "methodology": methodology,
                    "units": capped,
                    "calculated_units": pooled_units,
                    "profiles": [profile_payload(p) for p in profiles],
                    "entries": [
                        {"cpt_code": e.cpt_code, "minutes": e.minutes} for e in entries
                    ],
                },
            )

        results = []
        total = 0
        for entry in entries:
            profile = next(
                (p for p in profiles if p.cpt_code == entry.cpt_code), None
            )
            result = calculate_units_for_entry(
                entry, profile, methodology=methodology, area_sq_cm=area_sq_cm
            )
            results.append(result)
            total += result.units

        if len(entries) == 1:
            answer = _format_standard_unit_answer(results[0], profile)
        else:
            lines = ["**Billing Unit Calculation**", ""]
            for result in results:
                profile = next(
                    (p for p in profiles if p.cpt_code == result.cpt_code), None
                )
                lines.append(
                    _format_standard_unit_answer(result, profile, include_title=False)
                )
                lines.append("")
            lines.append(f"**Combined Total Billing Units:** **{total}**")
            answer = "\n".join(lines)

        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=EIGHT_MINUTE_RULE,
            answer=answer,
            structured={
                "tool": self.name,
                "methodology": methodology,
                "total_units": total if len(entries) > 1 else results[0].units,
                "profiles": [profile_payload(p) for p in profiles],
                "results": [
                    {
                        "cpt_code": r.cpt_code,
                        "minutes": r.minutes,
                        "units": r.units,
                        "calculated_units": r.calculated_units,
                    }
                    for r in results
                ],
            },
        )
