"""Area-based area-based wound billing tool with minimal procedure-family clarification."""

from __future__ import annotations

import re

from rag.billing_categories import (
    AreaProcedureFamily,
    CodeBillingProfile,
    get_billing_category_store,
)
from rag.billing_engine import AREA_BASED, apply_max_units_cap
from rag.category_tools import CategoryToolResult, profile_payload


class AreaBasedBillingTool:
    name = "area_based_billing_tool"
    billing_rules = (AREA_BASED,)

    def supports(self, billing_rule: str) -> bool:
        return billing_rule == AREA_BASED

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
        store = get_billing_category_store()
        family = self._detect_family(question, codes)
        category = store.get_category("F")

        if area_sq_cm is None and not codes:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer="",
                needs_clarification=True,
                clarification=(
                    "Area-based wound billing requires wound size in sq cm. "
                    "What is the total wound surface area?"
                ),
                structured={"category_id": "F"},
            )

        if family is None and not codes:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer="",
                needs_clarification=True,
                clarification=self._procedure_clarification(area_sq_cm),
                structured={
                    "category_id": "F",
                    "category_description": category.description if category else "",
                    "area_sq_cm": area_sq_cm,
                    "families": [
                        {"id": f.family_id, "label": f.label}
                        for f in store.area_procedure_families()
                    ],
                },
            )

        if family is None and codes:
            # Specific CPT already known - calculate against that profile.
            return self._calculate_for_codes(
                codes=codes,
                profiles=profiles,
                area_sq_cm=area_sq_cm,
            )

        assert family is not None
        return self._calculate_for_family(family, area_sq_cm)

    def _procedure_clarification(self, area_sq_cm: int | None) -> str:
        size_line = (
            f"Documented wound size: **{area_sq_cm} sq cm**.\n\n"
            if area_sq_cm is not None
            else ""
        )
        return (
            "I can help determine the correct CPT code using area-based "
            "wound billing rules.\n\n"
            f"{size_line}"
            "Which procedure was performed?\n\n"
            "• Selective debridement\n"
            "• Traditional negative pressure wound therapy\n"
            "• Disposable negative pressure wound therapy"
        )

    def _detect_family(
        self, question: str, codes: list[str]
    ) -> AreaProcedureFamily | None:
        store = get_billing_category_store()
        lowered = question.lower()
        code_set = {c.upper() for c in codes}

        # Prefer disposable NPWT over generic NPWT keywords.
        disposable = next(
            f
            for f in store.area_procedure_families()
            if f.family_id == "disposable_npwt"
        )
        if any(k in lowered for k in disposable.keywords if k == "disposable") or (
            "disposable" in lowered and ("npwt" in lowered or "vac" in lowered)
        ):
            return disposable
        if code_set & set(disposable.primary_codes + disposable.addon_codes):
            return disposable

        for family in store.area_procedure_families():
            if code_set & set(family.primary_codes + family.addon_codes):
                return family

        # Keyword match - skip bare "disposable" already handled.
        best: AreaProcedureFamily | None = None
        for family in store.area_procedure_families():
            if family.family_id == "disposable_npwt":
                continue
            if any(keyword in lowered for keyword in family.keywords):
                # Avoid matching generic "npwt" when disposable already ruled out.
                if family.family_id == "traditional_npwt" and "disposable" in lowered:
                    continue
                best = family
                if family.family_id == "selective_debridement" and "debrid" in lowered:
                    return family
        return best

    def _calculate_for_family(
        self, family: AreaProcedureFamily, area_sq_cm: int | None
    ) -> CategoryToolResult:
        if area_sq_cm is None:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer="",
                needs_clarification=True,
                clarification=(
                    f"For **{family.label}**, what is the total wound surface "
                    "area in sq cm?"
                ),
                structured={"family": family.family_id},
            )

        store = get_billing_category_store()
        rules = family.size_rules

        if family.family_id == "selective_debridement":
            unit = int(rules.get("unit_sq_cm", 20))
            base = rules["base_code"]
            addon = rules["addon_code"]
            if area_sq_cm <= 0:
                units_base, units_addon = 0, 0
            elif area_sq_cm <= unit:
                units_base, units_addon = 1, 0
            else:
                units_base = 1
                remainder = area_sq_cm - unit
                units_addon = (remainder + unit - 1) // unit
            answer = (
                "**Billing Unit Calculation - Area-Based**\n\n"
                f"**Procedure:** {family.label}\n\n"
                f"**Documented Area:** {area_sq_cm} sq cm\n\n"
                f"**Unit Requirement:** {unit} sq cm per unit "
                f"(primary **{base}**, add-on **{addon}**)\n\n"
                f"**Calculation:** First {unit} sq cm -> **{base} x {units_base}**; "
                f"remaining {max(area_sq_cm - unit, 0)} sq cm -> "
                f"**{addon} x {units_addon}**\n\n"
                f"**Total Billing Units:** {units_base + units_addon} "
                f"({base}: {units_base}, {addon}: {units_addon})"
            )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer=answer,
                structured={
                    "tool": self.name,
                    "family": family.family_id,
                    "area_sq_cm": area_sq_cm,
                    "codes": {
                        base: units_base,
                        addon: units_addon,
                    },
                },
            )

        if family.family_id == "traditional_npwt":
            threshold = int(rules.get("threshold_sq_cm", 50))
            code = (
                rules["large_code"]
                if area_sq_cm > threshold
                else rules["small_code"]
            )
            profile = store.get_profile(code)
            answer = (
                "**Billing Unit Calculation - Area-Based**\n\n"
                f"**Procedure:** {family.label}\n\n"
                f"**Documented Area:** {area_sq_cm} sq cm\n\n"
                f"**Selected CPT:** **{code}** "
                f"({'≥' if area_sq_cm > threshold else '≤'}{threshold} sq cm rule)\n\n"
                f"**Max Units Allowed:** "
                f"{profile.max_units_allowed if profile else '1 per wound'}\n\n"
                "**Total Billing Units:** **1**"
            )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer=answer,
                structured={
                    "tool": self.name,
                    "family": family.family_id,
                    "area_sq_cm": area_sq_cm,
                    "selected_code": code,
                    "units": 1,
                    "profile": profile_payload(profile) if profile else {},
                },
            )

        # Disposable NPWT
        threshold = int(rules.get("threshold_sq_cm", 50))
        base = rules["base_code"]
        addon = rules["addon_code"]
        addon_unit = int(rules.get("addon_unit_sq_cm", 50))
        if area_sq_cm <= threshold:
            units_base, units_addon = 1, 0
            calc = (
                f"{area_sq_cm} sq cm ≤ {threshold} -> bill **{base} x 1** "
                "(no add-on)"
            )
        else:
            units_base = 1
            remainder = area_sq_cm - threshold
            units_addon = (remainder + addon_unit - 1) // addon_unit
            calc = (
                f"First {threshold} sq cm -> **{base} x 1**; "
                f"remaining {remainder} sq cm -> **{addon} x {units_addon}** "
                f"({addon_unit} sq cm per add-on unit)"
            )
        answer = (
            "**Billing Unit Calculation - Area-Based**\n\n"
            f"**Procedure:** {family.label}\n\n"
            f"**Documented Area:** {area_sq_cm} sq cm\n\n"
            f"**Calculation:** {calc}\n\n"
            f"**Total Billing Units:** {units_base + units_addon} "
            f"({base}: {units_base}, {addon}: {units_addon})"
        )
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=AREA_BASED,
            answer=answer,
            structured={
                "tool": self.name,
                "family": family.family_id,
                "area_sq_cm": area_sq_cm,
                "codes": {base: units_base, addon: units_addon},
            },
        )

    def _calculate_for_codes(
        self,
        *,
        codes: list[str],
        profiles: list[CodeBillingProfile],
        area_sq_cm: int | None,
    ) -> CategoryToolResult:
        store = get_billing_category_store()
        if area_sq_cm is None:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer="",
                needs_clarification=True,
                clarification="What is the total wound surface area in sq cm?",
            )

        profile = profiles[0] if profiles else store.get_profile(codes[0])
        if profile is None:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer="",
                needs_clarification=True,
                clarification="I couldn't match that wound CPT to area-based wound billing.",
            )

        # Per-wound codes: always 1 when size qualifies for band.
        if profile.max_units_cap == 1 or "per wound" in (
            profile.max_units_allowed or ""
        ).lower():
            answer = (
                "**Billing Unit Calculation - Area-Based**\n\n"
                f"**CPT:** {profile.cpt_code}\n\n"
                f"**Documented Area:** {area_sq_cm} sq cm\n\n"
                f"**Unit Requirement:** {profile.min_time_for_1_unit or profile.billing_time}\n\n"
                f"**Max Units Allowed:** {profile.max_units_allowed}\n\n"
                "**Total Billing Units:** **1**"
            )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=AREA_BASED,
                answer=answer,
                structured={
                    "tool": self.name,
                    "profile": profile_payload(profile),
                    "area_sq_cm": area_sq_cm,
                    "units": 1,
                },
            )

        unit_size = profile.area_unit_sq_cm or 20
        calculated = max(1, (area_sq_cm + unit_size - 1) // unit_size)
        units = apply_max_units_cap(calculated, profile)
        answer = (
            "**Billing Unit Calculation - Area-Based**\n\n"
            f"**CPT:** {profile.cpt_code}\n\n"
            f"**Documented Area:** {area_sq_cm} sq cm\n\n"
            f"**Unit Requirement:** {unit_size} sq cm per unit\n\n"
            f"**Calculation:** {area_sq_cm} / {unit_size} = {calculated} units\n\n"
        )
        if units != calculated and profile.max_units_allowed:
            answer += (
                f"**Max Units Allowed:** {profile.max_units_allowed} "
                f"(capped from {calculated} to {units})\n\n"
            )
        answer += f"**Total Billing Units:** **{units}**"
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=AREA_BASED,
            answer=answer,
            structured={
                "tool": self.name,
                "profile": profile_payload(profile),
                "area_sq_cm": area_sq_cm,
                "calculated_units": calculated,
                "units": units,
            },
        )


def is_area_wound_question(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"\d+\s*(?:sq\.?\s*cm|square\s*centimeters?)", lowered):
        return True
    wound_hints = (
        "wound",
        "debrid",
        "npwt",
        "negative pressure",
        "wound vac",
    )
    ask_code = any(
        phrase in lowered
        for phrase in (
            "which cpt",
            "what cpt",
            "which code",
            "what code",
            "what should i bill",
            "billing unit",
            "how many units",
        )
    )
    return ask_code and any(hint in lowered for hint in wound_hints)
