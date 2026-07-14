"""Add-on billing tool — validate primary code before allowing add-on units."""

from __future__ import annotations

from rag.billing_categories import CodeBillingProfile, get_billing_category_store
from rag.billing_data import BillingDataStore
from rag.category_tools import CategoryToolResult, profile_payload


def _aoc_store() -> BillingDataStore:
    from app.config import settings

    return BillingDataStore(settings.json_dir)


class AddonBillingTool:
    name = "addon_billing_tool"
    billing_rules = ("addon",)

    def supports(self, billing_rule: str) -> bool:
        return billing_rule == "addon"

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
        if not codes:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule="addon",
                answer="",
                needs_clarification=True,
                clarification="Which add-on CPT code are you asking about?",
            )

        data = _aoc_store()
        store = get_billing_category_store()
        target = codes[0]
        aoc = data.get_aoc(target) or {}

        is_addon = bool(aoc.get("isAddonCode"))
        parent = aoc.get("parentCode")
        allowed = aoc.get("addonCodesAllowed") or []

        profile = store.get_profile(target)
        if is_addon:
            if parent and parent.upper() not in {c.upper() for c in codes}:
                return CategoryToolResult(
                    tool_name=self.name,
                    billing_rule=profile.billing_rule if profile else "addon",
                    answer="",
                    needs_clarification=True,
                    clarification=(
                        f"**{target}** is an add-on code and requires primary "
                        f"code **{parent}** on the same claim.\n\n"
                        f"Was **{parent}** also billed for this encounter?"
                    ),
                    structured={
                        "tool": self.name,
                        "addon_code": target,
                        "required_primary": parent,
                        "profile": profile_payload(profile) if profile else {},
                    },
                )
            answer = (
                f"**{target}** is an add-on code.\n\n"
                f"• Required primary code: **{parent or 'see billing guidance'}**\n"
                f"• Category rule: "
                f"`{profile.billing_rule if profile else 'see category JSON'}`\n"
                f"• Max units: "
                f"{profile.max_units_allowed if profile else 'per category'}"
            )
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=profile.billing_rule if profile else "addon",
                answer=answer,
                structured={
                    "tool": self.name,
                    "addon_code": target,
                    "parent": parent,
                    "profile": profile_payload(profile) if profile else {},
                },
            )

        answer = (
            f"**{target}** is a primary code.\n\n"
            f"• Allowed add-on codes: "
            f"{', '.join(f'**{c}**' for c in allowed) if allowed else 'None listed'}\n"
            f"• Category rule: "
            f"`{profile.billing_rule if profile else 'see category JSON'}`"
        )
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=profile.billing_rule if profile else "addon",
            answer=answer,
            structured={
                "tool": self.name,
                "primary_code": target,
                "addon_codes_allowed": allowed,
                "profile": profile_payload(profile) if profile else {},
            },
        )
