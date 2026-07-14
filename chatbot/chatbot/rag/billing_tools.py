"""Deterministic billing lookup tools backed by structured JSON data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.billing_data import BillingDataStore

_MODIFIER_REASONS = {
    "0": "NCCI PTP edit - not billable together even with a modifier.",
    "1": "NCCI PTP edit - may be billable with an appropriate modifier when clinically distinct.",
    "9": "NCCI PTP edit marked not applicable in the source data.",
}

_TIMED_BILLING_RULES = frozenset({"8_minute_rule", "full_block_required"})
_EIGHT_MINUTE_RULES = frozenset({"8_minute_rule"})


def _billing_rule_flags(billing_rule: str) -> tuple[bool | None, bool | None]:
    if not billing_rule:
        return None, None
    rule = billing_rule.strip().lower()
    if rule in _EIGHT_MINUTE_RULES:
        return True, True
    if rule in _TIMED_BILLING_RULES:
        return True, False
    if rule.startswith("untimed"):
        return False, False
    return None, None


class BillingTools:
    def __init__(self, json_dir: Path):
        self.store = BillingDataStore(json_dir)

    def lookup_mue(self, cpt_code: str) -> dict[str, Any]:
        record = self.store.get_mue(cpt_code)
        if not record:
            return {"found": False, "cpt_code": cpt_code}

        mue = record.get("mue", {})
        adjudication_level = mue.get("adjudication_level", "")
        description = mue.get("description", "")
        if adjudication_level:
            description = f"{description} ({adjudication_level})".strip()

        return {
            "found": True,
            "cpt_code": cpt_code,
            "limit": mue.get("limit"),
            "adjudication": mue.get("adjudication"),
            "description": description,
        }

    def check_ncci(self, cpt1: str, cpt2: str) -> dict[str, Any]:
        edits = self._find_ptp_edits(cpt1, cpt2)
        if not edits:
            return {
                "found": True,
                "cpt_codes": [cpt1, cpt2],
                "allowed": True,
                "modifier59_required": False,
                "edit_type": "none",
                "reason": "No NCCI PTP edit found between these codes in the knowledge base.",
                "edits": [],
            }

        has_modifier_override = any(edit["modifier_indicator"] == "1" for edit in edits)
        has_blocking_edit = any(edit["modifier_indicator"] == "0" for edit in edits)
        allowed = has_modifier_override or not has_blocking_edit
        modifier_required = has_modifier_override

        primary_edit = edits[0]
        return {
            "found": True,
            "cpt_codes": [cpt1, cpt2],
            "allowed": allowed,
            "modifier59_required": modifier_required,
            "edit_type": "PTP",
            "reason": self._summarize_ncci_reason(edits, allowed, modifier_required),
            "edits": edits,
            "column1_code": primary_edit.get("column1_code"),
            "column2_code": primary_edit.get("column2_code"),
            "modifier_indicator": primary_edit.get("modifier_indicator"),
        }

    def lookup_icd(self, cpt_code: str) -> dict[str, Any]:
        record = self.store.get_icd10(cpt_code)
        if not record:
            return {"found": False, "cpt_code": cpt_code, "valid_icd10": []}

        codes = [
            entry["code"]
            for entry in record.get("valid_icd10_codes", [])
            if isinstance(entry, dict) and entry.get("code")
        ]
        return {
            "found": True,
            "cpt_code": cpt_code,
            "valid_icd10": codes,
            "count": len(codes),
        }

    def validate_icd10(self, cpt_code: str, icd10_code: str) -> dict[str, Any]:
        record = self.store.get_icd10(cpt_code)
        normalized_icd = icd10_code.strip().upper()
        if not record:
            return {
                "found": False,
                "cpt_code": cpt_code,
                "icd10_code": normalized_icd,
                "valid": False,
            }

        valid_codes = {
            str(entry["code"]).upper()
            for entry in record.get("valid_icd10_codes", [])
            if isinstance(entry, dict) and entry.get("code")
        }
        is_valid = normalized_icd in valid_codes
        return {
            "found": True,
            "cpt_code": cpt_code,
            "icd10_code": normalized_icd,
            "valid": is_valid,
            "total_mapped_codes": len(valid_codes),
        }

    def explain_billing_rules(self, cpt_code: str) -> dict[str, Any]:
        cpt_info = self.lookup_cpt(cpt_code)
        if not cpt_info.get("found"):
            return {"found": False, "cpt_code": cpt_code}

        billing_rule = ""
        general = self.store.get_general(cpt_code)
        if general:
            billing_rule = str(general.get("billingRule", "")).strip()

        knowledge = self.store.get_knowledge(cpt_code)
        billing_guidance = ""
        if knowledge:
            notes = knowledge.get("notes", "")
            if notes:
                billing_guidance = str(notes)

        return {
            "found": True,
            "cpt_code": cpt_code,
            "timed": cpt_info.get("timed"),
            "eight_minute_rule": cpt_info.get("eight_minute_rule"),
            "billing_rule": billing_rule,
            "billing_guidance": billing_guidance,
            "billable_guidance": cpt_info.get("billable_guidance"),
            "description": cpt_info.get("description"),
        }

    def lookup_aoc(self, cpt_code: str) -> dict[str, Any]:
        record = self.store.get_aoc(cpt_code)
        if not record:
            return {"found": False, "cpt_code": cpt_code}

        billing_rule = str(record.get("billingRule", "")).strip()
        timed, _ = _billing_rule_flags(billing_rule)

        return {
            "found": True,
            "cpt_code": cpt_code,
            "is_addon_code": record.get("isAddonCode"),
            "parent_code": record.get("parentCode"),
            "addon_codes_allowed": record.get("addonCodesAllowed", []),
            "billing_rule": billing_rule,
            "billing_time": record.get("billingTime"),
            "is_timed": timed,
        }

    def lookup_cpt(self, cpt_code: str) -> dict[str, Any]:
        general = self.store.get_general(cpt_code)
        knowledge = self.store.get_knowledge(cpt_code)
        if not general and not knowledge:
            return {"found": False, "cpt_code": cpt_code}

        timed = None
        description = ""
        eight_minute_rule = None
        billing_rule = ""
        if general:
            description = general.get("description", "")
            billing_rule = str(general.get("billingRule", "")).strip()
            timed, eight_minute_rule = _billing_rule_flags(billing_rule)

        therapy_discipline: list[str] = []
        documentation_notes: list[str] = []
        billing_notes: list[str] = []
        short_name = ""
        service_category = ""
        if knowledge:
            short_name = knowledge.get("label", "")
            therapy_discipline = knowledge.get("disciplines", []) or []
            notes = knowledge.get("notes", "")
            if notes:
                billing_notes = [str(notes)]
            if not description:
                description = short_name

        return {
            "found": True,
            "cpt_code": cpt_code,
            "short_name": short_name,
            "description": description,
            "service_category": service_category,
            "billing_rule": billing_rule,
            "timed": timed,
            "untimed": not timed if timed is not None else None,
            "eight_minute_rule": eight_minute_rule,
            "therapy_discipline": therapy_discipline,
            "documentation_notes": documentation_notes,
            "billing_notes": billing_notes,
            "billable_guidance": (
                "Billable when medical necessity, documentation, and payer requirements are met."
                if general or knowledge
                else None
            ),
        }

    def summarize_ncci_restrictions(self, cpt_code: str) -> dict[str, Any]:
        record = self.store.get_ptp(cpt_code)
        if not record:
            return {"found": False, "cpt_code": cpt_code}

        ptp = record.get("ptp", {})
        bundles_others = ptp.get("bundles_others", [])
        bundled_into = ptp.get("bundled_into", [])
        modifier_required_count = sum(
            1
            for entry in bundles_others + bundled_into
            if str(entry.get("modifier_indicator")) == "1"
        )
        return {
            "found": True,
            "cpt_code": cpt_code,
            "ptp_edit_count": len(bundles_others) + len(bundled_into),
            "modifier_override_edits": modifier_required_count,
            "summary": (
                f"CPT {cpt_code} participates in {len(bundles_others) + len(bundled_into)} "
                "NCCI PTP relationships in Medexa's billing rules."
            ),
        }

    def run(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "lookup_mue":
            return self.lookup_mue(params["cpt_code"])
        if tool_name == "check_ncci":
            return self.check_ncci(params["cpt1"], params["cpt2"])
        if tool_name == "lookup_icd":
            return self.lookup_icd(params["cpt_code"])
        if tool_name == "validate_icd10":
            return self.validate_icd10(params["cpt_code"], params["icd10_code"])
        if tool_name == "explain_billing_rules":
            return self.explain_billing_rules(params["cpt_code"])
        if tool_name == "lookup_aoc":
            return self.lookup_aoc(params["cpt_code"])
        if tool_name == "lookup_cpt":
            return self.lookup_cpt(params["cpt_code"])
        if tool_name == "summarize_ncci_restrictions":
            return self.summarize_ncci_restrictions(params["cpt_code"])
        raise ValueError(f"Unknown billing tool: {tool_name}")

    def _find_ptp_edits(self, cpt1: str, cpt2: str) -> list[dict[str, Any]]:
        edits: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for column1, column2 in ((cpt1, cpt2), (cpt2, cpt1)):
            record = self.store.get_ptp(column1)
            if not record:
                continue
            ptp = record.get("ptp", {})
            for entry in ptp.get("bundles_others", []):
                if str(entry.get("bundled_code")) == column2:
                    indicator = str(entry.get("modifier_indicator", ""))
                    key = (column1, column2, indicator)
                    if key in seen:
                        continue
                    seen.add(key)
                    edits.append(
                        {
                            "column1_code": column1,
                            "column2_code": column2,
                            "modifier_indicator": indicator,
                            "direction": f"{column1} (column 1) / {column2} (column 2)",
                        }
                    )
            for entry in ptp.get("bundled_into", []):
                if str(entry.get("primary_code")) == column2:
                    indicator = str(entry.get("modifier_indicator", ""))
                    key = (column2, column1, indicator)
                    if key in seen:
                        continue
                    seen.add(key)
                    edits.append(
                        {
                            "column1_code": column2,
                            "column2_code": column1,
                            "modifier_indicator": indicator,
                            "direction": f"{column2} (column 1) / {column1} (column 2)",
                        }
                    )
        return edits

    def _summarize_ncci_reason(
        self, edits: list[dict[str, Any]], allowed: bool, modifier_required: bool
    ) -> str:
        if not edits:
            return "No NCCI PTP edit found between these codes."

        parts: list[str] = []
        for edit in edits[:3]:
            indicator = edit.get("modifier_indicator", "")
            base = _MODIFIER_REASONS.get(
                indicator,
                f"NCCI PTP edit with modifier indicator {indicator}.",
            )
            parts.append(f"{edit['direction']}: {base}")

        if modifier_required and allowed:
            parts.append(
                "Distinct procedural services may require modifier -59 or an applicable NCCI modifier with supporting documentation."
            )
        elif not allowed:
            parts.append("These codes should not be reported together per the PTP edit.")

        return " ".join(parts)
