"""PT/OT/SLP billing category lookup from pt_ot_slp_billing_categories.json.

This JSON is Medexa's source of truth for billing rules. Profiles and
category metadata are loaded once and reused by deterministic category tools.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CATEGORIES_FILENAME = "pt_ot_slp_billing_categories.json"


@dataclass(frozen=True)
class CodeBillingProfile:
    cpt_code: str
    billing_rule: str
    category_id: str
    subcategory_id: str | None = None
    segment_size_minutes: int = 15
    unit_threshold_minutes: int = 8
    block_minutes: int | None = None
    time_band: str | None = None
    billing_time: str = ""
    area_unit_sq_cm: int | None = None
    max_units_cap: int | None = None
    min_time_for_1_unit: str = ""
    max_units_allowed: str = ""
    description: str = ""
    session_type: str = ""
    encounter_type: str = ""


@dataclass(frozen=True)
class SubcategoryMeta:
    subcategory_id: str
    billing_rule: str
    codes: tuple[str, ...]
    description: str = ""
    billing_time: str = ""
    min_time_for_1_unit: str = ""
    max_units_allowed: str = ""
    block_minutes: int | None = None
    time_band: str | None = None
    area_unit_sq_cm: int | None = None
    max_units_cap: int | None = None
    session_type: str = ""
    encounter_type: str = ""


@dataclass(frozen=True)
class CategoryMeta:
    category_id: str
    billing_rule: str
    description: str
    codes: tuple[str, ...] = ()
    subcategories: tuple[SubcategoryMeta, ...] = ()
    segment_size_minutes: int = 15
    unit_threshold_minutes: int = 8
    min_time_for_1_unit: str = ""
    max_units_allowed: str = ""
    billing_time: str = ""
    max_units_cap: int | None = None


@dataclass(frozen=True)
class AreaProcedureFamily:
    """Logical wound CPT family within Category F for minimal clarification."""

    family_id: str
    label: str
    primary_codes: tuple[str, ...]
    addon_codes: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    size_rules: dict[str, Any] = field(default_factory=dict)


def _parse_max_units_cap(value: str) -> int | None:
    if not value:
        return None
    lowered = value.strip().lower()
    if lowered.startswith("1 ") or lowered == "1":
        return 1
    if "multiple" in lowered or "per additional" in lowered or "yes" in lowered:
        return None
    match = re.search(r"\b(\d+)\b", value)
    return int(match.group(1)) if match else None


def _parse_area_unit_sq_cm(billing_time: str | None) -> int | None:
    if not billing_time:
        return None
    match = re.search(r"(\d+)\s*sq\s*cm", billing_time, re.IGNORECASE)
    return int(match.group(1)) if match else None


# Category F procedure families (source of truth for wound clarification).
AREA_PROCEDURE_FAMILIES: tuple[AreaProcedureFamily, ...] = (
    AreaProcedureFamily(
        family_id="selective_debridement",
        label="Selective debridement",
        primary_codes=("97597",),
        addon_codes=("97598",),
        keywords=(
            "selective debrid",
            "sharp debrid",
            "enzymatic debrid",
            "wound debridement",
            "debridement",
        ),
        size_rules={"unit_sq_cm": 20, "base_code": "97597", "addon_code": "97598"},
    ),
    AreaProcedureFamily(
        family_id="traditional_npwt",
        label="Traditional negative pressure wound therapy",
        primary_codes=("97605", "97606"),
        keywords=(
            "traditional npwt",
            "traditional negative pressure",
            "non-disposable",
            "non disposable",
            "wound vac",
            "npwt",
            "negative pressure wound",
            "vacuum assisted",
        ),
        size_rules={
            "threshold_sq_cm": 50,
            "small_code": "97605",
            "large_code": "97606",
        },
    ),
    AreaProcedureFamily(
        family_id="disposable_npwt",
        label="Disposable negative pressure wound therapy",
        primary_codes=("97607",),
        addon_codes=("97608",),
        keywords=(
            "disposable npwt",
            "disposable wound vac",
            "disposable negative pressure",
            "disposable",
        ),
        size_rules={
            "threshold_sq_cm": 50,
            "base_code": "97607",
            "addon_code": "97608",
            "addon_unit_sq_cm": 50,
        },
    ),
)


class BillingCategoryStore:
    """Maps CPT codes to billing category profiles and category metadata."""

    def __init__(self, json_dir: Path):
        self.json_dir = json_dir
        self._lock = threading.Lock()
        self._loaded = False
        self._profiles: dict[str, CodeBillingProfile] = {}
        self._categories: dict[str, CategoryMeta] = {}
        self._rule_definitions: dict[str, str] = {}
        self._summary: dict[str, int] = {}
        self._total_codes: int = 0

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            path = self.json_dir / CATEGORIES_FILENAME
            if not path.exists():
                self._loaded = True
                return
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            self._rule_definitions = data.get("billing_rule_definitions", {})
            self._summary = {
                str(key): int(value)
                for key, value in (data.get("summary") or {}).items()
            }
            self._total_codes = int(data.get("total_codes") or 0)
            for category in data.get("categories", []):
                self._ingest_category(category)
            self._loaded = True

    def _ingest_category(self, category: dict) -> None:
        category_id = str(category.get("category_id", ""))
        billing_rule = str(category.get("billing_rule", ""))
        description = str(category.get("description", ""))
        base_fields = {
            "billing_rule": billing_rule,
            "category_id": category_id,
            "description": description,
            "segment_size_minutes": int(category.get("segment_size_minutes", 15)),
            "unit_threshold_minutes": int(category.get("unit_threshold_minutes", 8)),
            "min_time_for_1_unit": str(category.get("min_time_for_1_unit", "")),
            "max_units_allowed": str(category.get("max_units_allowed", "")),
            "billing_time": str(category.get("billing_time", "")),
            "max_units_cap": _parse_max_units_cap(
                str(category.get("max_units_allowed", ""))
            ),
        }

        subcategories = category.get("subcategories")
        collected_codes: list[str] = []
        subcategory_metas: list[SubcategoryMeta] = []

        if subcategories:
            for sub in subcategories:
                codes = self._ingest_code_group(sub, base_fields)
                collected_codes.extend(codes)
                subcategory_metas.append(
                    self._build_subcategory_meta(sub, base_fields, codes)
                )
        else:
            codes = self._ingest_code_group(category, base_fields)
            collected_codes.extend(codes)

        self._categories[category_id] = CategoryMeta(
            category_id=category_id,
            billing_rule=billing_rule,
            description=description,
            codes=tuple(dict.fromkeys(collected_codes)),
            subcategories=tuple(subcategory_metas),
            segment_size_minutes=base_fields["segment_size_minutes"],
            unit_threshold_minutes=base_fields["unit_threshold_minutes"],
            min_time_for_1_unit=base_fields["min_time_for_1_unit"],
            max_units_allowed=base_fields["max_units_allowed"],
            billing_time=base_fields["billing_time"],
            max_units_cap=base_fields["max_units_cap"],
        )

    def _build_subcategory_meta(
        self, group: dict, inherited: dict, codes: list[str]
    ) -> SubcategoryMeta:
        billing_time = str(group.get("billing_time", inherited.get("billing_time", "")))
        max_units_allowed = str(
            group.get("max_units_allowed", inherited["max_units_allowed"])
        )
        block = group.get("block_minutes")
        return SubcategoryMeta(
            subcategory_id=str(group.get("subcategory_id", "")),
            billing_rule=str(group.get("billing_rule") or inherited["billing_rule"]),
            codes=tuple(codes),
            description=str(group.get("description") or inherited["description"]),
            billing_time=billing_time,
            min_time_for_1_unit=str(
                group.get("min_time_for_1_unit", inherited["min_time_for_1_unit"])
            ),
            max_units_allowed=max_units_allowed,
            block_minutes=int(block) if block is not None else None,
            time_band=group.get("time_band"),
            area_unit_sq_cm=_parse_area_unit_sq_cm(billing_time),
            max_units_cap=_parse_max_units_cap(max_units_allowed),
            session_type=str(group.get("session_type", "")),
            encounter_type=str(group.get("encounter_type", "")),
        )

    def _ingest_code_group(self, group: dict, inherited: dict) -> list[str]:
        billing_rule = str(group.get("billing_rule") or inherited["billing_rule"])
        billing_time = str(group.get("billing_time", inherited.get("billing_time", "")))
        profile_fields = {
            "billing_rule": billing_rule,
            "category_id": inherited["category_id"],
            "subcategory_id": str(group.get("subcategory_id", "")) or None,
            "description": str(group.get("description") or inherited["description"]),
            "segment_size_minutes": int(
                group.get("segment_size_minutes", inherited["segment_size_minutes"])
            ),
            "unit_threshold_minutes": int(
                group.get(
                    "unit_threshold_minutes", inherited["unit_threshold_minutes"]
                )
            ),
            "block_minutes": group.get("block_minutes"),
            "time_band": group.get("time_band"),
            "billing_time": billing_time,
            "area_unit_sq_cm": _parse_area_unit_sq_cm(billing_time),
            "min_time_for_1_unit": str(
                group.get("min_time_for_1_unit", inherited["min_time_for_1_unit"])
            ),
            "max_units_allowed": str(
                group.get("max_units_allowed", inherited["max_units_allowed"])
            ),
            "max_units_cap": _parse_max_units_cap(
                str(group.get("max_units_allowed", inherited["max_units_allowed"]))
            ),
            "session_type": str(group.get("session_type", "")),
            "encounter_type": str(group.get("encounter_type", "")),
        }
        if profile_fields["block_minutes"] is not None:
            profile_fields["block_minutes"] = int(profile_fields["block_minutes"])

        ingested: list[str] = []
        for code in group.get("codes", []):
            normalized = str(code).strip().upper()
            if not normalized:
                continue
            self._profiles[normalized] = CodeBillingProfile(
                cpt_code=normalized,
                **profile_fields,
            )
            ingested.append(normalized)
        return ingested

    def get_profile(self, cpt_code: str) -> CodeBillingProfile | None:
        self.ensure_loaded()
        return self._profiles.get(cpt_code.strip().upper())

    def get_category(self, category_id: str) -> CategoryMeta | None:
        self.ensure_loaded()
        return self._categories.get(category_id.upper())

    def get_category_for_code(self, cpt_code: str) -> CategoryMeta | None:
        profile = self.get_profile(cpt_code)
        if profile is None:
            return None
        return self.get_category(profile.category_id)

    def list_categories(self) -> list[CategoryMeta]:
        self.ensure_loaded()
        return list(self._categories.values())

    def list_codes_for_rule(self, billing_rule: str) -> list[str]:
        self.ensure_loaded()
        rule = billing_rule.strip().lower()
        return sorted(
            code
            for code, profile in self._profiles.items()
            if profile.billing_rule == rule
        )

    def summary_counts(self) -> dict[str, int]:
        self.ensure_loaded()
        return dict(self._summary)

    def summary_count_for_rule(self, billing_rule: str) -> int | None:
        self.ensure_loaded()
        rule = billing_rule.strip().lower()
        if rule in self._summary:
            return self._summary[rule]
        # Fallback to live profile count when summary key is missing.
        codes = self.list_codes_for_rule(rule)
        return len(codes) if codes else None

    def known_billing_rules(self) -> list[str]:
        self.ensure_loaded()
        if self._summary:
            return list(self._summary.keys())
        return sorted({p.billing_rule for p in self._profiles.values()})

    def total_codes(self) -> int:
        self.ensure_loaded()
        return self._total_codes or len(self._profiles)

    def rule_definition(self, billing_rule: str) -> str:
        self.ensure_loaded()
        return self._rule_definitions.get(billing_rule, billing_rule)

    def area_procedure_families(self) -> tuple[AreaProcedureFamily, ...]:
        return AREA_PROCEDURE_FAMILIES

    def to_structured_profile(self, cpt_code: str) -> dict[str, Any] | None:
        profile = self.get_profile(cpt_code)
        if profile is None:
            return None
        return {
            "cpt_code": profile.cpt_code,
            "category_id": profile.category_id,
            "subcategory_id": profile.subcategory_id,
            "billing_rule": profile.billing_rule,
            "billing_rule_definition": self.rule_definition(profile.billing_rule),
            "billing_time": profile.billing_time,
            "min_time_for_1_unit": profile.min_time_for_1_unit,
            "max_units_allowed": profile.max_units_allowed,
            "max_units_cap": profile.max_units_cap,
            "block_minutes": profile.block_minutes,
            "time_band": profile.time_band,
            "area_unit_sq_cm": profile.area_unit_sq_cm,
            "description": profile.description,
            "segment_size_minutes": profile.segment_size_minutes,
            "unit_threshold_minutes": profile.unit_threshold_minutes,
        }


_default_store: BillingCategoryStore | None = None


def get_billing_category_store(json_dir: Path | None = None) -> BillingCategoryStore:
    global _default_store
    if json_dir is not None:
        return BillingCategoryStore(json_dir)
    if _default_store is None:
        from app.config import settings

        _default_store = BillingCategoryStore(settings.json_dir)
    return _default_store
