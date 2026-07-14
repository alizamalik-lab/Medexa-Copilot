"""Deterministic billing orchestrator: decompose multi-questions and call tools.

Priority:
  Billing Engine -> Category Engine -> MUE -> NCCI -> Modifier/ICD -> RAG -> LLM

The LLM never replaces these tools; it may only present structured results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag.billing_categories import get_billing_category_store
from rag.billing_engine import (
    AREA_BASED,
    EIGHT_MINUTE_RULE,
    FULL_BLOCK_REQUIRED,
    TIME_BAND_SELECT,
    UNTIMED_RULES,
    _extract_billing_codes,
    _format_comparison_answer,
    calculate_ama_units,
    calculate_cms_pooled_units,
    format_cms_ama_conceptual_comparison,
    is_cms_ama_comparison_question,
    is_rule_comparison_question,
    is_unit_calculation_guide_question,
    is_unit_calculation_question,
    parse_cpt_time_entries,
    try_unit_calculation_guide,
    try_unit_calculation_payload,
)
from rag.billing_tools import BillingTools
from rag.category_engine import try_category_engine
from rag.category_tools.area_based import is_area_wound_question
from rag.intent_detector import is_multi_topic_question
from rag.memory import ChatMessage
from rag.response_completeness import (
    append_completeness_gap_notice,
    check_response_completeness,
)
from rag.response_style import wants_calculation_steps, wants_concise, wants_expanded_answer


@dataclass
class OrchestratorResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    clarification: bool = False
    structured: dict[str, Any] = field(default_factory=dict)


_CPT_SUMMARY_HINTS = re.compile(
    r"^\s*(?:what\s+is|explain|describe|tell\s+me\s+about)\s+"
    r"(?:cpt\s*|hcpcs\s*)?#?\s*(\d{5}|[A-VJ-Z]\d{4})\s*\??\s*$",
    re.IGNORECASE,
)

_BILLED_UNITS = re.compile(
    r"\b(?:billed|bill(?:ing)?|claim(?:ed)?)\s+(\d{1,2})\s+units?\b",
    re.IGNORECASE,
)
_UNITS_CLAIMED = re.compile(
    r"\b(\d{1,2})\s+units?\b.{0,40}\b(?:billed|allowed|mue|limit)\b"
    r"|\bmue\b.{0,40}\b(\d{1,2})\s+units?\b",
    re.IGNORECASE,
)

_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("summary", re.compile(r"\b(explain|what is|describe|tell me about)\b", re.I)),
    ("timed", re.compile(r"\b(timed|untimed|time[\s-]based)\b", re.I)),
    ("mue", re.compile(r"\b(mue|medically unlikely|unit(?:s)?\s+limit|max(?:imum)?\s+units)\b", re.I)),
    (
        "ncci",
        re.compile(
            r"\b(ncci|ptp|bill(?:ed)?\s+with|billed together|together|same day|"
            r"can (?:it|they|these|codes) be billed)\b",
            re.I,
        ),
    ),
    (
        "modifier",
        re.compile(r"\b(modifier\s*59|modifier|mod\s*59)\b", re.I),
    ),
    ("addon", re.compile(r"\b(add[\s-]?on|addon|aoc)\b", re.I)),
    ("icd", re.compile(r"\b(icd[\s-]?10|diagnosis code)\b", re.I)),
    (
        "medicare",
        re.compile(
            r"\b(medicare\s+units?|cms\s+units?|medicare|cms 8[\s-]?minute)\b",
            re.I,
        ),
    ),
    (
        "ama",
        re.compile(r"\b(ama\s+units?|ama|rule of eight)\b", re.I),
    ),
    (
        "units",
        re.compile(
            r"\b(billing unit|billable unit|how many units|calculate|"
            r"brief calculation|unit calculation|minutes?)\b",
            re.I,
        ),
    ),
    ("category", re.compile(r"\b(categor(?:y|ies)|billing rule|which rule)\b", re.I)),
    ("compare", re.compile(r"\b(compare|difference|versus| vs\.? )\b", re.I)),
)

_BILLING_RULE_ALIASES: dict[str, str] = {
    "8 minute rule": "8_minute_rule",
    "8-minute rule": "8_minute_rule",
    "eight minute rule": "8_minute_rule",
    "full block required": "full_block_required",
    "full block": "full_block_required",
    "untimed per session": "untimed_per_session",
    "untimed per encounter": "untimed_per_encounter",
    "untimed per procedure": "untimed_per_procedure",
    "untimed per day": "untimed_per_day",
    "untimed per episode": "untimed_per_episode",
    "area based": "area_based",
    "area-based": "area_based",
    "time band select": "time_band_select",
    "time-band select": "time_band_select",
    "time band": "time_band_select",
}

_COUNT_RULE_HINTS = re.compile(
    r"\b("
    r"how many codes|how many cpt|number of codes|code count|"
    r"fall under|belong under|belong to|codes under|"
    r"codes (?:in|for) (?:the )?(?:category|rule)|"
    r"under the .+ categor"
    r")\b",
    re.IGNORECASE,
)
_LIST_RULE_HINTS = re.compile(
    r"\b("
    r"name(?:s)?(?:\s+them)?|list(?:\s+them)?|list (?:the )?codes|"
    r"which codes|what codes|what are the codes|"
    r"show (?:the )?codes|cpt(?:/hcpcs)? (?:code )?list|code list|"
    r"their names|code names"
    r")\b"
    r"|^\s*(?:name|list|names)\s*(?:them|those|these|it)?\s*[.?!]?\s*$",
    re.IGNORECASE,
)
_RULE_FOCUS_IN_ANSWER = re.compile(r"\*\*`([a-z0-9_]+)`\*\*")
_SHORT_LIST_FOLLOWUP = re.compile(
    r"^\s*(?:name|list|names|show)(?:\s+(?:them|those|these|it|the codes?))?\s*[.?!]?\s*$",
    re.IGNORECASE,
)
_SUMMARY_ALL_HINTS = re.compile(
    r"\b("
    r"all billing rules|summary of (?:billing )?rules|"
    r"how many codes (?:per|for each|by) (?:rule|category)|"
    r"how many categor(?:y|ies)|"
    r"how many billing (?:rules|categories)|"
    r"what (?:are )?(?:the )?categor(?:y|ies)|"
    r"list (?:all )?(?:the )?(?:billing )?(?:rules|categories)|"
    r"show (?:all )?(?:the )?(?:billing )?(?:rules|categories)|"
    r"all categor(?:y|ies)|"
    r"billing categor(?:y|ies) (?:are there|exist)|"
    r"categor(?:y|ies) (?:are there|exist|do we have)"
    r")\b",
    re.IGNORECASE,
)
_SHORT_CATEGORY_OVERVIEW = re.compile(
    r"^\s*(?:how many|what|list|show|name)\s+"
    r"(?:are\s+(?:the\s+)?)?(?:the\s+)?"
    r"(?:billing\s+)?categor(?:y|ies)\b"
    r".*$",
    re.IGNORECASE,
)


def try_billing_orchestrator(
    question: str,
    billing_tools: BillingTools | None,
    *,
    history: list[ChatMessage] | None = None,
    focus_billing_rule: str | None = None,
) -> OrchestratorResult | None:
    """
    Handle CPT summaries, multi-question prompts, MUE validation, CMS/AMA,
    and category explanations with deterministic tools only.
    """
    # Billing-rule inventory (counts / names) from category JSON summary.
    inventory = try_billing_rule_inventory(
        question,
        history=history,
        focus_billing_rule=focus_billing_rule,
    )
    if inventory is not None:
        return inventory

    codes = _extract_billing_codes(question)
    multi = is_multi_topic_question(question) or _has_multiple_sentences(question)
    topics = detect_requested_topics(question)

    # CPT complete summary ("What is CPT 97110?" / "What is 97110?")
    summary_match = _CPT_SUMMARY_HINTS.match(question.strip())
    if summary_match and billing_tools is not None and not multi:
        code = summary_match.group(1).upper()
        return OrchestratorResult(
            answer=build_cpt_summary(
                code, billing_tools, concise=wants_concise(question)
            ),
            sources=[
                "billing_tool:lookup_cpt",
                "billing_tool:lookup_mue",
                "billing_tool:lookup_aoc",
                "billing_category_engine",
            ],
            structured={"type": "cpt_summary", "cpt_code": code},
        )

    # CMS vs AMA — always compute BOTH sides when minutes exist
    if is_cms_ama_comparison_question(question) or (
        is_rule_comparison_question(question)
        and ("medicare" in question.lower() or "cms" in question.lower())
    ):
        compare = build_cms_ama_comparison(question)
        if compare:
            return OrchestratorResult(
                answer=compare,
                sources=["billing_engine"],
                structured={"type": "cms_ama_comparison"},
            )

    # "I billed 8 units" -> MUE tool
    if billing_tools is not None:
        mue_check = try_mue_units_claimed(question, codes, billing_tools)
        if mue_check is not None:
            return mue_check

    # Multi-question / multi-topic: answer every requested item
    if multi and codes and billing_tools is not None:
        # Pure unit-calc turns belong to the billing engine (e.g. untimed daily caps).
        cross_topics = topics.intersection(
            {"mue", "ncci", "modifier", "summary", "timed", "addon", "icd", "category"}
        )
        if is_unit_calculation_question(question) and not cross_topics and len(codes) <= 2:
            return None
        return build_multi_topic_answer(question, codes, topics, billing_tools)

    # Single CPT + multiple topics
    if (
        billing_tools is not None
        and codes
        and topics.intersection({"summary", "timed", "mue", "addon", "category", "ncci", "modifier"})
        and len(topics) >= 2
        and not is_unit_calculation_question(question)
        and not is_area_wound_question(question)
    ):
        return build_multi_topic_answer(question, codes, topics, billing_tools)

    # Concise single-topic answers
    if billing_tools is not None and codes and not is_unit_calculation_question(question):
        single = _concise_single_topic(question, codes, topics, billing_tools)
        if single is not None:
            return single

    # Category explanation for special categories (non-unit questions)
    if codes and (
        "category" in topics
        or _wants_category_explanation(question, codes)
    ):
        explanation = explain_special_category(
            codes[0], concise=wants_concise(question)
        )
        if explanation and not is_unit_calculation_question(question):
            if topics <= {"category", "summary", "timed"} or "category" in topics:
                if billing_tools is not None and "summary" in topics:
                    return build_multi_topic_answer(
                        question, codes, topics | {"category"}, billing_tools
                    )
                return OrchestratorResult(
                    answer=explanation,
                    sources=["billing_category_engine"],
                    structured={"type": "category_explanation", "cpt_code": codes[0]},
                )

    return None


def _concise_single_topic(
    question: str,
    codes: list[str],
    topics: set[str],
    tools: BillingTools,
) -> OrchestratorResult | None:
    if not wants_concise(question):
        return None
    primary = codes[0]
    store = get_billing_category_store()
    profile = store.get_profile(primary)

    if topics == {"timed"} or (
        "timed" in topics and topics <= {"timed", "summary"} and "what is" not in question.lower()
        and not _CPT_SUMMARY_HINTS.match(question.strip())
    ):
        if profile and profile.billing_rule == EIGHT_MINUTE_RULE:
            answer = (
                f"Yes. CPT **{primary}** is a timed code billed under the "
                "CMS 8-Minute Rule."
            )
        elif profile and profile.billing_rule == FULL_BLOCK_REQUIRED:
            answer = (
                f"Yes. CPT **{primary}** is timed using "
                f"**{profile.block_minutes}-minute blocks** "
                "(not the CMS 8-Minute Rule)."
            )
        elif profile and profile.billing_rule in UNTIMED_RULES:
            answer = f"No. CPT **{primary}** is untimed."
        elif profile and profile.billing_rule == AREA_BASED:
            answer = f"No. CPT **{primary}** is area-based, not time-based."
        else:
            return None
        return OrchestratorResult(
            answer=answer,
            sources=["billing_category_engine"],
            structured={"type": "timed", "cpt_code": primary},
        )

    if topics == {"mue"} or (
        "mue" in topics and topics <= {"mue"} 
    ):
        mue = tools.lookup_mue(primary)
        if mue.get("found"):
            return OrchestratorResult(
                answer=f"MUE: **{mue.get('limit')}** units.",
                sources=["billing_tool:lookup_mue"],
                structured={"type": "mue", "cpt_code": primary},
            )

    if "ncci" in topics or "modifier" in topics:
        partner = codes[1] if len(codes) >= 2 else None
        if partner is None:
            with_match = re.search(
                r"(?:with|and)\s*(?:cpt\s*)?(\d{5}|[A-VJ-Z]\d{4})",
                question,
                re.I,
            )
            if with_match:
                partner = with_match.group(1).upper()
        if partner:
            ncci = tools.check_ncci(primary, partner)
            lines = []
            if "ncci" in topics or re.search(r"bill", question, re.I):
                if ncci.get("allowed"):
                    lines.append(
                        f"Yes. Based on the current NCCI rules, CPT **{primary}** "
                        f"and **{partner}** can be billed together."
                    )
                else:
                    lines.append(
                        f"No. Based on the current NCCI rules, CPT **{primary}** "
                        f"and **{partner}** cannot be billed together."
                    )
            if "modifier" in topics:
                if ncci.get("modifier59_required"):
                    lines.append(
                        "Modifier 59 may be required when the services are distinct."
                    )
                elif not ncci.get("allowed"):
                    lines.append(
                        "Modifier 59 does not bypass this NCCI edit."
                    )
                else:
                    lines.append(
                        "No. Modifier 59 is not required based on the current NCCI data."
                    )
            if lines:
                return OrchestratorResult(
                    answer=" ".join(lines) if len(lines) == 1 else "\n".join(lines),
                    sources=["billing_tool:check_ncci"],
                    structured={"type": "ncci", "cpt_codes": [primary, partner]},
                )

    return None


def normalize_billing_rule_slug(raw: str) -> str | None:
    """Map user text / aliases to a canonical billing_rule key."""
    store = get_billing_category_store()
    known = store.known_billing_rules()
    text = raw.strip().strip("\"'`").lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    if text in known:
        return text
    spaced = raw.strip().strip("\"'`").lower().replace("_", " ").replace("-", " ")
    spaced = re.sub(r"\s+", " ", spaced).strip()
    if spaced in _BILLING_RULE_ALIASES:
        return _BILLING_RULE_ALIASES[spaced]
    # Fuzzy: rule key contained in question fragment
    for rule in known:
        if rule.replace("_", " ") == spaced or rule == text:
            return rule
    return None


def extract_billing_rules_from_question(question: str) -> list[str]:
    """Find one or more billing_rule keys mentioned in the question."""
    store = get_billing_category_store()
    known = store.known_billing_rules()
    found: list[str] = []

    # Quoted rule names first: "area_based"
    for match in re.finditer(r"[\"'`]([A-Za-z0-9 _\-]+)[\"'`]", question):
        slug = normalize_billing_rule_slug(match.group(1))
        if slug and slug not in found:
            found.append(slug)

    lowered = question.lower()
    # Longest keys first so untimed_per_episode beats untimed
    for rule in sorted(known, key=len, reverse=True):
        patterns = (
            rule,
            rule.replace("_", " "),
            rule.replace("_", "-"),
        )
        if any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in patterns):
            if rule not in found:
                found.append(rule)

    for alias, rule in _BILLING_RULE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered) and rule not in found:
            found.append(rule)

    return found


def infer_focus_billing_rule(
    history: list[ChatMessage] | None,
    focus_billing_rule: str | None = None,
) -> str | None:
    """Recover the active billing_rule from session focus or recent turns."""
    if focus_billing_rule:
        return focus_billing_rule.strip().lower()
    if not history:
        return None
    for msg in reversed(history[-8:]):
        if msg.role == "assistant":
            match = _RULE_FOCUS_IN_ANSWER.search(msg.content)
            if match:
                return match.group(1)
            # Also accept plain "Total codes:" replies headed by the rule name.
            heading = re.search(
                r"(?:^|\n)\*\*`?([a-z0-9_]+)`?\*\*",
                msg.content,
            )
            if heading:
                slug = normalize_billing_rule_slug(heading.group(1))
                if slug:
                    return slug
        if msg.role == "user":
            rules = extract_billing_rules_from_question(msg.content)
            if rules:
                return rules[0]
    return None


def try_billing_rule_inventory(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    focus_billing_rule: str | None = None,
) -> OrchestratorResult | None:
    """
    Answer count/list questions for billing_rule groups using the category JSON
    summary (and code lists when the user asks for names).
    """
    store = get_billing_category_store()
    store.ensure_loaded()
    lowered = question.lower().strip()

    wants_summary_all = bool(_SUMMARY_ALL_HINTS.search(question)) or bool(
        _SHORT_CATEGORY_OVERVIEW.match(question.strip())
    )
    wants_count = bool(_COUNT_RULE_HINTS.search(question)) or bool(
        re.search(r"\bhow many\b.{0,40}\b(code|codes|cpt)\b", lowered)
    )
    wants_names = bool(_LIST_RULE_HINTS.search(question)) or bool(
        _SHORT_LIST_FOLLOWUP.match(question.strip())
    )
    rules = extract_billing_rules_from_question(question)

    # Short follow-ups: "name them" / "list them" → prior billing rule.
    if not rules and wants_names:
        prior_rule = infer_focus_billing_rule(history, focus_billing_rule)
        if prior_rule:
            rules = [prior_rule]

    if wants_summary_all and not rules:
        counts = store.summary_counts()
        category_count = len(counts)
        lines = [
            f"There are **{category_count}** billing categories "
            f"({store.total_codes()} total CPT/HCPCS codes).",
            "",
            "CPT/HCPCS codes per category:",
        ]
        for rule, count in counts.items():
            lines.append(f"- `{rule}`: **{count}**")
        return OrchestratorResult(
            answer="\n".join(lines),
            sources=["billing_category_engine"],
            structured={
                "type": "billing_rule_summary",
                "category_count": category_count,
                "counts": counts,
            },
        )

    if not rules:
        return None
    if not (wants_count or wants_names):
        return None

    sections: list[str] = []
    structured_rules: list[dict] = []
    include_names = wants_names
    # Follow-up "name them" after a count answer should list codes (still show count).
    include_count = wants_count or not include_names

    for rule in rules:
        count = store.summary_count_for_rule(rule)
        codes = store.list_codes_for_rule(rule)
        live_count = len(codes)
        final_count = count if count is not None else live_count
        definition = store.rule_definition(rule)
        parts = [f"**`{rule}`**"]
        if include_count:
            parts.append(f"- Total codes: **{final_count}**")
        if definition and definition != rule:
            parts.append(f"- Definition: {definition}")
        if include_names:
            if codes:
                parts.append(
                    "- CPT/HCPCS codes: " + ", ".join(f"**{c}**" for c in codes)
                )
            else:
                parts.append("- CPT/HCPCS codes: None listed.")
        sections.append("\n".join(parts))
        structured_rules.append(
            {
                "billing_rule": rule,
                "count": final_count,
                "codes": codes if include_names else [],
            }
        )

    if not sections:
        return None

    return OrchestratorResult(
        answer="\n\n".join(sections),
        sources=["billing_category_engine"],
        structured={
            "type": "billing_rule_inventory",
            "rules": structured_rules,
            "include_names": include_names,
            "focus_billing_rule": rules[0] if rules else None,
        },
    )


def detect_requested_topics(question: str) -> set[str]:
    topics: set[str] = set()
    for name, pattern in _TOPIC_PATTERNS:
        if pattern.search(question):
            topics.add(name)
    if _CPT_SUMMARY_HINTS.match(question.strip()):
        topics.add("summary")
    if is_cms_ama_comparison_question(question):
        topics.add("compare")
    # Explicit Medicare + AMA unit asks → answer both (and comparison when useful)
    if "medicare" in topics and "ama" in topics:
        topics.add("compare")
        topics.add("units")
    if _BILLED_UNITS.search(question) or _UNITS_CLAIMED.search(question):
        topics.add("mue")
    if re.search(r"\bbrief\s+calculation\b", question, re.I):
        topics.add("units")
    return topics


def _has_multiple_sentences(question: str) -> bool:
    parts = [p.strip() for p in re.split(r"[\n?]+", question) if p.strip()]
    if len(parts) >= 2:
        return True
    # Lines like "Explain CPT 97110." + "Is it timed?"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", question) if s.strip()]
    return len(sentences) >= 2


def build_cpt_summary(
    cpt_code: str,
    tools: BillingTools,
    *,
    concise: bool = True,
) -> str:
    code = cpt_code.strip().upper()
    cpt = tools.lookup_cpt(code)
    mue = tools.lookup_mue(code)
    aoc = tools.lookup_aoc(code)
    store = get_billing_category_store()
    profile = store.get_profile(code)

    desc = ""
    if cpt.get("found"):
        desc = cpt.get("description") or cpt.get("short_name") or ""

    timed_label = "Unknown"
    if profile:
        if profile.billing_rule == EIGHT_MINUTE_RULE:
            timed_label = "Timed (CMS 8-Minute Rule)"
        elif profile.billing_rule == FULL_BLOCK_REQUIRED:
            timed_label = (
                f"Timed (full {profile.block_minutes or '?'}-minute block)"
            )
        elif profile.billing_rule in UNTIMED_RULES:
            timed_label = f"Untimed ({profile.billing_rule.replace('_', ' ')})"
        elif profile.billing_rule == AREA_BASED:
            timed_label = "Area-based"
        elif profile.billing_rule == TIME_BAND_SELECT:
            timed_label = f"Time-band ({profile.time_band or 'see category'})"
        else:
            timed_label = profile.billing_rule.replace("_", " ")
    elif cpt.get("found") and cpt.get("timed") is not None:
        timed_label = "Timed" if cpt.get("timed") else "Untimed"

    mue_text = (
        f"{mue['limit']} units"
        if mue.get("found") and mue.get("limit") is not None
        else "Not available"
    )

    if aoc.get("found") and aoc.get("is_addon_code"):
        addon_text = f"Add-on (parent {aoc.get('parent_code') or 'n/a'})"
    elif aoc.get("found") and aoc.get("addon_codes_allowed"):
        addon_text = ", ".join(str(c) for c in aoc["addon_codes_allowed"])
    else:
        addon_text = "None listed"

    category_text = (
        f"Category {profile.category_id}"
        + (f"/{profile.subcategory_id}" if profile and profile.subcategory_id else "")
        if profile
        else "Not categorized"
    )
    rule_text = (
        profile.billing_rule
        if profile
        else (cpt.get("billing_rule") or "unknown")
    )

    if concise:
        lines = [
            f"**CPT {code}**",
            f"- Description: {desc or 'Not available.'}",
            f"- Timed: {timed_label}.",
            f"- Category: {category_text}.",
            f"- MUE: {mue_text}.",
            f"- Add-on codes: {addon_text}.",
            f"- Billing rule: `{rule_text}`.",
        ]
        if profile and profile.billing_rule != EIGHT_MINUTE_RULE:
            short = explain_special_category(code, concise=True)
            if short:
                lines.append(f"- Note: {short}")
        return "\n".join(lines)

    # Expanded summary
    lines = [
        f"### CPT {code} Summary",
        "",
        "**Description**",
        f"• {desc}" if desc else "• Description is not available.",
        "",
        "**Timed / Untimed**",
        f"• {timed_label}",
        "",
        "**Billing Category**",
        f"• {category_text}",
    ]
    if profile:
        lines.append(
            f"• Max units allowed: **{profile.max_units_allowed or 'See category'}**"
        )
    lines.extend(["", "**MUE**", f"• MUE limit: **{mue_text}**", "", "**Add-on Codes**"])
    lines.append(f"• {addon_text}")
    lines.extend(
        [
            "",
            "**Common Billing Rule**",
            f"• `{rule_text}`",
        ]
    )
    if profile:
        lines.append(f"• {store.rule_definition(profile.billing_rule)}")
        if profile.billing_rule != EIGHT_MINUTE_RULE:
            extra = explain_special_category(code, concise=False)
            if extra:
                lines.extend(["", extra])
    return "\n".join(lines)


def explain_special_category(
    cpt_code: str, *, concise: bool = True
) -> str | None:
    store = get_billing_category_store()
    profile = store.get_profile(cpt_code)
    if profile is None:
        return None

    rule = profile.billing_rule
    cat = profile.category_id
    if rule == EIGHT_MINUTE_RULE:
        return (
            f"Category {cat}; CMS 8-Minute Rule."
            if concise
            else (
                f"CPT **{cpt_code}** belongs to Category **{cat}**.\n\n"
                "Units are calculated under the **CMS 8-Minute Rule**."
            )
        )
    if rule == FULL_BLOCK_REQUIRED:
        block = profile.block_minutes or "?"
        if concise:
            return (
                f"Category {cat}; billed in {block}-minute blocks "
                f"(not CMS 8-Minute Rule). Max: {profile.max_units_allowed}."
            )
        return (
            f"CPT **{cpt_code}** belongs to Category **{cat}**"
            + (f" ({profile.subcategory_id})" if profile.subcategory_id else "")
            + ".\n\n"
            f"Units are calculated in **{block}-minute blocks** rather than using "
            "the CMS 8-Minute Rule. Partial blocks do not receive credit.\n\n"
            f"**Max units allowed:** {profile.max_units_allowed}"
        )
    if rule in UNTIMED_RULES:
        friendly = rule.replace("untimed_", "").replace("_", " ")
        if concise:
            return (
                f"Category {cat}; untimed ({friendly}). "
                f"Max: {profile.max_units_allowed}."
            )
        return (
            f"CPT **{cpt_code}** belongs to Category **{cat}**.\n\n"
            f"This is an **untimed** code billed **{friendly}**. "
            f"**Max units allowed:** {profile.max_units_allowed}"
        )
    if rule == AREA_BASED:
        if concise:
            return (
                f"Category {cat}; area-based. "
                f"Max: {profile.max_units_allowed}."
            )
        return (
            f"CPT **{cpt_code}** belongs to Category **{cat}** (area-based).\n\n"
            f"**Requirement:** {profile.min_time_for_1_unit or profile.billing_time}\n"
            f"**Max units allowed:** {profile.max_units_allowed}"
        )
    if rule == TIME_BAND_SELECT:
        band = profile.time_band or profile.min_time_for_1_unit
        if concise:
            return f"Category {cat}; pick one code by time band ({band})."
        return (
            f"CPT **{cpt_code}** belongs to Category **{cat}**.\n\n"
            f"Select **one** code based on the total time band ({band})."
        )
    return f"Category {cat} (`{rule}`)."


def _wants_category_explanation(question: str, codes: list[str]) -> bool:
    if not codes:
        return False
    store = get_billing_category_store()
    profile = store.get_profile(codes[0])
    if profile is None:
        return False
    if profile.billing_rule == EIGHT_MINUTE_RULE:
        return False
    lowered = question.lower()
    return any(
        hint in lowered
        for hint in (
            "what is",
            "explain",
            "how is it billed",
            "billing rule",
            "how are units",
            "calculate",
            "how to calculate",
            "category",
        )
    )


def try_mue_units_claimed(
    question: str,
    codes: list[str],
    tools: BillingTools,
) -> OrchestratorResult | None:
    match = _BILLED_UNITS.search(question)
    claimed = int(match.group(1)) if match else None
    if claimed is None:
        alt = _UNITS_CLAIMED.search(question)
        if alt:
            claimed = int(alt.group(1) or alt.group(2))
    concise = wants_concise(question)

    if claimed is None or not codes:
        if codes and re.search(r"\bmue\b", question, re.I) and not is_multi_topic_question(
            question
        ):
            mue = tools.lookup_mue(codes[0])
            if mue.get("found"):
                if concise:
                    answer = f"MUE: **{mue.get('limit')}** units."
                else:
                    answer = (
                        f"### MUE for CPT {codes[0]}\n\n"
                        f"• MUE limit: **{mue.get('limit')}**\n"
                    )
                    if mue.get("description"):
                        answer += f"• {mue['description']}\n"
                return OrchestratorResult(
                    answer=answer.strip(),
                    sources=["billing_tool:lookup_mue"],
                    structured={"type": "mue", "cpt_code": codes[0], **mue},
                )
        return None

    code = codes[0]
    mue = tools.lookup_mue(code)
    if not mue.get("found"):
        return OrchestratorResult(
            answer=f"MUE for CPT **{code}** is not available.",
            sources=["billing_tool:lookup_mue"],
        )

    limit = mue.get("limit")
    try:
        limit_int = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        limit_int = None

    if concise:
        if limit_int is None:
            answer = f"MUE for CPT **{code}**: **{limit}**."
        elif claimed <= limit_int:
            answer = (
                f"**{claimed}** units are within the MUE limit of **{limit_int}** "
                f"for CPT **{code}**."
            )
        else:
            answer = (
                f"**{claimed}** units exceed the MUE limit of **{limit_int}** "
                f"for CPT **{code}**."
            )
        return OrchestratorResult(
            answer=answer,
            sources=["billing_tool:lookup_mue"],
            structured={
                "type": "mue_units_check",
                "cpt_code": code,
                "claimed_units": claimed,
                "mue_limit": limit,
            },
        )

    lines = [
        f"### MUE Check — CPT {code}",
        "",
        f"• Units billed/claimed: **{claimed}**",
        f"• MUE limit: **{limit}**",
    ]
    if mue.get("description"):
        lines.append(f"• {mue['description']}")
    if limit_int is not None:
        if claimed <= limit_int:
            lines.append(
                f"\n**Result:** **{claimed}** units are within the MUE limit of **{limit_int}**."
            )
        else:
            lines.append(
                f"\n**Result:** **{claimed}** units exceed the MUE limit of **{limit_int}**."
            )
    return OrchestratorResult(
        answer="\n".join(lines),
        sources=["billing_tool:lookup_mue", "billing_category_engine"],
        structured={
            "type": "mue_units_check",
            "cpt_code": code,
            "claimed_units": claimed,
            "mue_limit": limit,
        },
    )


def build_cms_ama_comparison(question: str) -> str | None:
    entries = parse_cpt_time_entries(question)
    concise = wants_concise(question) and not wants_calculation_steps(question)

    if not entries:
        if concise:
            return (
                "| Rule | Method |\n"
                "|------|--------|\n"
                "| Medicare (CMS) | Pool timed minutes, then apply 8-minute table |\n"
                "| AMA | Calculate each CPT separately |\n\n"
                "Provide CPT minutes for a numeric comparison."
            )
        return format_cms_ama_conceptual_comparison()

    cms = calculate_cms_pooled_units(entries)
    ama = calculate_ama_units(entries)
    cms_total = sum(cms.values())
    ama_total = sum(ama.values())

    if concise:
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

    detail = _format_comparison_answer(entries)
    why = (
        "\n\n### Why they differ\n\n"
        "• **Medicare/CMS** pools total timed therapy minutes across eligible codes.\n"
        "• **AMA** calculates each CPT independently.\n"
    )
    table = (
        "\n\n| Rule | Units |\n|------|------:|\n"
        f"| Medicare (CMS) | {cms_total} |\n"
        f"| AMA | {ama_total} |\n"
    )
    return detail + table + why


def build_multi_topic_answer(
    question: str,
    codes: list[str],
    topics: set[str],
    tools: BillingTools,
) -> OrchestratorResult:
    """Answer every requested item for EVERY CPT/HCPCS code in the question."""
    concise = wants_concise(question)
    sources: list[str] = []
    answered: list[str] = []
    sections: list[str] = []
    store = get_billing_category_store()

    # Always process every extracted code independently.
    for code in codes:
        section, section_answered, section_sources = _build_code_section(
            code=code,
            question=question,
            topics=topics,
            tools=tools,
            store=store,
            concise=concise,
        )
        sections.append(section)
        answered.extend(section_answered)
        sources.extend(section_sources)

    # Shared cross-code topics (NCCI / Modifier / Medicare-AMA / calculation)
    shared, shared_answered, shared_sources = _build_shared_sections(
        question=question,
        codes=codes,
        topics=topics,
        tools=tools,
        concise=concise,
    )
    if shared:
        sections.extend(shared)
    answered.extend(shared_answered)
    sources.extend(shared_sources)

    answer = "\n\n".join(s for s in sections if s.strip())
    answered_unique = list(dict.fromkeys(answered))

    # Completeness gate — fill any remaining CPT/topic gaps before return.
    report = check_response_completeness(
        answer=answer,
        expected_codes=codes,
        expected_topics=topics,
        answered_topics=answered_unique,
    )
    if not report.ok:
        filler, filled_topics, fill_sources = _fill_completeness_gaps(
            question=question,
            codes=codes,
            topics=topics,
            missing_codes=report.missing_codes,
            missing_topics=report.missing_topics,
            tools=tools,
            store=store,
        )
        if filler:
            answer = f"{answer}\n\n{filler}".strip()
            answered_unique.extend(filled_topics)
            sources.extend(fill_sources)
            report = check_response_completeness(
                answer=answer,
                expected_codes=codes,
                expected_topics=topics,
                answered_topics=answered_unique,
            )
        if not report.ok:
            answer = append_completeness_gap_notice(answer, report)
            print(f"[completeness] gaps remain: {report.summary}")
        else:
            print("[completeness] every CPT processed; every topic answered")
    else:
        print("[completeness] every CPT processed; every topic answered")

    return OrchestratorResult(
        answer=answer,
        sources=list(dict.fromkeys(sources)),
        structured={
            "type": "multi_topic",
            "cpt_codes": codes,
            "topics": sorted(topics),
            "answered": list(dict.fromkeys(answered_unique)),
            "concise": concise,
            "completeness": {
                "ok": report.ok,
                "missing_codes": report.missing_codes,
                "missing_topics": report.missing_topics,
            },
        },
    )


def _timed_label_for_profile(profile) -> tuple[bool | None, str]:
    if profile is None:
        return None, "Unknown"
    if profile.billing_rule == EIGHT_MINUTE_RULE:
        return True, "Yes. Timed under the CMS 8-Minute Rule."
    if profile.billing_rule == FULL_BLOCK_REQUIRED:
        return True, (
            f"Yes. Timed in {profile.block_minutes}-minute blocks "
            "(not CMS 8-Minute Rule)."
        )
    if profile.billing_rule in UNTIMED_RULES:
        return False, "No. Untimed."
    if profile.billing_rule == AREA_BASED:
        return False, "No. Area-based."
    if profile.billing_rule == TIME_BAND_SELECT:
        band = profile.time_band or "see time band"
        return None, f"Time-band selection ({band}) — one CPT, not additive units."
    return None, f"`{profile.billing_rule}`"


def _build_code_section(
    *,
    code: str,
    question: str,
    topics: set[str],
    tools: BillingTools,
    store,
    concise: bool,
) -> tuple[str, list[str], list[str]]:
    answered: list[str] = []
    sources: list[str] = []
    bullets: list[str] = []
    profile = store.get_profile(code)
    cpt = tools.lookup_cpt(code)
    desc = ""
    if cpt.get("found"):
        desc = cpt.get("description") or cpt.get("short_name") or ""

    include_summaryish = "summary" in topics or not topics or len(topics) >= 2
    if include_summaryish or "summary" in topics:
        bullets.append(f"- Description: {desc or 'Not available.'}")
        answered.append("summary")
        sources.extend(
            [
                "billing_tool:lookup_cpt",
                "billing_category_engine",
            ]
        )

    if "timed" in topics or "summary" in topics:
        _yes, timed_text = _timed_label_for_profile(profile)
        bullets.append(f"- Timed: {timed_text}")
        answered.append("timed")
        sources.append("billing_category_engine")

    if "mue" in topics or "summary" in topics:
        mue = tools.lookup_mue(code)
        if mue.get("found") and mue.get("limit") is not None:
            bullets.append(f"- MUE: **{mue['limit']}** units.")
        else:
            bullets.append("- MUE: Not available.")
        answered.append("mue")
        sources.append("billing_tool:lookup_mue")

    if "addon" in topics:
        aoc = tools.lookup_aoc(code)
        if aoc.get("found") and aoc.get("addon_codes_allowed"):
            bullets.append(
                "- Add-on codes: "
                + ", ".join(f"**{c}**" for c in aoc["addon_codes_allowed"])
                + "."
            )
        elif aoc.get("found") and aoc.get("is_addon_code"):
            bullets.append(f"- Add-on code (parent **{aoc.get('parent_code')}**).")
        else:
            bullets.append("- Add-on codes: None listed.")
        answered.append("addon")
        sources.append("billing_tool:lookup_aoc")

    if "icd" in topics:
        icd = tools.lookup_icd(code)
        if icd.get("found"):
            bullets.append(f"- ICD-10 mappings: **{icd.get('count', 0)}** codes.")
        else:
            bullets.append("- ICD-10 mapping: Not available.")
        answered.append("icd")
        sources.append("billing_tool:lookup_icd")

    if "category" in topics or (
        profile is not None and profile.billing_rule != EIGHT_MINUTE_RULE and include_summaryish
    ):
        note = explain_special_category(code, concise=True)
        if note:
            bullets.append(f"- Category: {note}")
            answered.append("category")
            sources.append("billing_category_engine")
        elif profile:
            bullets.append(
                f"- Category: {profile.category_id} (`{profile.billing_rule}`)."
            )
            answered.append("category")
            sources.append("billing_category_engine")

    # Deduplicate bullets
    deduped: list[str] = []
    seen: set[str] = set()
    for bullet in bullets:
        key = bullet.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(bullet)

    heading = f"**CPT {code}**" if concise else f"### CPT {code}"
    if not deduped:
        # No per-code bullets for this topic set — omit the section entirely.
        return "", answered, sources
    return f"{heading}\n" + "\n".join(deduped), answered, sources


def _ncci_pairs(codes: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(codes):
        for right in codes[i + 1 :]:
            pairs.append((left, right))
    return pairs


def _build_shared_sections(
    *,
    question: str,
    codes: list[str],
    topics: set[str],
    tools: BillingTools,
    concise: bool,
) -> tuple[list[str], list[str], list[str]]:
    sections: list[str] = []
    answered: list[str] = []
    sources: list[str] = []

    if ("ncci" in topics or "modifier" in topics) and len(codes) >= 2:
        lines = ["**Billed together (NCCI)**" if concise else "### Billed together (NCCI)"]
        any_requires_59 = False
        any_blocked = False
        for left, right in _ncci_pairs(codes):
            ncci = tools.check_ncci(left, right)
            if ncci.get("allowed"):
                lines.append(f"- **{left}** + **{right}**: Yes, can be billed together.")
            else:
                any_blocked = True
                lines.append(
                    f"- **{left}** + **{right}**: No, not billable together under "
                    "current NCCI rules."
                )
            if ncci.get("modifier59_required"):
                any_requires_59 = True
            sources.append("billing_tool:check_ncci")
        answered.append("ncci")

        if "modifier" in topics:
            lines.append("**Modifier 59**" if concise else "### Modifier 59")
            if any_requires_59:
                lines.append(
                    "- Modifier 59 may be required when listed pairs are distinct."
                )
            elif any_blocked:
                lines.append("- Modifier 59 does not bypass NCCI edits that block a pair.")
            else:
                lines.append(
                    "- Not required based on the current NCCI data for these pairs."
                )
            answered.append("modifier")
        sections.append("\n".join(lines))
    elif "ncci" in topics and len(codes) < 2:
        sections.append(
            "**Billed together (NCCI)**\n"
            "- Provide a second CPT/HCPCS code to check billing together."
        )
        answered.append("ncci")
        if "modifier" in topics:
            sections.append(
                "**Modifier 59**\n- Need the second CPT billed the same day."
            )
            answered.append("modifier")
    elif "modifier" in topics and len(codes) < 2:
        sections.append(
            "**Modifier 59**\n- Need the second CPT billed the same day."
        )
        answered.append("modifier")

    wants_medicare = "medicare" in topics or "compare" in topics
    wants_ama = "ama" in topics or "compare" in topics
    wants_units = "units" in topics or is_unit_calculation_question(question)

    if wants_medicare or wants_ama or wants_units or "compare" in topics:
        entries = parse_cpt_time_entries(question)
        unit_lines: list[str] = []

        if entries:
            cms = calculate_cms_pooled_units(entries)
            ama = calculate_ama_units(entries)
            cms_total = sum(cms.values())
            ama_total = sum(ama.values())
            unit_lines.append(
                "**Medicare / AMA units**"
                if concise
                else "### Medicare / AMA units"
            )
            if wants_medicare or "compare" in topics or wants_units:
                unit_lines.append(f"- Medicare (CMS) units: **{cms_total}**.")
                answered.append("medicare")
            if wants_ama or "compare" in topics or wants_units:
                unit_lines.append(f"- AMA units: **{ama_total}**.")
                answered.append("ama")
            if wants_units or re.search(r"\bbrief\s+calculation\b", question, re.I):
                parts = ", ".join(
                    f"{e.cpt_code}={e.minutes} min" for e in entries
                )
                unit_lines.append(
                    f"- Brief calculation: documented {parts} → "
                    f"Medicare **{cms_total}**, AMA **{ama_total}**."
                )
                answered.append("units")
            if "compare" in topics:
                unit_lines.append(
                    "- CMS pools timed minutes; AMA calculates each CPT independently."
                )
                answered.append("compare")
            sources.append("billing_engine")
        else:
            unit_lines.append(
                "**Medicare / AMA units**"
                if concise
                else "### Medicare / AMA units"
            )
            if wants_medicare or "compare" in topics:
                unit_lines.append(
                    "- Medicare (CMS): Pool timed therapy minutes, then apply the "
                    "8-minute conversion table."
                )
                answered.append("medicare")
            if wants_ama or "compare" in topics:
                unit_lines.append(
                    "- AMA: Calculate each timed CPT independently (Rule of Eights)."
                )
                answered.append("ama")
            if wants_units:
                unit_lines.append(
                    "- Brief calculation: provide minutes per CPT for numeric units "
                    "(Category G time-band codes select one CPT — they are not "
                    "additive unit stacks)."
                )
                answered.append("units")
            if "compare" in topics:
                answered.append("compare")
            sources.append("billing_engine")

            # Still give per-code unit guidance when minutes were not provided
            for code in codes:
                profile = get_billing_category_store().get_profile(code)
                if profile and profile.billing_rule == TIME_BAND_SELECT:
                    unit_lines.append(
                        f"- **{code}**: Category G time-band — select **one** CPT "
                        "for the documented discussion band (not unit math)."
                    )
                elif profile and profile.billing_rule == FULL_BLOCK_REQUIRED:
                    unit_lines.append(
                        f"- **{code}**: full "
                        f"{profile.block_minutes}-minute blocks; "
                        f"max {profile.max_units_allowed}."
                    )
                elif profile and profile.billing_rule == AREA_BASED:
                    unit_lines.append(
                        f"- **{code}**: area-based (sq cm), not minute units."
                    )
                elif profile and profile.billing_rule in UNTIMED_RULES:
                    unit_lines.append(
                        f"- **{code}**: untimed — typically **1** unit when documented."
                    )
                elif profile and profile.billing_rule == EIGHT_MINUTE_RULE:
                    unit_lines.append(
                        f"- **{code}**: CMS 8-Minute Rule (timed)."
                    )

        sections.append("\n".join(unit_lines))

    return sections, answered, sources


def _fill_completeness_gaps(
    *,
    question: str,
    codes: list[str],
    topics: set[str],
    missing_codes: list[str],
    missing_topics: list[str],
    tools: BillingTools,
    store,
) -> tuple[str, list[str], list[str]]:
    parts: list[str] = []
    filled: list[str] = []
    sources: list[str] = []

    for code in missing_codes:
        mue = tools.lookup_mue(code)
        profile = store.get_profile(code)
        lines = [f"**CPT {code}**"]
        if mue.get("found") and mue.get("limit") is not None:
            lines.append(f"- MUE: **{mue['limit']}** units.")
            sources.append("billing_tool:lookup_mue")
        if profile:
            lines.append(
                f"- Category: {profile.category_id} (`{profile.billing_rule}`)."
            )
            sources.append("billing_category_engine")
        parts.append("\n".join(lines))
        filled.append("summary")

    for topic in missing_topics:
        if topic == "mue":
            for code in codes:
                mue = tools.lookup_mue(code)
                limit = mue.get("limit") if mue.get("found") else "n/a"
                parts.append(f"**MUE — {code}:** **{limit}**")
            filled.append("mue")
            sources.append("billing_tool:lookup_mue")
        elif topic in {"medicare", "ama", "compare", "units"}:
            compare = build_cms_ama_comparison(question)
            if compare:
                parts.append(compare)
            else:
                parts.append(
                    "**Medicare / AMA:** CMS pools timed minutes; AMA calculates "
                    "each CPT independently. Provide minutes for numeric units."
                )
            filled.extend(["medicare", "ama", "compare", "units"])
            sources.append("billing_engine")
        elif topic in {"ncci", "modifier"} and len(codes) >= 2:
            left, right = codes[0], codes[1]
            ncci = tools.check_ncci(left, right)
            status = "Yes" if ncci.get("allowed") else "No"
            parts.append(
                f"**NCCI — {left} + {right}:** {status}, billed together. "
                f"Modifier 59 required: "
                f"{'Yes' if ncci.get('modifier59_required') else 'No'}."
            )
            filled.extend(["ncci", "modifier"])
            sources.append("billing_tool:check_ncci")
        else:
            parts.append(f"**{topic.title()}:** See billing tools / category rules above.")
            filled.append(topic)

    return "\n\n".join(parts), filled, sources
