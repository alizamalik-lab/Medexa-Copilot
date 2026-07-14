"""Category G — phone / online time-band billing (pick one CPT, not additive)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.billing_categories import CodeBillingProfile, get_billing_category_store
from rag.billing_engine import TIME_BAND_SELECT, minutes_in_time_band
from rag.category_tools import CategoryToolResult, profile_payload


@dataclass(frozen=True)
class ServiceFamily:
    family_id: str
    label: str
    keywords: tuple[str, ...]
    # ordered (band_label, code) pairs — most specific bands first where needed
    codes_by_band: tuple[tuple[str, str], ...]


# Logical service families spanning Category G subcategories.
_SERVICE_FAMILIES: tuple[ServiceFamily, ...] = (
    ServiceFamily(
        family_id="telephone",
        label="Telephone assessment and management",
        keywords=(
            "telephone",
            "phone call",
            "phone",
            "called",
            "spoke",
            "spoke with",
            "talked",
            "telephonic",
        ),
        codes_by_band=(
            ("5-10 minutes", "98966"),
            ("11-20 minutes", "98967"),
            ("21-30 minutes", "98968"),
        ),
    ),
    ServiceFamily(
        family_id="online_digital",
        label="Online digital assessment and management",
        keywords=(
            "online digital",
            "patient portal",
            "portal message",
            "secure message",
            "e-visit",
            "evisit",
            "asynchronous",
            "online",
            "digital assessment",
        ),
        codes_by_band=(
            ("5-10 minutes", "98970"),
            ("11-20 minutes", "98971"),
            ("21+ minutes", "98972"),
        ),
    ),
    ServiceFamily(
        family_id="brief_check_in",
        label="Brief communication technology-based check-in",
        keywords=(
            "virtual check-in",
            "virtual check in",
            "brief check-in",
            "brief check in",
            "g2251",
        ),
        codes_by_band=(("5-10 minutes", "G2251"),),
    ),
)

_PHONE_ONLINE_HINTS = re.compile(
    r"\b("
    r"telephone|phone call|phone|called|spoke|talked|telephonic|"
    r"patient portal|portal message|secure message|e-?visit|online digital|"
    r"virtual check[\s-]?in|brief check[\s-]?in|"
    r"advice|consultation|consult"
    r")\b",
    re.IGNORECASE,
)

_DURATION = re.compile(
    r"(?:lasts|lasted|took|took about|session(?:\s+lasts)?|discussion|call)?\s*"
    r"(?:for|of|about|over|with\s+me\s+for|with\s+them\s+for)?\s*"
    r"(?:about\s+|approximately\s+|around\s+|exactly\s+)?"
    r"(\d{1,3})\s*(?:minutes?|mins?)\b"
    r"|"
    r"\b(?:spoke|talked|discussed|discussion|session)\b.{0,40}?\b(\d{1,3})\s*(?:minutes?|mins?)\b"
    r"|"
    r"\b(\d{1,3})\s*(?:minutes?|mins?)\b"
    r"(?:\s+(?:call|conversation|discussion|phone|telephone|session))?",
    re.IGNORECASE,
)

_CODE_LABELS = {
    "98966": "Telephone assessment and management, 5–10 minutes",
    "98967": "Telephone assessment and management, 11–20 minutes",
    "98968": "Telephone assessment and management, 21–30 minutes",
    "98970": "Online digital assessment and management, 5–10 minutes",
    "98971": "Online digital assessment and management, 11–20 minutes",
    "98972": "Online digital assessment and management, 21+ minutes",
    "G2251": "Brief communication technology-based check-in, 5–10 minutes",
}


def is_phone_online_question(question: str) -> bool:
    """True for phone/online consult questions (Category G), even without a CPT."""
    lowered = question.lower()
    if not _PHONE_ONLINE_HINTS.search(lowered):
        return False
    # Prefer Category G when discussing advice/call duration without wound context
    if re.search(r"\b(wound|debrid|sq\.?\s*cm)\b", lowered):
        return False
    return True


def is_time_band_category_question(question: str) -> bool:
    """True when the user explicitly asks about Category G / time_band_select."""
    lowered = question.lower()
    if re.search(r"\b(wound|debrid|sq\.?\s*cm)\b", lowered):
        return False
    return bool(
        re.search(
            r"\b("
            r"time[\s_-]?band[\s_-]?select|time[\s_-]?band|"
            r"category\s*g\b|categor(?:y|ies)\s+g\b|"
            r"which time[\s_-]?band code|time[\s_-]?band code"
            r")\b",
            lowered,
        )
    )


def is_category_g_question(question: str) -> bool:
    """Route Category G whenever phone/online OR explicit time-band selection is asked."""
    return is_phone_online_question(question) or is_time_band_category_question(question)


def extract_discussion_minutes(question: str) -> int | None:
    match = _DURATION.search(question)
    if not match:
        return None
    raw = next((g for g in match.groups() if g), None)
    return int(raw) if raw else None


def detect_service_family(question: str) -> ServiceFamily | None:
    """Detect service family; phone call cues default to telephone (not additive units)."""
    lowered = question.lower()

    # Brief check-in first (most specific)
    check_in = _SERVICE_FAMILIES[2]
    if any(keyword in lowered for keyword in check_in.keywords):
        return check_in

    online = _SERVICE_FAMILIES[1]
    online_markers = (
        "portal",
        "secure message",
        "e-visit",
        "evisit",
        "asynchronous",
        "online digital",
        "digital assessment",
        "online assessment",
        "online consult",
        "online consultation",
    )
    if any(k in lowered for k in online_markers):
        return online

    telephone = _SERVICE_FAMILIES[0]
    if any(
        k in lowered
        for k in (
            "telephone assessment",
            "telephone a&m",
            "telephonic assessment",
            "phone assessment",
        )
    ) or (re.search(r"\btelephone\b", lowered) and "assessment" in lowered):
        return telephone

    # Explicit selection from clarification replies
    if re.search(r"\btelephone assessment and management\b", lowered):
        return telephone
    if re.search(r"\bonline digital assessment and management\b", lowered):
        return online
    if re.search(r"\bbrief communication technology-based check-in\b", lowered):
        return check_in
    if lowered.strip() in {"telephone", "phone", "phone call"}:
        return telephone
    if lowered.strip() in {"online", "online digital", "portal", "digital"}:
        return online
    if "check-in" in lowered or "check in" in lowered:
        return check_in

    # Strong phone-call scenario → telephone family (skip clarification).
    phone_cues = (
        "phone call",
        "telephone",
        "telephonic",
        "called me",
        "called the",
        "called patient",
        "patient called",
        "spoke with",
        "spoke for",
        "talked for",
        "on the phone",
        "that call",
        "for that call",
        "phone consultation",
        "phone consult",
    )
    if any(cue in lowered for cue in phone_cues) or re.search(
        r"\b(called|phone)\b", lowered
    ):
        if not any(k in lowered for k in online_markers):
            return telephone

    return None


def matching_codes_for_minutes(minutes: int) -> list[tuple[ServiceFamily, str, str]]:
    """Return (family, band_label, code) candidates whose band includes minutes."""
    matches: list[tuple[ServiceFamily, str, str]] = []
    for family in _SERVICE_FAMILIES:
        for band, code in family.codes_by_band:
            if minutes_in_time_band(minutes, band):
                matches.append((family, band, code))
    return matches


def format_time_band_label(band: str) -> str:
    # Keep "minutes" for readability: "21–30 minutes"
    return band.replace("-", "–")


def _preferred_band_label(
    minutes: int, matches: list[tuple[ServiceFamily, str, str]]
) -> str:
    """Choose a friendly band label for clarifications (prefer closed ranges)."""
    bands = [band for _, band, _ in matches]
    closed = [b for b in bands if re.search(r"\d+\s*-\s*\d+", b)]
    if closed:
        for band in closed:
            if minutes_in_time_band(minutes, band):
                return band
        return closed[0]
    return bands[0]


class TimeBandBillingTool:
    """Handles Category G time-band codes and phone/online recommendation flow."""

    name = "time_band_billing_tool"
    billing_rules = (TIME_BAND_SELECT,)

    def supports(self, billing_rule: str) -> bool:
        return billing_rule == TIME_BAND_SELECT

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
        # Phone/online recommendation path (often no CPT yet)
        if is_phone_online_question(question) or not codes:
            phone_result = self._run_phone_online(question, codes, minutes_by_code)
            if phone_result is not None:
                return phone_result

        return self._run_known_code(
            question=question,
            codes=codes,
            profiles=profiles,
            minutes_by_code=minutes_by_code,
        )

    def _run_phone_online(
        self,
        question: str,
        codes: list[str],
        minutes_by_code: dict[str, int],
    ) -> CategoryToolResult | None:
        if codes:
            # Known Category G code — still allow family resolution if minutes given
            store = get_billing_category_store()
            profile = store.get_profile(codes[0])
            if profile is None or profile.billing_rule != TIME_BAND_SELECT:
                if not is_phone_online_question(question):
                    return None

        minutes = extract_discussion_minutes(question)
        if minutes is None and minutes_by_code:
            minutes = next(iter(minutes_by_code.values()))
        if minutes is None:
            # Try bare duration already parsed into minutes_by_code empty - ask
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer="",
                needs_clarification=True,
                clarification=(
                    "Telephone and online services are billed by total discussion time "
                    "(one CPT for the matching time band — not additive units).\n\n"
                    "What was the total documented discussion time in minutes?"
                ),
                structured={"tool": self.name, "needs": "duration"},
            )

        family = detect_service_family(question)
        matches = matching_codes_for_minutes(minutes)
        if not matches:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer=(
                    f"Based on **{minutes} minutes**, this does not fall into a "
                    "standard phone/online assessment time band "
                    "(typical bands: 5–10, 11–20, 21–30, or 21+ minutes).\n\n"
                    "Confirm the documented discussion time, or share the CPT you use."
                ),
                structured={"tool": self.name, "minutes": minutes, "matches": []},
            )

        # Filter by detected family when possible
        if family is not None:
            family_matches = [m for m in matches if m[0].family_id == family.family_id]
            if family_matches:
                matches = family_matches

        # Unique code for the minutes + family
        unique_codes = list(dict.fromkeys(code for _, _, code in matches))
        if len(unique_codes) == 1:
            band = matches[0][1]
            code = unique_codes[0]
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer=self._format_selection(minutes, band, code, question=question),
                structured={
                    "tool": self.name,
                    "minutes": minutes,
                    "time_band": band,
                    "selected_code": code,
                    "units": 1,
                },
            )

        # Multiple candidates across service types → surface JSON band overlap.
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=TIME_BAND_SELECT,
            answer="",
            needs_clarification=True,
            clarification=self._format_overlap_clarification(minutes, matches),
            structured={
                "tool": self.name,
                "minutes": minutes,
                "overlap": True,
                "candidates": [
                    {"family": f.family_id, "band": band, "code": code}
                    for f, band, code in matches
                ],
                "needs": "service_type",
            },
        )

    def _clarification_options(
        self, matches: list[tuple[ServiceFamily, str, str]]
    ) -> list[str]:
        seen: set[str] = set()
        options: list[str] = []
        for family, _band, _code in matches:
            if family.family_id in seen:
                continue
            seen.add(family.family_id)
            options.append(family.label)
        return options

    def _format_overlap_clarification(
        self,
        minutes: int,
        matches: list[tuple[ServiceFamily, str, str]],
    ) -> str:
        """Explain overlapping Category G bands from the JSON dataset (no payer ask)."""
        store = get_billing_category_store()
        lines = [
            f"Based on **{minutes} minutes**, this duration overlaps more than one "
            "Category G (`time_band_select`) time band in the billing dataset. "
            "This is a data ambiguity between code families — the JSON already "
            "lists the matching candidates below.",
            "",
            "Candidate codes from the JSON:",
        ]
        for family, band, code in matches:
            profile = store.get_profile(code)
            sub_id = profile.subcategory_id if profile else None
            json_band = (
                profile.time_band
                if profile and profile.time_band
                else band
            )
            sub_label = f"**{sub_id}** " if sub_id else ""
            lines.append(
                f"• {sub_label}(`{format_time_band_label(json_band)}`) → "
                f"**{code}** — {family.label}"
            )
        lines.extend(
            [
                "",
                "Only **one** time-band CPT can be selected for the encounter "
                "(bands are not additive).",
                "",
                "To disambiguate, which service/code family applies?",
                "",
            ]
        )
        lines.extend(f"• {option}" for option in self._clarification_options(matches))
        return "\n".join(lines)

    def _format_selection(
        self, minutes: int, band: str, code: str, *, question: str = ""
    ) -> str:
        label = _CODE_LABELS.get(code, f"CPT {code}")
        bill_ask = bool(
            re.search(
                r"\b("
                r"can i bill|can we bill|bill(?:able)?|"
                r"bill (?:him|her|them|this|for|the)|"
                r"is (?:this|it) billable"
                r")\b",
                question,
                re.IGNORECASE,
            )
        )
        lead = "Yes. " if bill_ask else ""
        return (
            f"{lead}Based on **{minutes} minutes**, this falls into the "
            f"**{format_time_band_label(band)}** time band.\n\n"
            f"**Recommended CPT:** **{code}** — {label}.\n\n"
            "Category G is CPT selection by time band — report **one** CPT for "
            "the encounter (not additive unit stacking across time-band codes)."
        )

    def _run_known_code(
        self,
        *,
        question: str,
        codes: list[str],
        profiles: list[CodeBillingProfile],
        minutes_by_code: dict[str, int],
    ) -> CategoryToolResult:
        store = get_billing_category_store()
        profile = profiles[0] if profiles else None
        code = codes[0] if codes else (profile.cpt_code if profile else None)
        if profile is None and code:
            profile = store.get_profile(code)
        if profile is None or code is None:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer="",
                needs_clarification=True,
                clarification=(
                    "Telephone and online services use time-band CPTs. "
                    "What was the total documented discussion time in minutes?"
                ),
            )

        minutes = minutes_by_code.get(code)
        if minutes is None:
            minutes = extract_discussion_minutes(question)
        if minutes is None and len(minutes_by_code) == 1:
            minutes = next(iter(minutes_by_code.values()))

        if minutes is None:
            band = profile.time_band or profile.min_time_for_1_unit or "see time band"
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer=(
                    f"**{code}** is a time-band code (**{band}**). "
                    "Bill **1 unit** when total discussion time falls in that band.\n\n"
                    "What was the total documented discussion time in minutes?"
                ),
                structured={
                    "tool": self.name,
                    "profile": profile_payload(profile),
                },
            )

        band = profile.time_band or profile.min_time_for_1_unit or ""
        qualifies = minutes_in_time_band(minutes, band) if band else False
        if qualifies:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer=self._format_selection(minutes, band, code, question=question),
                structured={
                    "tool": self.name,
                    "minutes": minutes,
                    "time_band": band,
                    "selected_code": code,
                    "units": 1,
                    "profile": profile_payload(profile),
                },
            )

        # Suggest better-fitting codes for the same service family
        matches = matching_codes_for_minutes(minutes)
        family = next(
            (
                f
                for f in _SERVICE_FAMILIES
                if any(c == code for _, c in f.codes_by_band)
            ),
            None,
        )
        if family:
            matches = [m for m in matches if m[0].family_id == family.family_id]
        if len(matches) == 1:
            alt_band, alt_code = matches[0][1], matches[0][2]
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule=TIME_BAND_SELECT,
                answer=(
                    f"**{minutes} minutes** does not fall in the "
                    f"**{format_time_band_label(band)}** band for **{code}**.\n\n"
                    + self._format_selection(
                        minutes, alt_band, alt_code, question=question
                    )
                ),
                structured={
                    "tool": self.name,
                    "minutes": minutes,
                    "requested_code": code,
                    "selected_code": alt_code,
                },
            )

        suggestions = ", ".join(f"**{c}** ({b})" for _, b, c in matches) or "none"
        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=TIME_BAND_SELECT,
            answer=(
                f"**{minutes} minutes** does not match **{code}** "
                f"({format_time_band_label(band)}). "
                f"Possible matches: {suggestions}."
            ),
            structured={"tool": self.name, "minutes": minutes, "requested_code": code},
        )
