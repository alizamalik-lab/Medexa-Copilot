import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class CodeType(str, Enum):
    CPT = "cpt"
    ICD10 = "icd10"
    HCPCS = "hcpcs"
    MODIFIER = "modifier"


@dataclass(frozen=True)
class DetectedCode:
    code: str
    code_type: CodeType
    metadata_field: str | None = None


@dataclass
class QueryRoute:
    """Parsed retrieval intent for a user question."""

    question: str
    codes: list[DetectedCode] = field(default_factory=list)
    mode: Literal["semantic", "metadata", "hybrid"] = "semantic"

    @property
    def has_exact_codes(self) -> bool:
        return bool(self.codes)


# Metadata field used in Chroma for each code type (extensible for future indexes).
CODE_METADATA_FIELDS: dict[CodeType, str | None] = {
    CodeType.CPT: "cpt_code",
    CodeType.HCPCS: "cpt_code",
    CodeType.ICD10: None,  # ICD codes live inside record content today
    CodeType.MODIFIER: None,  # Modifiers live inside PTP / policy content today
}

# Regex patterns for billing code detection.
CPT_PATTERN = re.compile(r"\b(\d{5})\b")
HCPCS_PATTERN = re.compile(r"\b([A-VJ-Z]\d{4})\b", re.IGNORECASE)
ICD10_PATTERN = re.compile(
    r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b",
    re.IGNORECASE,
)
MODIFIER_PATTERN = re.compile(
    r"\b(?:modifier|mod\.?)\s*[#:]?\s*([A-Z0-9]{2})\b",
    re.IGNORECASE,
)
STANDALONE_MODIFIER_PATTERN = re.compile(
    r"\b(?:modifier\s+)?(59|25|26|50|51|52|53|54|55|56|57|58|76|77|78|79|"
    r"80|81|82|90|91|92|95|96|97|99|LT|RT|TC|26|XE|XP|XS|XU)\b",
    re.IGNORECASE,
)


class QueryRouter:
    """Detect billing codes in a question and choose a retrieval mode."""

    def route(self, question: str) -> QueryRoute:
        codes = self._detect_codes(question)
        if not codes:
            mode: Literal["semantic", "metadata", "hybrid"] = "semantic"
        elif self._looks_like_mixed_question(question, codes):
            mode = "hybrid"
        else:
            mode = "metadata"
        return QueryRoute(question=question, codes=codes, mode=mode)

    def _detect_codes(self, question: str) -> list[DetectedCode]:
        found: dict[tuple[str, CodeType], DetectedCode] = {}

        for match in ICD10_PATTERN.finditer(question):
            code = match.group(1).upper()
            self._add_code(found, code, CodeType.ICD10)

        for match in HCPCS_PATTERN.finditer(question):
            code = match.group(1).upper()
            if self._is_icd10(code):
                continue
            self._add_code(found, code, CodeType.HCPCS)

        for match in CPT_PATTERN.finditer(question):
            code = match.group(1)
            self._add_code(found, code, CodeType.CPT)

        for pattern in (MODIFIER_PATTERN, STANDALONE_MODIFIER_PATTERN):
            for match in pattern.finditer(question):
                code = match.group(1).upper()
                self._add_code(found, code, CodeType.MODIFIER)

        return list(found.values())

    def _add_code(
        self,
        found: dict[tuple[str, CodeType], DetectedCode],
        code: str,
        code_type: CodeType,
    ) -> None:
        key = (code, code_type)
        if key in found:
            return
        found[key] = DetectedCode(
            code=code,
            code_type=code_type,
            metadata_field=CODE_METADATA_FIELDS[code_type],
        )

    def _is_icd10(self, code: str) -> bool:
        return bool(ICD10_PATTERN.fullmatch(code))

    def _looks_like_mixed_question(
        self, question: str, codes: list[DetectedCode]
    ) -> bool:
        lowered = question.lower()
        semantic_hints = (
            "what is",
            "explain",
            "describe",
            "tell me",
            "how does",
            "when should",
            "documentation",
            "billing",
            "policy",
            "rule",
            "guideline",
        )
        has_semantic_intent = any(hint in lowered for hint in semantic_hints)
        return has_semantic_intent and len(codes) >= 1
