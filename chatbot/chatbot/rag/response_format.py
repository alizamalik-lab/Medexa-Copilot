"""Detect the best response format based on user question intent."""

from __future__ import annotations

import re
from enum import Enum

from rag.billing_engine import is_rule_comparison_question, is_unit_calculation_question
from rag.intent_detector import is_multi_topic_question


class ResponseFormat(str, Enum):
    SIMPLE = "simple"
    COMPARISON = "comparison"
    MULTIPLE = "multiple"
    CALCULATION = "calculation"
    LONG_EXPLANATION = "long_explanation"


_COMPARISON_HINTS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"medicare.{0,20}ama",
    r"ama.{0,20}medicare",
    r"cms.{0,20}ama",
    r"\bpt\b.{0,15}\bot\b",
    r"\bot\b.{0,15}\bslp\b",
    r"timed.{0,15}untimed",
    r"evaluation.{0,20}re-?evaluation",
)

_CALCULATION_HINTS = (
    r"calculate",
    r"how many units",
    r"billable units",
    r"8[\s-]minute",
    r"eight[\s-]minute",
    r"rule of eight",
    r"unit calculation",
)

_LONG_EXPLANATION_HINTS = (
    r"in detail",
    r"walk me through",
    r"comprehensive",
    r"documentation requirements",
    r"workflow",
    r"how does",
    r"when should",
    r"best practices",
    r"explain everything",
    r"full explanation",
)


def detect_response_format(
    question: str,
    *,
    has_unit_calculation: bool = False,
    tool_count: int = 0,
) -> ResponseFormat:
    lowered = question.lower().strip()

    if has_unit_calculation and tool_count <= 1:
        return ResponseFormat.CALCULATION

    if is_rule_comparison_question(question) or _matches_any(
        lowered, _COMPARISON_HINTS
    ):
        return ResponseFormat.COMPARISON

    if (
        is_multi_topic_question(question)
        or tool_count > 1
        or (has_unit_calculation and tool_count > 0)
    ):
        return ResponseFormat.MULTIPLE

    if has_unit_calculation or (
        _matches_any(lowered, _CALCULATION_HINTS)
        and is_unit_calculation_question(question)
    ):
        return ResponseFormat.CALCULATION

    if _matches_any(lowered, _LONG_EXPLANATION_HINTS) or (
        len(question) >= 120
        and any(
            hint in lowered
            for hint in ("explain", "describe", "overview", "guidance", "requirements")
        )
    ):
        return ResponseFormat.LONG_EXPLANATION

    return ResponseFormat.SIMPLE


def get_format_instructions(response_format: ResponseFormat) -> str:
    instructions = {
        ResponseFormat.SIMPLE: SIMPLE_FORMAT_RULES,
        ResponseFormat.COMPARISON: COMPARISON_FORMAT_RULES,
        ResponseFormat.MULTIPLE: MULTIPLE_FORMAT_RULES,
        ResponseFormat.CALCULATION: CALCULATION_FORMAT_RULES,
        ResponseFormat.LONG_EXPLANATION: LONG_EXPLANATION_FORMAT_RULES,
    }
    return instructions[response_format]


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


SIMPLE_FORMAT_RULES = """
**Format: Concise direct answer (1–5 lines)**

Give only the direct answer unless the user asked for detail.
Lead with the fact. Bold key values. No long setup paragraphs.

Examples:
- MUE: **6** units.
- Yes. It is a timed CPT code billed under the CMS 8-Minute Rule.
- Medicare Units: **2**.
"""

CALCULATION_FORMAT_RULES = """
**Format: Calculation result (concise by default)**

If the user did NOT ask to show calculation / step-by-step:
- Return only the final units (e.g. "Medicare Units: **2**.").

If they asked for steps, show numbered calculation steps and the final answer.
For CMS Medicare calculations, use the CMS conversion table (NOT "divide by 8").
"""

COMPARISON_FORMAT_RULES = """
**Format: Compact comparison table**

Show only key differences in a short Markdown table, then one sentence on why they differ.
Do not include lengthy calculation walkthroughs unless asked.

Example:

| Rule | Units |
|------|------:|
| Medicare (CMS) | 1 |
| AMA | 2 |

CMS pools timed minutes; AMA calculates each CPT independently.
"""

MULTIPLE_FORMAT_RULES = """
**Format: Compact multi-topic bullets**

Answer EACH asked item with one concise bullet under the CPT heading.
Do not write long paragraphs.

Example:

**CPT 97110**
- Description: ...
- Timed: Yes.
- MUE: **6** units.
- With 97530: Yes, they can be billed together.
- Modifier 59: Not required based on the current NCCI data.
"""

LONG_EXPLANATION_FORMAT_RULES = """
**Format: Summary + details**

Start with a short **Summary** (1–2 sentences), then provide structured details.

Example layout:

**Summary:**
A therapy re-evaluation is performed when the patient's condition changes significantly.

**Details:**
• Definition
• When it is performed
• Documentation requirements
• Billing considerations
• Practical example

Use `###` subheadings or bullets. Keep paragraphs short.
"""
