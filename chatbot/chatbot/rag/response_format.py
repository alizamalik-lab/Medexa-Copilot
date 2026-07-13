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
**Format: Simple answer (2–4 lines)**

Give a concise, confident answer in 2–4 short lines.
Lead with the direct answer. Bold key values such as codes, limits, and unit counts.

Example:
The MUE (Medically Unlikely Edit) limit for CPT 97110 is **6 units** per date of service.
"""

COMPARISON_FORMAT_RULES = """
**Format: Comparison table**

The user is comparing two or more concepts. Use a clear Markdown table — not a long paragraph.

For CMS 8-Minute Rule vs AMA Rule of Eight, use:

| Topic | CMS 8-Minute Rule | AMA Rule of Eight |
|-------|-------------------|-------------------|
| Used By | Medicare | Commonly used by many commercial payers |
| Calculation Method | Pool all timed CPT minutes | Calculate each CPT code separately |
| Unit Assignment | Based on total pooled minutes | Each CPT must independently qualify |
| Typical Use | Medicare billing | Commercial payer billing (when adopted) |

End with:
"Always verify the payer's billing guidelines, as commercial insurance policies may differ."
"""

MULTIPLE_FORMAT_RULES = """
**Format: Multiple sections**

The user asked several questions in one message. Do NOT write one long paragraph.
Organize the answer with `###` headings and bullet points under each section.

Example layout:

### CPT 97110
• Description...

### MUE
• **6 units**

### ICD-10 Mapping
• ...

### Billing
• ...

### Medicare Units
• **2 billable units** under the CMS 8-Minute Rule
"""

CALCULATION_FORMAT_RULES = """
**Format: Step-by-step calculation**

Walk through the calculation with numbered steps, then state the final answer.
For CMS Medicare calculations, use the CMS conversion table (NOT "divide by 8").

Example layout:

**Step 1:** Total timed minutes = **23 minutes**

**Step 2:** Rule applied = **CMS 8-Minute Rule**
• 8–22 minutes = 1 unit
• 23–37 minutes = 2 units

**Step 3:** Billable units = **2 units**

**Final Answer:** **2 units**
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
