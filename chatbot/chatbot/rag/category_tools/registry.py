"""Registry mapping billing_rule → category tool (Bedrock-ready)."""

from __future__ import annotations

from rag.billing_engine import (
    AREA_BASED,
    EIGHT_MINUTE_RULE,
    FULL_BLOCK_REQUIRED,
    TIME_BAND_SELECT,
    UNTIMED_RULES,
)
from rag.category_tools.addon import AddonBillingTool
from rag.category_tools.area_based import AreaBasedBillingTool
from rag.category_tools.full_block import FullBlockBillingTool
from rag.category_tools.time_band import TimeBandBillingTool
from rag.category_tools.timed import TimedBillingTool
from rag.category_tools.untimed import UntimedBillingTool


_TOOLS = (
    TimedBillingTool(),
    FullBlockBillingTool(),
    UntimedBillingTool(),
    AreaBasedBillingTool(),
    TimeBandBillingTool(),
    AddonBillingTool(),
)


def get_tool_for_rule(billing_rule: str):
    for tool in _TOOLS:
        if tool.supports(billing_rule):
            return tool
    return None


def list_category_tools() -> list[dict[str, str]]:
    """Describe tools for future Bedrock agent tool selection."""
    return [
        {
            "name": tool.name,
            "billing_rules": ",".join(tool.billing_rules),
        }
        for tool in _TOOLS
    ]


def tool_name_for_rule(billing_rule: str) -> str:
    mapping = {
        EIGHT_MINUTE_RULE: "timed_billing_tool",
        FULL_BLOCK_REQUIRED: "full_block_billing_tool",
        AREA_BASED: "area_based_billing_tool",
        TIME_BAND_SELECT: "time_band_billing_tool",
        **{rule: "untimed_billing_tool" for rule in UNTIMED_RULES},
        "addon": "addon_billing_tool",
    }
    return mapping.get(billing_rule, "untimed_billing_tool")
