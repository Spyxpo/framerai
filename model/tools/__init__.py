"""Tools the model can call during a turn.

``model/serve.py`` builds a registry from ``--tools`` and hands it to the loop.
A capability that was not asked for is never registered, so the default worker
behaves exactly as it did before this package existed.
"""

from .base import Tool, ToolError, ToolRegistry, ToolResult
from .loop import ToolCall, ToolCallError, ToolStep, ToolTrace, parse_tool_call, run_tool_loop
from .web import SearchClient, SearchResult, WebFetchTool, WebSearchTool, web_tools

TOOLSETS = ("web",)
TOOLSET_TOOLS = {"web": ("web_search", "web_fetch")}


def expand_toolsets(names: list[str]) -> list[str]:
    """Expand toolset names to tool names, leaving tool names untouched."""
    expanded: list[str] = []
    for name in names:
        expanded.extend(TOOLSET_TOOLS.get(name, (name,)))
    return expanded


def build_registry(names: list[str] | str | None) -> ToolRegistry:
    """A registry for the named toolsets. ``None`` or empty gives an empty one."""
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    registry = ToolRegistry()
    for name in names or []:
        if name == "web":
            for tool in web_tools():
                registry.register(tool)
        else:
            raise ValueError(f"unknown toolset {name!r}; available: {', '.join(TOOLSETS)}")
    return registry


__all__ = [
    "TOOLSETS",
    "TOOLSET_TOOLS",
    "SearchClient",
    "SearchResult",
    "Tool",
    "ToolCall",
    "ToolCallError",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolStep",
    "ToolTrace",
    "WebFetchTool",
    "WebSearchTool",
    "build_registry",
    "expand_toolsets",
    "parse_tool_call",
    "run_tool_loop",
    "web_tools",
]
