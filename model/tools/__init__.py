"""Tools the model can call during a turn.

``model/serve.py`` builds a registry from ``--tools`` and hands it to the loop.
A capability that was not asked for is never registered, so the default worker
behaves exactly as it did before this package existed.
"""

from .base import Tool, ToolError, ToolRegistry, ToolResult
from .cli import ListDirTool, ReadFileTool, ShellPolicy, ShellTool, cli_tools
from .loop import ToolCall, ToolCallError, ToolStep, ToolTrace, parse_tool_call, run_tool_loop
from .web import SearchClient, SearchResult, WebFetchTool, WebSearchTool, web_tools

TOOLSETS = ("web", "cli")
TOOLSET_TOOLS = {
    "web": ("web_search", "web_fetch"),
    "cli": ("shell", "read_file", "list_dir"),
}


def expand_toolsets(names: list[str]) -> list[str]:
    """Expand toolset names to tool names, leaving tool names untouched."""
    expanded: list[str] = []
    for name in names:
        expanded.extend(TOOLSET_TOOLS.get(name, (name,)))
    return expanded


def build_registry(
    names: list[str] | str | None, cli_policy: ShellPolicy | None = None
) -> ToolRegistry:
    """A registry for the named toolsets. ``None`` or empty gives an empty one.

    The cli toolset needs a policy; without one it is built with the default,
    which is mode ``off`` and therefore refuses every command. That is
    deliberate - a caller has to say what the shell may do.
    """
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    registry = ToolRegistry()
    for name in names or []:
        if name == "web":
            tools = web_tools()
        elif name == "cli":
            tools = cli_tools(cli_policy)
        else:
            raise ValueError(f"unknown toolset {name!r}; available: {', '.join(TOOLSETS)}")
        for tool in tools:
            registry.register(tool)
    return registry


__all__ = [
    "TOOLSETS",
    "TOOLSET_TOOLS",
    "ListDirTool",
    "ReadFileTool",
    "SearchClient",
    "SearchResult",
    "ShellPolicy",
    "ShellTool",
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
    "cli_tools",
    "expand_toolsets",
    "parse_tool_call",
    "run_tool_loop",
    "web_tools",
]
