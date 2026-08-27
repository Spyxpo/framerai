"""The tool protocol every FramerAI tool implements.

A tool is a name, a one-line description, a JSON parameter schema, and a
``run()`` that returns a :class:`ToolResult`. Nothing here knows about the web,
the shell, or the model - the loop in ``model/tools/loop.py`` drives tools
purely through this interface, so gating a capability is a matter of not
registering it.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """A tool refused or failed.

    Raised rather than returned when the tool itself cannot proceed. The
    registry converts it into a failed :class:`ToolResult`, because a model that
    called a tool wrongly should see the reason and get another turn, not lose
    the conversation to a traceback.
    """


@dataclass
class ToolResult:
    """What a tool hands back: text for the model, data for the caller."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **data: Any) -> "ToolResult":
        return cls(ok=True, content=content, data=data)

    @classmethod
    def failure(cls, content: str, **data: Any) -> "ToolResult":
        return cls(ok=False, content=content, data=data)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "content": self.content, "data": self.data}


class Tool:
    """Base class for tools. Subclasses set the three class attributes."""

    name: str = ""
    description: str = ""
    parameters: dict[str, str] = {}

    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def describe(self) -> str:
        """One compact line for the prompt: name, arguments, purpose."""
        args = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({args}) - {self.description}"


class ToolRegistry:
    """The set of tools a worker is willing to run, in registration order."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> "ToolRegistry":
        if not tool.name:
            raise ValueError("a tool needs a name")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A registry holding only the named tools, ignoring unknown names.

        Unknown names are dropped rather than rejected: a client asking for a
        capability this worker was not started with should get a worker without
        that capability, not an error.
        """
        picked = [self._tools[n] for n in names if n in self._tools]
        return ToolRegistry(picked)

    def describe(self) -> str:
        return "\n".join(tool.describe() for tool in self._tools.values())

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool by name, turning every failure into a failed result."""
        tool = self.get(name)
        if tool is None:
            known = ", ".join(self.names()) or "none"
            return ToolResult.failure(f"unknown tool {name!r}; available: {known}")
        if not isinstance(arguments, dict):
            return ToolResult.failure(f"arguments for {name!r} must be an object")
        try:
            return tool.run(**arguments)
        except ToolError as exc:
            return ToolResult.failure(str(exc))
        except TypeError as exc:
            # Wrong or missing arguments: the model gets to correct itself.
            return ToolResult.failure(f"bad arguments for {name!r}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the turn
            return ToolResult.failure(f"{name} failed: {exc}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
