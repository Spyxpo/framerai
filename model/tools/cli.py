"""Command line access: a sandboxed shell tool and read-only file helpers.

A model that can list a directory, read a file, and run the test suite is a
different tool from one that can only write about doing those things. What makes
that safe is not the shell - it is the policy in front of it.

:class:`ShellPolicy` decides before anything is spawned: a mode (``off``,
``ask``, ``allow``), an allowlist of permitted programs, a deny list that applies
in every mode, a sandbox root no argument may escape, a timeout, and an output
cap. ``ShellTool.run`` asks the policy first and never reaches ``subprocess`` on
a refusal, so a denial costs nothing and is recorded with its reason.

Default is ``off``: without ``--tools cli`` the tools are never registered and
the worker has no shell at all.
"""

import os
import re
import shlex
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .base import Tool, ToolError, ToolResult

MODES = ("off", "ask", "allow")

#: Programs that read, build, or test. Nothing here writes outside the sandbox
#: on its own, and the deny list still applies to every one of them.
DEFAULT_ALLOWLIST = (
    "awk",
    "basename",
    "cat",
    "cut",
    "date",
    "diff",
    "dirname",
    "du",
    "echo",
    "find",
    "git",
    "grep",
    "head",
    "ls",
    "node",
    "npm",
    "printenv",
    "pwd",
    "python",
    "python3",
    "pytest",
    "rg",
    "ruff",
    "sed",
    "sort",
    "stat",
    "tail",
    "tr",
    "uniq",
    "wc",
    "which",
)

#: Shapes that are refused in every mode, including ``allow``. Matched against
#: the normalised argv, so ``rm  -rf   /`` and ``rm -rf /`` are the same input.
DENY_PATTERNS = (
    (r"^rm\b.*\s-[a-z]*[rf][a-z]*\b.*\s(/|/\*|~|\$HOME)\s*$", "recursive delete of a root path"),
    (r"^rm\b.*\s-[a-z]*[rf]", "recursive or forced delete"),
    (r"^(mkfs|fdisk|parted)\b", "filesystem or partition modification"),
    (r"^dd\b.*\bof=/dev/", "raw write to a device"),
    (r"^(shutdown|reboot|halt|poweroff|init)\b", "host power state change"),
    (r"^(sudo|su|doas)\b", "privilege escalation"),
    (r"^chmod\b.*\b777\b", "world-writable permissions"),
    (r"^(curl|wget)\b.*\|\s*(sh|bash|zsh)", "piping a download into a shell"),
    (r"^kill\b.*\s-9\s+-1\b", "killing every process"),
    (r":\(\)\s*\{.*\|.*&.*\}", "fork bomb"),
    (r"^(nc|ncat|netcat|telnet)\b", "raw network access; use the web tools"),
    (r"^(git)\b.*\bpush\b.*--force", "force push"),
)

#: Operators that only mean something to a shell. Commands run with
#: ``shell=False``, so an unquoted one would be handed to the program as literal
#: text and silently do nothing - refusing is clearer than surprising the caller.
#: Quoted occurrences are fine: ``python -c 'import time; time.sleep(1)'`` is one
#: program with one argument, and the semicolon belongs to Python.
SHELL_OPERATORS = frozenset(";|&<>()")

#: The child gets a scrubbed environment: enough to find programs and read
#: files, and none of the parent's tokens.
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM")

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_OUTPUT = 10_000
DEFAULT_MAX_LINES = 200

_COMPILED_DENY = tuple((re.compile(pattern), reason) for pattern, reason in DENY_PATTERNS)

Approver = Callable[[str, list[str]], bool]


def _normalise(argv: list[str]) -> str:
    return " ".join(argv)


def lex(command: str) -> list[str]:
    """Split a command, keeping unquoted shell operators as their own tokens.

    ``shlex.split`` would hide an unquoted ``;`` inside a token; with
    ``punctuation_chars`` the operators come back separately, which is what lets
    ``ls; rm -rf /`` be refused while ``python -c 'a; b'`` is not.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


@dataclass
class Decision:
    """Why a command may or may not run."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class ShellPolicy:
    """Everything that decides whether a command runs, in one object."""

    mode: str = "off"
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST
    root: str = "."
    timeout: float = DEFAULT_TIMEOUT
    max_output: int = DEFAULT_MAX_OUTPUT
    approve: Approver | None = None
    denied: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown cli mode {self.mode!r}; expected one of {', '.join(MODES)}")
        self.root = os.path.realpath(self.root)
        self.allowlist = tuple(self.allowlist)

    def resolve(self, path: str) -> str:
        """A path inside the sandbox, or a :class:`ToolError`."""
        candidate = os.path.realpath(os.path.join(self.root, os.path.expanduser(path)))
        if candidate != self.root and not candidate.startswith(self.root + os.sep):
            raise ToolError(f"{path!r} is outside the sandbox at {self.root}")
        return candidate

    def decide(self, argv: list[str]) -> Decision:
        """Decide before anything is spawned."""
        if not argv:
            return Decision(False, "no command given")
        if self.mode == "off":
            return Decision(False, "the cli tool is off; start the worker with --tools cli")

        command = _normalise(argv)
        for pattern, reason in _COMPILED_DENY:
            if pattern.search(command):
                return Decision(False, f"refused ({reason})")

        program = os.path.basename(argv[0])
        escaped = self._escaping_argument(argv)
        if escaped is not None:
            return Decision(False, f"{escaped!r} is outside the sandbox at {self.root}")

        if self.mode == "allow" and program in self.allowlist:
            return Decision(True, f"{program} is allowlisted")

        if self.approve is None:
            listed = ", ".join(self.allowlist)
            return Decision(
                False,
                f"{program!r} is not allowlisted and no approver is attached; allowed: {listed}",
            )
        if self.approve(command, argv):
            return Decision(True, "approved")
        return Decision(False, f"{program!r} was not approved")

    def _escaping_argument(self, argv: list[str]) -> str | None:
        """The first path-shaped argument that leaves the sandbox, if any."""
        for argument in argv[1:]:
            if argument.startswith("-"):
                continue
            if os.sep not in argument and not argument.startswith("~") and ".." not in argument:
                continue
            try:
                self.resolve(argument)
            except ToolError:
                return argument
        return None

    def record(self, reason: str) -> None:
        self.denied.append(reason)


def _child_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}
    env.setdefault("PATH", os.defpath)
    return env


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n[truncated]", True


class ShellTool(Tool):
    """Run one command inside the sandbox."""

    name = "shell"
    description = "Run a shell command in the project directory and return its output."
    parameters = {"command": "string", "timeout": "number, optional"}

    def __init__(self, policy: ShellPolicy | None = None):
        self.policy = policy or ShellPolicy()

    def run(self, command: str = "", timeout: float | None = None, **_: Any) -> ToolResult:
        if not command.strip():
            return ToolResult.failure("shell needs a command")

        try:
            argv = lex(command)
        except ValueError as exc:
            return ToolResult.failure(f"could not parse {command!r}: {exc}")

        operators = [
            token
            for token in argv
            if token and (set(token) <= SHELL_OPERATORS or "`" in token or token == "$")
        ]
        if operators:
            return ToolResult.failure(
                f"{command!r} uses shell syntax ({' '.join(operators)}); commands run without a "
                "shell, so run one program at a time"
            )

        decision = self.policy.decide(argv)
        if not decision:
            self.policy.record(decision.reason)
            return ToolResult.failure(decision.reason, command=command, allowed=False)

        return self._spawn(command, argv, timeout)

    def _spawn(self, command: str, argv: list[str], timeout: float | None) -> ToolResult:
        limit = float(timeout or self.policy.timeout)
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, shell=False, scrubbed env
                argv,
                cwd=self.policy.root,
                env=_child_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            return ToolResult.failure(f"{argv[0]!r} was not found", command=command)
        except OSError as exc:
            return ToolResult.failure(f"could not run {command!r}: {exc}", command=command)

        try:
            stdout, stderr = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            # The whole group, not just the child: a killed shell that leaves
            # its children running is not a timeout, it is a leak.
            _kill_group(process)
            stdout, stderr = process.communicate()
            stdout, _ = _truncate(stdout or "", self.policy.max_output)
            return ToolResult.failure(
                f"{command!r} did not finish within {limit:g}s and was killed",
                command=command,
                stdout=stdout,
                timed_out=True,
            )

        stdout, cut_out = _truncate(stdout or "", self.policy.max_output)
        stderr, cut_err = _truncate(stderr or "", self.policy.max_output)
        code = process.returncode
        body = stdout if stdout else "(no output)"
        if stderr:
            body = f"{body}\n[stderr]\n{stderr}"

        result = ToolResult.success if code == 0 else ToolResult.failure
        return result(
            f"exit {code}\n{body}",
            command=command,
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            truncated=cut_out or cut_err,
        )


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


class ReadFileTool(Tool):
    """Read part of a file without spawning anything."""

    name = "read_file"
    description = "Read a file from the project directory, optionally a line range."
    parameters = {"path": "string", "start": "integer, 1-based", "lines": "integer, optional"}

    def __init__(self, policy: ShellPolicy | None = None):
        self.policy = policy or ShellPolicy()

    def run(
        self, path: str = "", start: int = 1, lines: int = DEFAULT_MAX_LINES, **_: Any
    ) -> ToolResult:
        if not path:
            return ToolResult.failure("read_file needs a path")
        resolved = self.policy.resolve(path)
        if not os.path.isfile(resolved):
            return ToolResult.failure(f"{path!r} is not a file")

        start = max(1, int(start))
        count = max(1, int(lines))
        with open(resolved, encoding="utf-8", errors="replace") as handle:
            selected = []
            for number, line in enumerate(handle, start=1):
                if number < start:
                    continue
                if len(selected) >= count:
                    break
                selected.append(f"{number}\t{line.rstrip()}")

        if not selected:
            return ToolResult.failure(f"{path!r} has no line {start}")
        body, truncated = _truncate("\n".join(selected), self.policy.max_output)
        return ToolResult.success(
            body, path=path, start=start, lines=len(selected), truncated=truncated
        )


class ListDirTool(Tool):
    """List a directory without spawning anything."""

    name = "list_dir"
    description = "List the entries of a directory in the project directory."
    parameters = {"path": "string, default '.'"}

    def __init__(self, policy: ShellPolicy | None = None):
        self.policy = policy or ShellPolicy()

    def run(self, path: str = ".", **_: Any) -> ToolResult:
        resolved = self.policy.resolve(path or ".")
        if not os.path.isdir(resolved):
            return ToolResult.failure(f"{path!r} is not a directory")

        entries = []
        for entry in sorted(os.scandir(resolved), key=lambda e: e.name):
            if entry.is_dir():
                entries.append({"name": entry.name, "type": "dir", "size": None})
            else:
                entries.append({"name": entry.name, "type": "file", "size": entry.stat().st_size})

        listing = "\n".join(
            f"{item['name']}/" if item["type"] == "dir" else f"{item['name']}  {item['size']}"
            for item in entries
        )
        body, truncated = _truncate(listing or "(empty)", self.policy.max_output)
        return ToolResult.success(body, path=path, entries=entries, truncated=truncated)


def cli_tools(policy: ShellPolicy | None = None, **kwargs: Any) -> list[Tool]:
    """All three tools sharing one policy, so one decision governs the lot."""
    policy = policy or ShellPolicy(**kwargs)
    return [ShellTool(policy), ReadFileTool(policy), ListDirTool(policy)]
