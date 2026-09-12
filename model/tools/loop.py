"""The tool-calling loop.

The model is shown the available tools, generates, and may emit one block:

    <tool_call>{"name": "web_search", "arguments": {"query": "rectified flow"}}</tool_call>

The loop runs that tool, appends the result as ``<tool_result>``, and generates
again, up to ``max_steps``. Text with no block ends the turn.

The loop takes a plain ``generate(prompt) -> str`` callable, so it is not tied
to :class:`~model.generate.FramerGenerator` and can be tested without a model.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .base import ToolRegistry, ToolResult

CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
RESULT_OPEN = "<tool_result>"
RESULT_CLOSE = "</tool_result>"

_CALL_RE = re.compile(re.escape(CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(CALL_CLOSE), re.DOTALL)

# A prompt opening with one of these was formatted by its author, not typed by
# a user, so it is passed through rather than wrapped in a turn of its own.
_PREFORMATTED = ("<system>", "<user>", "<assistant>", "<reasoning>", "<tool>", "<tool_call>")

INSTRUCTIONS = f"""You can use tools to find information you do not already have.

To use one, emit exactly one block and then stop:
{CALL_OPEN}{{"name": "TOOL", "arguments": {{"ARG": "VALUE"}}}}{CALL_CLOSE}

The result comes back as {RESULT_OPEN}...{RESULT_CLOSE}. Use it to answer.
When you can answer, reply in plain text with no block. Cite any URL you used.

Tools:
{{tools}}"""


class ToolCallError(Exception):
    """The model emitted something that looked like a call but was not one."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolStep:
    """One tool invocation and what came back."""

    name: str
    arguments: dict[str, Any]
    result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result.to_dict()}


@dataclass
class ToolTrace:
    """Every tool the turn ran, and why the loop stopped.

    Travels back with the reply for the same reason the cognition trace does:
    a caller should be able to show what was searched and read instead of
    guessing where an answer came from.
    """

    steps: list[ToolStep] = field(default_factory=list)
    stopped: str = "answered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "stopped": self.stopped,
            "used": sorted({step.name for step in self.steps}),
        }

    def context(self) -> str:
        """The successful results as plain text, for a caller that generates
        its own reply (the cognition layer) rather than continuing this loop."""
        lines = []
        for step in self.steps:
            if step.result.ok:
                lines.append(f"[{step.name}] {step.result.content}")
        return "\n\n".join(lines)


def render_prompt(registry: ToolRegistry, prompt: str | list[dict[str, Any]]) -> str:
    """The instruction block, the tool schemas, and the user's prompt."""
    from model.tokenizer.chat_template import ChatTemplate

    instructions = INSTRUCTIONS.replace("{tools}", registry.describe())
    if isinstance(prompt, list):
        has_system = False
        messages = []
        for msg in prompt:
            if msg.get("role") == "system" and not has_system:
                has_system = True
                content = msg.get("content", "")
                sys_content = f"{instructions}\n{content}".strip() if content else instructions
                messages.append({"role": "system", "content": sys_content})
            else:
                messages.append(msg)
        if not has_system:
            messages.insert(0, {"role": "system", "content": instructions})
        return ChatTemplate(version="v1").format_messages(messages, add_generation_prompt=True)

    if prompt.startswith("<system>"):
        return prompt.replace("<system>", f"<system>{instructions}\n", 1)
    if prompt.startswith(_PREFORMATTED):
        return f"<system>{instructions}{prompt}"

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    return ChatTemplate(version="v1").format_messages(messages, add_generation_prompt=True)


def _instructed_messages(registry: ToolRegistry, prompt: str | list[dict[str, Any]]) -> list[dict]:
    """The conversation render_prompt would format, as messages."""
    instructions = INSTRUCTIONS.replace("{tools}", registry.describe())
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]

    has_system = False
    messages = []
    for msg in prompt:
        if msg.get("role") == "system" and not has_system:
            has_system = True
            content = msg.get("content", "")
            sys_content = f"{instructions}\n{content}".strip() if content else instructions
            messages.append({"role": "system", "content": sys_content})
        else:
            messages.append(msg)
    if not has_system:
        messages.insert(0, {"role": "system", "content": instructions})
    return messages


def render_prompt_ids(
    registry: ToolRegistry, prompt: str | list[dict[str, Any]], tokenizer
) -> list[int]:
    """What render_prompt renders, encoded so the roles survive as control ids.

    Encoding the rendered string instead would have to allow the role markers
    back wholesale, and a question that merely mentions ``<assistant>`` would
    open a turn. Here the template encodes turn by turn, so only the markers it
    wrote become ids.

    A prompt that arrives already formatted is the exception: its author placed
    those markers, so they are honoured as written.
    """
    from model.tokenizer.chat_template import ChatTemplate

    template = ChatTemplate(version="v1")
    if isinstance(prompt, str) and prompt.startswith(_PREFORMATTED):
        return tokenizer.encode(
            render_prompt(registry, prompt),
            add_special=False,
            allowed_special=template.allowed_special,
        )
    return template.encode_prompt(
        _instructed_messages(registry, prompt), tokenizer, add_generation_prompt=True
    )


def parse_tool_call(text: str) -> ToolCall | None:
    """Pull the first tool call out of generated text.

    Returns None when there is no call. Raises :class:`ToolCallError` when there
    is something call-shaped that will not parse, so the caller can feed the
    reason back rather than silently treating a broken call as an answer.
    """
    match = _CALL_RE.search(text)
    if match is None:
        if CALL_OPEN in text:
            raise ToolCallError(f"unterminated {CALL_OPEN} block; it needs {CALL_CLOSE}")
        return None

    body = match.group(1)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"the tool call is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolCallError("the tool call must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ToolCallError("the tool call needs a string 'name'")
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolCallError("'arguments' must be a JSON object")
    return ToolCall(name=name, arguments=arguments)


def _continuation(text: str, prompt: str) -> str:
    """The generated text alone; the generator echoes the prompt back."""
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text


def _format_result(name: str, result: ToolResult) -> str:
    status = "ok" if result.ok else "error"
    return f"{RESULT_OPEN}{name} ({status}): {result.content}{RESULT_CLOSE}"


def _extend(transcript: str, transcript_ids, reply: str, rendered: str, tokenizer):
    """Append a reply and a tool result to the transcript, in both forms.

    What is appended is model output and tool output, neither of which this
    codebase wrote, so nothing in it is allowed to become a control id.
    """
    chunk = f"\n{reply}\n{rendered}\n"
    if transcript_ids is not None:
        transcript_ids = transcript_ids + tokenizer.encode(chunk, add_special=False)
    return transcript + chunk, transcript_ids


def run_tool_loop(
    generate: Callable[[str | list[int]], str],
    registry: ToolRegistry,
    prompt: str | list[dict[str, Any]],
    max_steps: int = 4,
    tokenizer=None,
) -> tuple[str, ToolTrace]:
    """Generate with tools available, returning ``(reply, trace)``.

    With an empty registry this is one plain generation, so a caller can hand
    the loop an empty registry instead of branching around it.

    Given a ``tokenizer``, the transcript is encoded here and ``generate`` is
    called with ids, which is what lets the role markers reach the model as
    boundaries while a marker inside a question, a reply, or a fetched page
    stays as the characters it is. Without one the transcript is passed as
    text and every marker in it is literal.
    """
    trace = ToolTrace()
    if not len(registry):
        trace.stopped = "no_tools"
        from model.tokenizer.chat_template import ChatTemplate

        template = ChatTemplate(version="v1")
        if isinstance(prompt, list):
            prompt_str = template.format_messages(prompt, add_generation_prompt=True)
            sent = (
                template.encode_prompt(prompt, tokenizer, add_generation_prompt=True)
                if tokenizer is not None
                else prompt_str
            )
        else:
            prompt_str = prompt
            sent = tokenizer.encode(prompt, add_special=False) if tokenizer is not None else prompt
        return _continuation(generate(sent), prompt_str).strip(), trace

    transcript = render_prompt(registry, prompt)
    transcript_ids = (
        render_prompt_ids(registry, prompt, tokenizer) if tokenizer is not None else None
    )
    reply = ""

    for _ in range(max(1, max_steps)):
        sent = transcript if transcript_ids is None else transcript_ids
        reply = _continuation(generate(sent), transcript).strip()

        try:
            call = parse_tool_call(reply)
        except ToolCallError as exc:
            # A malformed call is fed back as an error result: the model gets a
            # chance to fix its own syntax, which is cheaper than failing here.
            result = ToolResult.failure(str(exc))
            trace.steps.append(ToolStep(name="(malformed)", arguments={}, result=result))
            transcript, transcript_ids = _extend(
                transcript, transcript_ids, reply, _format_result("tool_call", result), tokenizer
            )
            continue

        if call is None:
            trace.stopped = "answered"
            return reply, trace

        result = registry.run(call.name, call.arguments)
        trace.steps.append(ToolStep(name=call.name, arguments=call.arguments, result=result))
        transcript, transcript_ids = _extend(
            transcript, transcript_ids, reply, _format_result(call.name, result), tokenizer
        )

    trace.stopped = "max_steps"
    # The last reply is still a call, so strip it: a caller wants prose, and the
    # results are in the trace either way.
    return _CALL_RE.sub("", reply).strip(), trace
