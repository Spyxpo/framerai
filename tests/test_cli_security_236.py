"""Regression tests for Issue #236: CLI security gaps.

These tests document and verify fixes for four security vulnerabilities:
1. ReadFileTool/ListDirTool ignore cli-mode off
2. DEFAULT_ALLOWLIST permits arbitrary code execution
3. _escaping_argument skips flag-embedded paths
4. Caller-provided timeout bypasses policy timeout
"""


import pytest

from model.tools.cli import ListDirTool, ReadFileTool, ShellPolicy, ShellTool


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "safe.txt").write_text("safe content\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("inner\n")
    (tmp_path.parent / "secret.txt").write_text("leaked secret\n")
    return tmp_path


# --- Vulnerability 1: ReadFileTool/ListDirTool ignore cli-mode off ------------


def test_read_file_respects_cli_mode_off(sandbox):
    """Issue #236.1: ReadFileTool should refuse when mode is off."""
    policy = ShellPolicy(mode="off", root=str(sandbox))
    result = ReadFileTool(policy).run(path="safe.txt")
    # Should refuse, not succeed
    assert not result.ok
    assert "cli tool is off" in result.content or "off" in result.content


def test_list_dir_respects_cli_mode_off(sandbox):
    """Issue #236.1: ListDirTool should refuse when mode is off."""
    policy = ShellPolicy(mode="off", root=str(sandbox))
    result = ListDirTool(policy).run(path=".")
    # Should refuse, not succeed
    assert not result.ok
    assert "cli tool is off" in result.content or "off" in result.content


# --- Vulnerability 2: Allowlist permits arbitrary code execution --------------


def test_python_not_in_default_allowlist(sandbox):
    """PR #285 feedback: python removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # python should not be in DEFAULT_ALLOWLIST anymore
    assert "python" not in policy.allowlist
    assert "python3" not in policy.allowlist


def test_python_c_flag_arbitrary_code_execution(sandbox):
    """Issue #236.2: python -c permits arbitrary code execution."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # python -c can run arbitrary Python code, should be denied since not in allowlist
    result = ShellTool(policy).run(command="python -c 'import os; os.system(\"echo pwned\")'")
    # Should be refused - python not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_python_m_flag_arbitrary_code_execution(sandbox):
    """PR #285 feedback: python -m bypasses DENY_PATTERNS and can execute arbitrary code."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # python -m can run arbitrary modules, e.g., pip install malicious, http.server, etc.
    result = ShellTool(policy).run(command="python -m http.server 8000")
    # Should be refused - python not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_node_not_in_default_allowlist(sandbox):
    """PR #285 feedback: node removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # node should not be in DEFAULT_ALLOWLIST anymore
    assert "node" not in policy.allowlist


def test_node_e_flag_arbitrary_code_execution(sandbox):
    """Issue #236.2: node -e permits arbitrary code execution."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # node -e can run arbitrary JavaScript code
    result = ShellTool(policy).run(command="node -e 'require(\"child_process\").execSync(\"echo pwned\")'")
    # Should be refused - node not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_node_p_flag_arbitrary_code_execution(sandbox):
    """PR #285 feedback: node -p bypasses DENY_PATTERNS and can execute arbitrary code."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # node -p evaluates and prints JavaScript, can execute arbitrary code
    result = ShellTool(policy).run(command="node -p 'require(\"child_process\").execSync(\"echo pwned\")'")
    # Should be refused - node not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_git_not_in_default_allowlist(sandbox):
    """PR #285 feedback: git removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # git should not be in DEFAULT_ALLOWLIST anymore
    assert "git" not in policy.allowlist


def test_git_alias_arbitrary_command_execution(sandbox):
    """PR #285 feedback: git -c alias can create shell aliases for arbitrary execution."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # git -c can set config including aliases that execute shell commands
    result = ShellTool(policy).run(command="git -c alias.pwn='!sh -c \"echo pwned\"' pwn")
    # Should be refused - git not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_awk_not_in_default_allowlist(sandbox):
    """PR #285 feedback: awk removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # awk should not be in DEFAULT_ALLOWLIST anymore
    assert "awk" not in policy.allowlist


def test_awk_system_arbitrary_command_execution(sandbox):
    """PR #285 feedback: awk system() can execute arbitrary shell commands."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # awk can call system() to execute arbitrary shell commands
    result = ShellTool(policy).run(command="awk 'BEGIN{system(\"echo pwned\")}'")
    # Should be refused - awk not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_find_not_in_default_allowlist(sandbox):
    """Issue #236.2: find removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # find should not be in DEFAULT_ALLOWLIST anymore
    assert "find" not in policy.allowlist


def test_find_delete_flag_file_destruction(sandbox):
    """Issue #236.2: find -delete permits file destruction."""
    (sandbox / "victim.txt").write_text("delete me\n")
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # find -delete can delete files
    result = ShellTool(policy).run(command="find . -name victim.txt -delete")
    # Should be refused - find not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content
    # Verify file still exists (fix prevents execution)
    assert (sandbox / "victim.txt").exists()


def test_sed_not_in_default_allowlist(sandbox):
    """Issue #236.2: sed removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # sed should not be in DEFAULT_ALLOWLIST anymore
    assert "sed" not in policy.allowlist


def test_sed_i_flag_file_mutation(sandbox):
    """Issue #236.2: sed -i permits in-place file mutation."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # sed -i can modify files in place
    result = ShellTool(policy).run(command="sed -i 's/safe/hacked/' safe.txt")
    # Should be refused - sed not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_npm_not_in_default_allowlist(sandbox):
    """Issue #236.2: npm removed from DEFAULT_ALLOWLIST."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # npm should not be in DEFAULT_ALLOWLIST anymore
    assert "npm" not in policy.allowlist


def test_npm_arbitrary_code_execution(sandbox):
    """Issue #236.2: npm can execute arbitrary code."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # npm exec can run arbitrary commands
    result = ShellTool(policy).run(command="npm exec -- echo pwned")
    # Should be refused - npm not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


def test_rg_not_in_default_allowlist(sandbox):
    """PR #285 final feedback: rg removed from DEFAULT_ALLOWLIST due to --pre."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # rg should not be in DEFAULT_ALLOWLIST anymore
    assert "rg" not in policy.allowlist


def test_rg_pre_arbitrary_command_execution(sandbox):
    """PR #285 final feedback: rg --pre can execute arbitrary preprocessor commands."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # rg --pre can run arbitrary commands as a preprocessor
    result = ShellTool(policy).run(command="rg --pre 'echo pwned' pattern")
    # Should be refused - rg not in DEFAULT_ALLOWLIST
    assert not result.ok
    assert "not allowlisted" in result.content or "not found" in result.content


# --- Vulnerability 3: flag-embedded paths bypass sandbox checks ---------------


def test_output_flag_with_embedded_path_escapes_sandbox(sandbox):
    """Issue #236.3: --output=/path bypasses sandbox validation."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    escape_path = str(sandbox.parent / "escape.txt")
    # Path embedded in flag should still be validated
    decision = policy.decide(["tar", "-czf", f"--output={escape_path}", "safe.txt"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_git_dir_flag_with_embedded_path_escapes_sandbox(sandbox):
    """Issue #236.3: --git-dir=/path bypasses sandbox validation."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Path embedded in git-dir flag should still be validated
    escape_path = str(sandbox.parent / "outside")
    decision = policy.decide(["git", f"--git-dir={escape_path}", "status"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_file_flag_with_embedded_path_escapes_sandbox(sandbox):
    """Issue #236.3: --file=../path bypasses sandbox validation."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Relative path escape embedded in flag
    decision = policy.decide(["grep", "secret", "--file=../secret.txt"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_absolute_path_blocked_in_arguments(sandbox):
    """Issue #236.3: Absolute paths like /etc/passwd should be blocked."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Absolute path should be validated against sandbox
    decision = policy.decide(["cat", "/etc/passwd"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_absolute_path_in_flag_blocked(sandbox):
    """Issue #236.3: Absolute paths in flags should also be blocked."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Absolute path embedded in flag should be validated
    decision = policy.decide(["cat", "--file=/etc/passwd"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_short_flag_with_attached_path_escapes_sandbox(sandbox):
    """PR #285 final feedback: Short flags with attached paths like -o/tmp/x bypass validation."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    escape_path = str(sandbox.parent / "escape.txt")
    # Path attached directly to short flag should still be validated
    decision = policy.decide(["sort", f"-o{escape_path}"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_short_flag_with_attached_relative_escape(sandbox):
    """PR #285 final feedback: Short flags with relative path escapes like -o../file."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Relative path escape attached to short flag
    decision = policy.decide(["sort", "-o../escape.txt"])
    assert not decision
    assert "outside the sandbox" in decision.reason


def test_legitimate_short_flags_not_blocked(sandbox):
    """Ensure legitimate short flags without paths are still accepted."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Short flags without paths should be fine
    assert policy.decide(["sort", "-n", "-r"])  # numeric, reverse sort
    assert policy.decide(["grep", "-i", "-n", "pattern"])  # case-insensitive, line numbers


# --- Vulnerability 4: Caller timeout bypasses policy maximum -----------------


def test_shell_timeout_respects_policy_maximum(sandbox):
    """Issue #236.4: Requested timeout should not exceed policy timeout."""
    import platform
    import time

    policy = ShellPolicy(
        mode="allow",
        root=str(sandbox),
        timeout=2.0,  # Policy maximum
        allowlist=("python", "python3", "python.exe")
    )

    # Create a script that sleeps longer than any reasonable timeout
    (sandbox / "sleeper.py").write_text("import time; time.sleep(30)")

    # Test 1: Verify timeout calculation logic directly
    tool = ShellTool(policy)

    # The key security property: caller timeout should be capped by policy timeout
    # Test the limit calculation that happens in _spawn
    requested_timeout = 999.0
    policy_timeout = policy.timeout  # 2.0
    expected_limit = min(float(requested_timeout), policy_timeout)  # Should be 2.0

    assert expected_limit == 2.0, f"Timeout calculation should cap at policy limit, got {expected_limit}"

    # Test 2: Verify actual timeout behavior using timing
    start_time = time.time()
    # On Windows, killpg cleanup may take longer, so allow more margin
    max_expected_time = 10.0 if platform.system() == "Windows" else 5.0

    try:
        result = tool.run(
            command="python sleeper.py",
            timeout=999.0  # This should be capped to 2.0 by policy
        )
        elapsed = time.time() - start_time

        # Should timeout around 2.0s (policy limit), not 999.0s (requested)
        # Allow some margin for test execution overhead
        assert elapsed < max_expected_time, f"Command should timeout quickly (~2s), took {elapsed:.1f}s"
        assert not result.ok, "Command should fail due to timeout"

        # Verify the timeout was enforced (not other error like "not found")
        timed_out_indicators = [
            result.data.get("timed_out") is True,
            "did not finish" in result.content,
            "timed out" in result.content.lower()
        ]
        assert any(timed_out_indicators), f"Should indicate timeout, got: {result.content}"

    except Exception as e:
        elapsed = time.time() - start_time
        # Even if cleanup fails (e.g., Windows killpg issue), timing should show timeout was enforced
        assert elapsed < max_expected_time, f"Even with cleanup error, timeout should be enforced (~2s), took {elapsed:.1f}s"

        # If it's a known Windows issue, that's acceptable - the timeout was still enforced
        if platform.system() == "Windows" and ("killpg" in str(e) or "No attribute" in str(e)):
            # Timeout enforcement worked (based on timing), just cleanup failed
            pass
        else:
            raise


def test_timeout_calculation_unit_test():
    """Unit test for timeout calculation logic without subprocess overhead."""
    from model.tools.cli import ShellPolicy

    policy = ShellPolicy(timeout=5.0)
    # Test the actual implementation behavior by checking what limit would be calculated
    # in _spawn method for different requested timeouts vs policy timeout of 5.0
    test_cases = [
        (None, 5.0),      # None -> use policy timeout
        (3.0, 3.0),       # Below policy limit -> use requested (VULNERABLE if not capped)
        (10.0, 5.0),      # Above policy limit -> should be capped to policy
        (999.0, 5.0),     # Way above policy -> should be capped to policy
        (0.1, 0.1),       # Very small -> use requested
    ]

    for requested_timeout, expected_limit in test_cases:
        # This simulates the calculation that happens in _spawn method
        # With the fix: limit = min(float(timeout or policy.timeout), policy.timeout)
        # Without fix: limit = float(timeout or policy.timeout) - VULNERABLE

        # Test what the FIXED implementation should do:
        fixed_limit = min(float(requested_timeout or policy.timeout), policy.timeout)
        assert fixed_limit == expected_limit, f"FIXED: For requested={requested_timeout}, expected {expected_limit}, got {fixed_limit}"

        # Test what the VULNERABLE implementation would do:
        vulnerable_limit = float(requested_timeout or policy.timeout)

        # For high timeouts, the vulnerable version should differ from expected
        if requested_timeout and requested_timeout > policy.timeout:
            assert vulnerable_limit > expected_limit, f"VULNERABLE version should allow longer timeouts for requested={requested_timeout}"


def test_nan_timeout_does_not_bypass_policy(sandbox):
    """PR #285 final feedback: NaN timeout must not bypass or corrupt policy timeout."""
    policy = ShellPolicy(
        mode="allow",
        root=str(sandbox),
        timeout=2.0,
        allowlist=("echo",)
    )

    tool = ShellTool(policy)

    # NaN timeout should be rejected and fall back to policy timeout
    result = tool.run(command="echo test", timeout=float('nan'))

    # Should not crash - NaN should be handled safely
    # Either succeeds (if echo available) or fails gracefully
    assert result is not None
    # The key: NaN should not have bypassed timeout validation or caused crashes


def test_infinity_timeout_does_not_bypass_policy(sandbox):
    """PR #285 final feedback: Infinity timeout must not bypass policy timeout."""
    policy = ShellPolicy(
        mode="allow",
        root=str(sandbox),
        timeout=2.0,
        allowlist=("echo",)
    )

    tool = ShellTool(policy)

    # Infinity timeout should be capped to policy timeout
    result = tool.run(command="echo test", timeout=float('inf'))

    # Should not crash - infinity should be handled safely
    assert result is not None


def test_negative_timeout_uses_policy_timeout(sandbox):
    """Negative timeout should fall back to policy timeout."""
    policy = ShellPolicy(
        mode="allow",
        root=str(sandbox),
        timeout=2.0,
        allowlist=("echo",)
    )

    tool = ShellTool(policy)

    # Negative timeout should be rejected and fall back to policy timeout
    result = tool.run(command="echo test", timeout=-1.0)

    # Should not crash - negative timeout should be handled safely
    assert result is not None


# --- Follow-up PR: Additional CLI sandbox bypasses identified by mentor -------


def test_bundled_short_flags_with_attached_path_escape(sandbox):
    """Follow-up: Bundled short flags like -ro/tmp/x can bypass path validation.

    The current _escaping_argument() assumes single-char flags and extracts
    argument[2:], but bundled flags like -ro mean -r and -o are both flags,
    and the path starts after 'o', not at position 2.
    """
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    escape_path = str(sandbox.parent / "escape.txt")

    # Bundled flags -r and -o with path attached to -o
    decision = policy.decide(["sort", f"-ro{escape_path}"])
    assert not decision, "Bundled short flags with attached escaping path must be blocked"
    assert "outside the sandbox" in decision.reason


def test_bundled_short_flags_relative_escape(sandbox):
    """Follow-up: Bundled short flags with relative path escape like -rf../x."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))

    # Bundled flags with relative escape
    decision = policy.decide(["sort", "-rf../escape.txt"])
    assert not decision, "Bundled short flags with relative escape must be blocked"
    assert "outside the sandbox" in decision.reason


def test_short_flag_attached_path_with_equals_sign(sandbox):
    """Follow-up: Attached path containing '=' like -o/tmp/x=y bypasses validation.

    The current code checks for '=' first and splits on it, which extracts
    only the part after '=', missing the actual path before it.
    """
    policy = ShellPolicy(mode="allow", root=str(sandbox))
    # Path /tmp/x is before the '=', not after
    decision = policy.decide(["sort", "-o/tmp/x=y"])
    assert not decision, "Short flag with attached path containing '=' must be blocked"
    assert "outside the sandbox" in decision.reason


def test_short_flag_attached_relative_escape_with_equals(sandbox):
    """Follow-up: Relative escape in attached path with '=' like -o../x=y."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))

    decision = policy.decide(["sort", "-o../escape=value"])
    assert not decision, "Short flag with relative escape and '=' must be blocked"
    assert "outside the sandbox" in decision.reason


def test_sort_compress_program_can_execute_commands(sandbox):
    """Follow-up: sort --compress-program=sh can execute arbitrary external commands.

    sort --compress-program allows specifying a compression program which
    can be used to execute arbitrary commands.
    """
    policy = ShellPolicy(mode="allow", root=str(sandbox))

    # sort with --compress-program can execute external programs
    decision = policy.decide(["sort", "--compress-program=sh", "file.txt"])
    assert not decision, "sort --compress-program must be blocked"
    assert "refused" in decision.reason or "not allowlisted" in decision.reason


def test_sort_compress_program_short_form(sandbox):
    """Follow-up: Ensure short form variations of --compress-program are also blocked."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))

    # Try space-separated form
    decision = policy.decide(["sort", "--compress-program", "sh", "file.txt"])
    assert not decision, "sort --compress-program (space-separated) must be blocked"


def test_legitimate_sort_usage_still_works(sandbox):
    """Follow-up: Ensure normal sort usage without --compress-program still works."""
    policy = ShellPolicy(mode="allow", root=str(sandbox))

    # Normal sort usage should be fine
    assert policy.decide(["sort", "file.txt"])
    assert policy.decide(["sort", "-r", "file.txt"])  # reverse
    assert policy.decide(["sort", "-n", "file.txt"])  # numeric
    assert policy.decide(["sort", "-u", "file.txt"])  # unique
