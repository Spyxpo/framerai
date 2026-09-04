"""
Test that benchmark scripts can be imported correctly when executed from repository root.

This test ensures that the sys.path.insert() logic in benchmark scripts works correctly
and prevents regression of import issues reported by reviewers.
"""

import subprocess
import sys


def test_distillation_benchmark_script_imports():
    """Test that distillation_image_gen.py can be executed and imports work."""
    # Test that the script can be executed with --help (minimal test)
    result = subprocess.run(
        [sys.executable, "benchmarks/distillation_image_gen.py", "--help"],
        capture_output=True,
        text=True,
        cwd=".",  # Ensure we're running from repo root
    )

    # Should exit successfully and show help
    assert result.returncode == 0
    assert "Benchmark distilled image generation" in result.stdout
    assert "--teacher" in result.stdout
    assert "--student" in result.stdout


def test_moe_throughput_script_imports():
    """Test that moe_throughput.py can be executed and imports work."""
    # Test that the script can be executed with --help (minimal test)
    result = subprocess.run(
        [sys.executable, "benchmarks/moe_throughput.py", "--help"],
        capture_output=True,
        text=True,
        cwd=".",  # Ensure we're running from repo root
    )

    # Should exit successfully and show help
    assert result.returncode == 0
    assert "Benchmark MoE throughput" in result.stdout
    assert "--preset" in result.stdout
    assert "--device" in result.stdout


def test_direct_import_from_benchmarks():
    """Test that we can directly import functions from benchmark scripts."""
    # This tests the import path setup without executing the scripts

    # Test distillation benchmark
    from benchmarks.distillation_image_gen import load_captions

    # These should work without errors
    captions = load_captions(caption_file=None, num_images=3)
    assert len(captions) == 3
    assert all(isinstance(c, str) for c in captions)

    # Test MoE benchmark - just import the function to verify it's accessible
    from benchmarks.moe_throughput import benchmark_moe_forward

    # Function should exist and be callable
    assert callable(benchmark_moe_forward)
