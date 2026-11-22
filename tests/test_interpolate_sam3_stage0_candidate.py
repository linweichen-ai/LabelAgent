import ast
from pathlib import Path


SOURCE = Path("tools/interpolate_sam3_stage0_candidate.py").read_text(encoding="utf-8")


def test_interpolation_tool_has_strict_boundaries():
    ast.parse(SOURCE)
    assert '"test24" in str(path).lower()' in SOURCE
    assert '"test24" in str(output).lower() or output.exists()' in SOURCE
    assert '"human_truth": False' in SOURCE
    assert '"not_for_deployment": True' in SOURCE
    assert '"training_return_allowed": False' in SOURCE


def test_interpolation_only_touches_stage0_allowlist():
    assert "if tuple(state) != EXPECTED_PARAMETER_NAMES" in SOURCE
    assert "for name in EXPECTED_PARAMETER_NAMES" in SOURCE
    assert "before + args.alpha * (after - before)" in SOURCE
    assert "0.0 < args.alpha < 1.0" in SOURCE
