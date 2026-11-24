import argparse

import pytest

from tools.expert_vlm_resolution_matrix import parse_size, run_matrix


def _fake_smoke(**kwargs):
    width, height = kwargs["width"], kwargs["height"]
    tokens = width * height // 100
    return {
        "provider": "openai_chat",
        "status": "passed" if kwargs["execute"] else None,
        "latency_ms": tokens / 2 if kwargs["execute"] else None,
        "synthetic_image_sha256": f"{width:032x}{height:032x}",
        "factory_data_uploaded": False,
        "expert_metadata": {"external_vlm": {"usage": {
            "prompt_tokens": tokens - 2, "completion_tokens": 2,
            "total_tokens": tokens}}} if kwargs["execute"] else {},
    }


def test_execute_compares_usage_without_claiming_quality():
    result = run_matrix(
        provider="openai_chat", url="https://example.test/v1", model="expert",
        key_env="TEST_KEY", sizes=[(640, 512), (320, 256)], execute=True,
        acknowledge_synthetic_upload=True, environ={"TEST_KEY": "secret"},
        smoke_fn=_fake_smoke)
    assert result["summary"]["usage_complete"] is True
    assert result["provider"] == "openai_chat"
    assert result["summary"]["lowest_token_size"]["width"] == 320
    assert result["summary"]["quality_evaluated"] is False
    assert result["summary"]["deployment_recommendation"] is False
    assert result["factory_data_uploaded"] is False
    assert result["test24_used"] is False


def test_dry_run_never_claims_usage():
    result = run_matrix(
        provider="openai_chat", url="https://example.test/v1", model="expert",
        key_env="TEST_KEY", sizes=[(640, 512), (320, 256)],
        smoke_fn=_fake_smoke)
    assert result["mode"] == "dry_run"
    assert result["summary"]["usage_complete"] is False
    assert result["summary"]["lowest_token_size"] is None


def test_size_and_matrix_bounds_fail_closed():
    assert parse_size("320x256") == (320, 256)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size("bad")
    with pytest.raises(ValueError, match="2-6 unique"):
        run_matrix(
            provider="openai_chat", url="https://example.test/v1",
            model="expert", key_env="TEST_KEY", sizes=[(320, 256)],
            smoke_fn=_fake_smoke)
