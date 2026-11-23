from pathlib import Path

from tools.browser_candidate_comparison_acceptance import validate_url


def test_candidate_comparison_browser_acceptance_is_loopback_only():
    assert validate_url("http://127.0.0.1:8877") == "http://127.0.0.1:8877/"
    for value in (
            "https://127.0.0.1:8877", "http://localhost:8877",
            "http://127.0.0.1:8877/not-root",
            "http://127.0.0.1:8877/?token=x", "http://user@127.0.0.1:8877"):
        try:
            validate_url(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)


def test_acceptance_source_records_non_get_and_protected_writes():
    source = Path("tools/browser_candidate_comparison_acceptance.py").read_text(
        encoding="utf-8")
    assert 'request.method != "GET"' in source
    assert '"/api/truth"' in source
    assert "sessionStorage.getItem('labelagentCandidateComparisonPreferencesV1')" in source
    assert "empty_refresh_failed_closed" in source
