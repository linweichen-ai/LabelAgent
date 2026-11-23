"""Run a bounded synthetic-only connectivity matrix for external expert VLMs."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.expert_vlm_smoke import run_smoke


SCHEMA_VERSION = "labelagent.external-vlm-synthetic-matrix.v1"
PROFILE_FIELDS = frozenset({"name", "provider", "url", "model", "key_env"})
FORBIDDEN_FIELDS = frozenset({
    "api_key", "key", "secret", "token", "authorization", "password",
})


def _load_profiles(value):
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError("profile file does not exist")
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"profiles"}:
        raise ValueError("matrix input must contain only profiles")
    profiles = value["profiles"]
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 8:
        raise ValueError("matrix requires 1-8 profiles")
    normalized, names = [], set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("each matrix profile must be an object")
        lowered = {str(key).casefold() for key in profile}
        if lowered & FORBIDDEN_FIELDS:
            raise ValueError("raw credentials are forbidden in matrix profiles")
        if set(profile) != PROFILE_FIELDS:
            raise ValueError("matrix profile fields are invalid")
        name = profile["name"]
        if (not isinstance(name, str) or not 1 <= len(name.strip()) <= 80
                or name.strip() in names):
            raise ValueError("matrix profile name is invalid or duplicated")
        joined = json.dumps(profile, ensure_ascii=False).casefold()
        if "test24" in joined:
            raise ValueError("test24 is evaluation-only and cannot enter connectivity matrix")
        names.add(name.strip())
        normalized.append({key: str(profile[key]).strip() for key in PROFILE_FIELDS})
    return sorted(normalized, key=lambda item: item["name"])


def run_matrix(profiles, *, execute=False, acknowledge_synthetic_upload=False,
               environ=None, smoke_fn=run_smoke):
    normalized = _load_profiles(profiles)
    environ = os.environ if environ is None else environ
    secrets = {str(environ.get(item["key_env"]) or "") for item in normalized}
    secrets.discard("")
    if execute and acknowledge_synthetic_upload is not True:
        raise ValueError("execute requires explicit synthetic-upload acknowledgement")
    results = []
    for profile in normalized:
        try:
            receipt = smoke_fn(
                provider=profile["provider"], url=profile["url"],
                model=profile["model"], key_env=profile["key_env"],
                execute=execute,
                acknowledge_synthetic_upload=acknowledge_synthetic_upload,
                environ=environ,
            )
            serialized = json.dumps(receipt, ensure_ascii=False)
            if any(secret in serialized for secret in secrets):
                raise ValueError("smoke receipt contained credential material")
            results.append({"name": profile["name"], "ok": True,
                            "receipt": receipt})
        except Exception as error:
            message = str(error)
            for secret in secrets:
                message = message.replace(secret, "[REDACTED]")
            results.append({"name": profile["name"], "ok": False,
                            "error_type": type(error).__name__,
                            "error": message[:300]})
    passed = sum(item["ok"] for item in results)
    tested = sum(bool(item.get("receipt", {}).get("connection_tested"))
                 for item in results)
    uploads = sum(bool(item.get("receipt", {}).get("external_upload_performed"))
                  for item in results)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "execute" if execute else "dry_run",
        "profile_count": len(results),
        "profile_manifest_sha256": hashlib.sha256(json.dumps(
            normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "successful_profiles": passed,
        "failed_profiles": len(results) - passed,
        "connections_tested": tested,
        "external_uploads_performed": uploads,
        "uploaded_content": ("generated_synthetic_images_only" if uploads
                             else "none"),
        "factory_data_uploaded": False,
        "test24_used": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "deployment_decision": "not_evaluated",
        "quality_improvement_claimed": False,
        "results": results,
    }
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate or run a synthetic-only external VLM matrix")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-synthetic-upload", action="store_true")
    args = parser.parse_args(argv)
    result = run_matrix(
        args.profiles, execute=args.execute,
        acknowledge_synthetic_upload=args.acknowledge_synthetic_upload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed_profiles"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
