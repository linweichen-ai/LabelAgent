#!/usr/bin/env python3
"""Validate one unchanged full closed-loop bundle; never project a core subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import canonical_json


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_bundle_file(bundle_path, schema_path):
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as error:
        raise RuntimeError(
            "install tools/requirements-schema-validation.txt in a project environment"
        ) from error

    bundle_path, schema_path = Path(bundle_path), Path(schema_path)
    original = bundle_path.read_bytes()
    original_sha = sha256(original)
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes)
    bundle = json.loads(original)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(bundle),
                    key=lambda item: [str(part) for part in item.absolute_path])

    declared = bundle.get("bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256", None)
    calculated = sha256(canonical_json(unsigned))
    hash_valid = isinstance(declared, str) and declared == calculated
    unchanged = bundle_path.read_bytes() == original
    result = {"schema_path": str(schema_path.resolve()),
              "schema_file_sha256": sha256(schema_bytes),
              "bundle_path": str(bundle_path.resolve()),
              "bundle_file_sha256_before": original_sha,
              "bundle_file_sha256_after": sha256(bundle_path.read_bytes()),
              "bundle_file_unchanged": unchanged,
              "bundle_sha256_declared": declared,
              "bundle_sha256_calculated": calculated,
              "bundle_sha256_valid": hash_valid,
              "full_root_valid": not errors,
              "errors": [{"path": "$." + ".".join(map(str, error.absolute_path)),
                          "message": error.message} for error in errors]}
    result["valid"] = result["full_root_valid"] and hash_valid and unchanged
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--schema", default=str(
        ROOT / "schemas" / "labelagent-closed-loop.schema.json"))
    args = parser.parse_args()
    result = validate_bundle_file(args.bundle, args.schema)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
