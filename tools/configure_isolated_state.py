"""Apply one bounded model setting to an explicitly isolated state copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--llm-max-tokens", required=True, type=int)
    args = parser.parse_args()

    state_file = args.state_file.resolve()
    repository = Path(__file__).resolve().parents[1]
    artifacts = (repository / "artifacts").resolve()
    if not state_file.is_relative_to(artifacts):
        raise SystemExit("state file must be inside this repository's artifacts directory")
    if not state_file.is_file():
        raise SystemExit("state file does not exist")
    before = sha256(state_file)
    if before != args.expected_sha256.upper():
        raise SystemExit(f"state hash mismatch: {before}")
    if not 320 <= args.llm_max_tokens <= 1024:
        raise SystemExit("llm max tokens must be between 320 and 1024")

    state = json.loads(state_file.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise SystemExit("state does not contain a config object")
    previous = state["config"].get("llm_max_tokens")
    state["config"]["llm_max_tokens"] = args.llm_max_tokens

    temporary = state_file.with_name(state_file.name + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, state_file)
    print(json.dumps({
        "state_file": str(state_file),
        "before_sha256": before,
        "after_sha256": sha256(state_file),
        "previous_llm_max_tokens": previous,
        "llm_max_tokens": args.llm_max_tokens,
        "isolated_only": True,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
