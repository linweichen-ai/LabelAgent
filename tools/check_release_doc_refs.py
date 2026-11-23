"""Read-only validation of repository-local references in release documents."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/LabelAgent_同事完整复现教程_20261123.md",
    "docs/REPRODUCIBLE_SETUP.md",
)
REPOSITORY_ROOTS = ("tools", "docs", "source", "tests", "artifacts")
CHECKABLE_DIRECTORY_ROOTS = ("tools", "docs", "source", "tests")
RUNTIME_OR_EXTERNAL_PREFIXES = (
    ".git/",
    ".repro/",
    ".venv/",
    "work/",
    "source/sam3/weights/",
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
ROOTED_PATH_RE = re.compile(
    r"(?<![\w./:\\-])((?:tools|docs|source|tests|artifacts)[\\/]"
    r"[^\s`'\"<>()\[\]{}|,;:，。；：！？]+)"
)
EXTERNAL_MARKER_RE = re.compile(
    r"<!--\s*release-doc-ref:\s*external\s+([^>]+?)\s*-->", re.IGNORECASE
)


@dataclass(frozen=True)
class Reference:
    document: str
    line: int
    raw: str
    path: str
    status: str


def normalize_reference(raw: str) -> str:
    value = raw.strip().strip("'\"").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.rstrip("/.,;:，。；：！？")
    return str(PurePosixPath(value)) if value else ""


def is_external_or_runtime(value: str) -> bool:
    lowered = value.lower()
    if not value or lowered.startswith(("http://", "https://", "mailto:", "#")):
        return True
    if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:/", value):
        return True
    if any(token in value for token in ("$", "<", ">", "*", "{", "}")):
        return True
    normalized = lowered.rstrip("/") + "/"
    return any(normalized.startswith(prefix.lower()) for prefix in RUNTIME_OR_EXTERNAL_PREFIXES)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _candidate_references(text: str, document: str) -> Iterable[tuple[int, str, str]]:
    seen: set[tuple[int, str]] = set()

    for match in MARKDOWN_LINK_RE.finditer(text):
        raw = match.group(1).strip().split(maxsplit=1)[0]
        normalized = normalize_reference(raw)
        if is_external_or_runtime(normalized):
            continue
        key = (_line_number(text, match.start()), normalized)
        if normalized and key not in seen:
            seen.add(key)
            yield key[0], raw, normalized

    searchable: list[tuple[int, str]] = [(1, text)]
    searchable.extend(
        (_line_number(text, match.start()), match.group(1))
        for match in INLINE_CODE_RE.finditer(text)
    )
    for base_line, fragment in searchable:
        exact = normalize_reference(fragment)
        exact_root = exact.rstrip("/")
        if exact_root in CHECKABLE_DIRECTORY_ROOTS:
            key = (base_line, exact_root)
            if key not in seen:
                seen.add(key)
                yield base_line, fragment, exact_root
        for match in ROOTED_PATH_RE.finditer(fragment):
            raw = match.group(1)
            normalized = normalize_reference(raw)
            if is_external_or_runtime(normalized):
                continue
            line = base_line + fragment.count("\n", 0, match.start())
            key = (line, normalized)
            if normalized and key not in seen:
                seen.add(key)
                yield line, raw, normalized


def audit_release_documents(
    root: Path = ROOT,
    documents: Iterable[str] = DEFAULT_DOCUMENTS,
) -> dict:
    root = root.resolve()
    results: list[Reference] = []
    missing_documents: list[str] = []

    for relative_document in documents:
        document_path = root / relative_document
        if not document_path.is_file():
            missing_documents.append(relative_document)
            continue
        text = document_path.read_text(encoding="utf-8")
        external = {
            normalize_reference(match.group(1))
            for match in EXTERNAL_MARKER_RE.finditer(text)
        }
        for line, raw, normalized in _candidate_references(text, relative_document):
            if normalized in external:
                status = "external"
                resolved = normalized
            else:
                if not normalized.startswith(REPOSITORY_ROOTS):
                    resolved_path = document_path.parent / normalized
                else:
                    resolved_path = root / normalized
                resolved = normalized
                status = "present" if resolved_path.exists() else "missing"
            results.append(Reference(relative_document, line, raw, resolved, status))

    missing = [asdict(item) for item in results if item.status == "missing"]
    return {
        "schema_version": 1,
        "root": str(root),
        "documents": list(documents),
        "ready": not missing_documents and not missing,
        "missing_documents": missing_documents,
        "missing": missing,
        "external": [asdict(item) for item in results if item.status == "external"],
        "checked": [asdict(item) for item in results if item.status == "present"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check repository-local paths referenced by release documentation"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = audit_release_documents(args.root)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["missing"]:
            print(f"[MISSING] {item['document']}:{item['line']} -> {item['path']}")
        for item in report["external"]:
            print(f"[EXTERNAL] {item['document']}:{item['line']} -> {item['path']}")
        print(
            f"checked={len(report['checked'])} external={len(report['external'])} "
            f"missing={len(report['missing'])}"
        )
        print("READY" if report["ready"] else "NOT READY")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
