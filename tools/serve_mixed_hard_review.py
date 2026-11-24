"""Validate and serve a mixed-hard human review workspace on loopback only."""
import argparse
import hashlib
import ipaddress
import json
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


RECEIPT_SCHEMA = "labelagent.mixed-hard-adjudication-workspace-receipt.v2"
TEMPLATE_SCHEMA = "labelagent.mixed-hard-adjudication-template.v2"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review workspace JSON: {Path(path).name}") from error


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("review receipt contains an invalid file path")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("review receipt contains an unsafe file path")
    return candidate.as_posix()


def _inside(root, path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_workspace(workspace):
    """Fail closed unless every declared review file and safety flag is valid."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("review workspace does not exist")
    receipt_path = root / "receipt.json"
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported review workspace receipt schema")
    required_false = (
        "human_truth", "approved_for_training", "automatic_approval")
    if receipt.get("candidate_only") is not True or any(
            receipt.get(key) is not False for key in required_false):
        raise ValueError("review workspace safety flags are invalid")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("review workspace receipt has no files")
    manifest_sha = receipt.get("review_manifest_sha256")
    if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
            or any(character not in "0123456789abcdef"
                   for character in manifest_sha.lower())):
        raise ValueError("review manifest SHA-256 is invalid")
    verified = {}
    for raw_name, expected in files.items():
        name = _safe_relative_path(raw_name)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef"
                       for character in expected.lower())):
            raise ValueError(f"invalid SHA-256 for {name}")
        path = (root / Path(*PurePosixPath(name).parts)).resolve()
        if not _inside(root, path) or not path.is_file():
            raise ValueError(f"review workspace file is missing: {name}")
        actual = file_sha256(path)
        if actual != expected.lower():
            raise ValueError(f"review workspace hash mismatch: {name}")
        verified[name] = actual
    for name in ("reviewer.html", "adjudication-template.json"):
        if name not in verified:
            raise ValueError(f"review workspace receipt omits {name}")
    template = _read_json(root / "adjudication-template.json")
    rows = template.get("rows")
    sample_count = template.get("sample_count")
    if (template.get("schema_version") != TEMPLATE_SCHEMA
            or not isinstance(rows, list)
            or sample_count != len(rows)
            or not 24 <= sample_count <= 40):
        raise ValueError("review template is not a bounded 24-40 row set")
    if (template.get("candidate_only") is not True
            or template.get("human_truth") is not False
            or template.get("approved_for_training") is not False
            or template.get("training_authorized") is not False
            or template.get("automatic_approval") is not False):
        raise ValueError("review template safety flags are invalid")
    sample_ids = [row.get("sample_id") for row in rows
                  if isinstance(row, dict)]
    if (len(sample_ids) != sample_count or len(set(sample_ids)) != sample_count
            or any(not value for value in sample_ids)):
        raise ValueError("review template sample identities are invalid")
    return {
        "status": "ready",
        "workspace": str(root),
        "sample_count": sample_count,
        "verified_file_count": len(verified),
        "review_manifest_sha256": manifest_sha.lower(),
        "reviewer_sha256": verified["reviewer.html"],
        "allowed_paths": frozenset(verified),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "external_call_performed": False,
    }


def _loopback_host(host):
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.version == 4 and address.is_loopback
    except ValueError:
        return False


class ReviewHandler(SimpleHTTPRequestHandler):
    """Serve only hash-declared files; never expose the parent checkout."""

    allowed_paths = frozenset()

    def _requested_path(self):
        path = unquote(urlsplit(self.path).path).lstrip("/")
        return path or "reviewer.html"

    def _allowed(self):
        return self._requested_path() in self.allowed_paths

    def do_GET(self):
        if not self._allowed():
            self.send_error(404)
            return
        self.path = "/" + self._requested_path()
        super().do_GET()

    def do_HEAD(self):
        if not self._allowed():
            self.send_error(404)
            return
        self.path = "/" + self._requested_path()
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self' 'unsafe-inline' data:; "
                         "img-src 'self' data:; connect-src 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, _format, *_args):
        return


def build_server(workspace, *, host="127.0.0.1", port=0):
    if not _loopback_host(host):
        raise ValueError("review server must bind to a loopback address")
    summary = validate_workspace(workspace)
    handler = type("BoundReviewHandler", (ReviewHandler,), {
        "allowed_paths": summary["allowed_paths"],
    })
    server = ThreadingHTTPServer(
        (host, port), partial(handler, directory=summary["workspace"]))
    server.daemon_threads = True
    return server, summary


def public_summary(summary, *, url=None):
    return {key: value for key, value in summary.items()
            if key != "allowed_paths"} | ({"url": url} if url else {})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_only:
            result = validate_workspace(args.workspace)
            print(json.dumps(public_summary(result), ensure_ascii=False, indent=2))
            return 0
        server, result = build_server(
            args.workspace, host=args.host, port=args.port)
        bound_host, bound_port = server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
        url = f"http://{display_host}:{bound_port}/reviewer.html"
        print(json.dumps(public_summary(result, url=url),
                         ensure_ascii=False, indent=2), flush=True)
        if not args.no_browser:
            threading.Timer(0.15, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "refused", "error": str(error),
                          "candidate_only": True, "human_truth": False,
                          "approved_for_training": False},
                         ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
