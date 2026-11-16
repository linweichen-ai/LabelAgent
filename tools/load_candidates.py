"""Load a capture and import 2-D candidates through the workbench API."""
import argparse
import json
from pathlib import Path
import urllib.request


def load_candidates(path, candidates_file, base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(endpoint, value=None):
        payload = None if value is None else json.dumps(value).encode("utf-8")
        req = urllib.request.Request(base.rstrip("/") + endpoint, data=payload,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=180) as response:
            return json.load(response)

    path = str(Path(path).resolve())
    candidates = json.loads(Path(candidates_file).read_text(encoding="utf-8"))
    if request("/api/run/status")["status"] != "idle":
        raise ValueError("Workbench is running a task; import was not started")
    entry = request("/api/files", {"path": path})["file"]
    request(f"/api/files/{entry['id']}/load", {})
    result = request("/api/candidates/import", {"path": path, "candidates_by_topic": candidates})
    if result.get("saved_as_truth") is not False:
        raise ValueError("Import did not confirm unreviewed-candidate status")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="MCAP or image corresponding to the candidates")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    load_candidates(args.path, args.candidates, args.base)
