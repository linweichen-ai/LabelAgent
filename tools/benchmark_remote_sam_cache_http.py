"""Read-only live acceptance for the remote SAM exact-response cache."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import time
import urllib.request


def get_bytes(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def post(url, body):
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def stable_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def summary(result, seconds):
    cache = (result.get("diagnostics") or {}).get("response_cache") or {}
    core = result.get("prompts") if isinstance(result.get("prompts"), dict) \
        else result.get("results")
    counts = ({label: len(value.get("results") or [])
               for label, value in result["prompts"].items()}
              if isinstance(result.get("prompts"), dict)
              else {"single": len(result.get("results") or [])})
    return {
        "wall_seconds": round(seconds, 6),
        "cache_hit": cache.get("hit"),
        "coalesced": cache.get("coalesced"),
        "coalesced_wait_seconds": cache.get("coalesced_wait_seconds"),
        "cache_entries": cache.get("entries"),
        "cache_bytes": cache.get("bytes"),
        "core_sha256": stable_hash(core),
        "result_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-url", default="http://127.0.0.1:28766")
    parser.add_argument("--image-url", default="http://127.0.0.1:8765/api/frame/148")
    args = parser.parse_args()
    image = get_bytes(args.image_url)
    encoded = base64.b64encode(image).decode("ascii")
    endpoint = args.sam_url.rstrip("/") + "/segment_text_multi"

    exact = {"image": encoded,
             "prompts": {"floor": "floor", "slope": "traversable surface"}}
    first_result, first_seconds = post(endpoint, exact)
    second_result, second_seconds = post(endpoint, exact)

    changed = {"image": encoded,
               "prompts": {"floor": "warehouse floor",
                           "slope": "traversable surface"}}
    changed_result, changed_seconds = post(endpoint, changed)

    concurrent_body = {
        "image": encoded,
        "prompts": {"floor": "floor", "slope": "traversable surface",
                    "ego": "ego robot"},
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(post, endpoint, concurrent_body) for _ in range(2)]
        concurrent_results = [future.result() for future in futures]

    single_body = {"image": encoded, "prompt": "person"}
    single_prefetch, single_prefetch_seconds = post(
        args.sam_url.rstrip("/") + "/prefetch_text", single_body)
    single_foreground, single_foreground_seconds = post(
        args.sam_url.rstrip("/") + "/segment_text", single_body)

    report = {
        "schema_version": "labelagent.remote-sam-cache-live.v1",
        "image_bytes": len(image),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "first": summary(first_result, first_seconds),
        "exact_repeat": summary(second_result, second_seconds),
        "changed_prompt": summary(changed_result, changed_seconds),
        "concurrent_identical": [summary(result, seconds)
                                 for result, seconds in concurrent_results],
        "single_prefetch": summary(single_prefetch, single_prefetch_seconds),
        "single_foreground": summary(single_foreground, single_foreground_seconds),
    }
    report["acceptance"] = {
        "exact_core_match": report["first"]["core_sha256"]
                            == report["exact_repeat"]["core_sha256"],
        "exact_repeat_hit": report["exact_repeat"]["cache_hit"] is True,
        "changed_prompt_miss": report["changed_prompt"]["cache_hit"] is False,
        "concurrent_core_match": len({item["core_sha256"] for item in
                                      report["concurrent_identical"]}) == 1,
        "concurrent_coalesced": any(item["coalesced"] is True for item in
                                    report["concurrent_identical"]),
        "single_core_match": report["single_prefetch"]["core_sha256"]
                             == report["single_foreground"]["core_sha256"],
        "single_foreground_hit": report["single_foreground"]["cache_hit"] is True,
    }
    report["accepted"] = all(report["acceptance"].values())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
