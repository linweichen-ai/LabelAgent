"""Dependency-light contract test for the deployed FastAPI SAM cache wrapper."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
import types


service = None
ExactResponseCoordinator = None


PNG_1X1 = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)).decode("ascii")


def core(payload):
    return payload["prompts"]


def reset():
    service.response_cache = ExactResponseCoordinator(max_entries=4, max_bytes=1024 * 1024)
    service.request_lock = threading.Lock()


def main():
    global service, ExactResponseCoordinator
    fake_point_segment = types.ModuleType("point_segment")
    fake_point_segment._model_lock = threading.Lock()
    fake_point_segment._predictor = lambda: (None, None)
    fake_point_segment.model_loaded = lambda: True
    fake_point_segment.segment = lambda *args, **kwargs: {"results": []}
    sys.modules["point_segment"] = fake_point_segment

    import sam_service_fastapi as imported_service
    from response_cache import ExactResponseCoordinator as imported_coordinator
    service = imported_service
    ExactResponseCoordinator = imported_coordinator
    calls = []

    def fake_predict(_image, prompts):
        calls.append(list(prompts.items()))
        time.sleep(0.05)
        return ({label: {"results": [{"parts": [], "score": 0.5}],
                         "diagnostics": {"returned_count": 1}}
                 for label in prompts}, {"total_seconds": 0.05})

    service.predict_text_prompts = fake_predict
    body = {"image": PNG_1X1, "prompts": {"floor": "floor", "slope": "ramp"}}

    reset()
    first = service.segment_text_multi(body)
    second = service.segment_text_multi(body)
    assert core(first) == core(second)
    assert len(calls) == 1
    assert first["diagnostics"]["response_cache"]["hit"] is False
    assert second["diagnostics"]["response_cache"]["hit"] is True

    reset()
    calls.clear()
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        service.segment_text_multi(body))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert len(results) == 2 and core(results[0]) == core(results[1])
    assert any(item["diagnostics"]["response_cache"]["coalesced"]
               for item in results)

    reset()
    calls.clear()
    service.request_lock.acquire()
    try:
        busy = service.prefetch_text_multi(body)
    finally:
        service.request_lock.release()
    assert busy.status_code == 202
    assert json.loads(busy.body)["code"] == "sam_busy"
    assert calls == []

    reset()
    calls.clear()
    warmed = service.prefetch_text_multi(body)
    foreground = service.segment_text_multi(body)
    assert warmed["prefetched"] is True
    assert foreground["diagnostics"]["response_cache"]["hit"] is True
    assert core(warmed) == core(foreground)
    assert len(calls) == 1

    changed = service.segment_text_multi({
        "image": PNG_1X1,
        "prompts": {"floor": "warehouse floor", "slope": "ramp"},
    })
    assert changed["diagnostics"]["response_cache"]["hit"] is False
    assert len(calls) == 2

    reset()
    calls.clear()
    single_body = {"image": PNG_1X1, "prompt": "traversable surface"}
    single_warm = service.prefetch_text(single_body)
    single_foreground = service.segment_text(single_body)
    assert single_warm["prefetched"] is True
    assert single_foreground["diagnostics"]["response_cache"]["hit"] is True
    assert single_warm["results"] == single_foreground["results"]
    assert len(calls) == 1

    print(json.dumps({
        "status": "pass",
        "exact_repeat_computations": 1,
        "coalesced_computations": 1,
        "busy_prefetch_status": 202,
        "prefetch_then_foreground_computations": 1,
        "different_prompt_was_cache_miss": True,
        "single_prefetch_then_foreground_computations": 1,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
