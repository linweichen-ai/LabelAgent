"""Keep the workbench online while SAM3 and NF4 Qwen take turns on one GPU.

Only processes created by this manager are terminated. Public endpoints remain
stable; private workers are replaced at model transitions to release CUDA/RAM.
"""
import argparse
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import psutil

from flask import Flask, Response, jsonify, request
from werkzeug.serving import make_server


def attach_cleanup_job():
    """Windows descendants exit with this manager, including venv launchers."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                    ("max_working", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", Counters), ("process_mem", ctypes.c_size_t),
                    ("job_mem", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # Retain until process exit. Closing it here would terminate this process.
    return handle


def http(url, body=None, timeout=10):
    headers = {"Content-Type": "application/json"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), "application/json"


class WorkerManager:
    def __init__(self, specs, root, log_dir, event_path, startup_timeout=150):
        self.specs, self.root = specs, str(root)
        self.log_dir, self.event_path = Path(log_dir), Path(event_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self.lock = threading.Lock()
        self.child = None
        self.kind = None
        self.log = None
        self.ready = False
        self.events = []
        self.started = 0
        self.stopped = 0

    def event(self, event, **details):
        self.events.append({"at": datetime.now(timezone.utc).isoformat(), "event": event, **details})
        temporary = self.event_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        temporary.replace(self.event_path)

    def stop(self):
        if self.child is None:
            return
        # Popen is retained from our own spawn, never selected by a process name.
        child, kind = self.child, self.kind
        descendants = []
        if child.poll() is None:
            try:
                descendants = psutil.Process(child.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            # Windows venv python.exe is a launcher; terminate its owned child
            # as well so that the actual CUDA context is gone before switching.
            for process in reversed(descendants):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(descendants, timeout=15)
            for process in alive:
                process.kill()
            psutil.wait_procs(alive, timeout=10)
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        self.stopped += 1
        if any(process.is_running() for process in descendants):
            raise RuntimeError("An owned worker descendant has not exited; no replacement was started")
        self.event("worker_exited", kind=kind, pid=child.pid, return_code=child.returncode,
                   descendant_pids=[process.pid for process in descendants], descendants_exited=True)
        self.child, self.kind, self.ready = None, None, False
        if self.log:
            self.log.close()
            self.log = None
        # Give Windows time to reclaim the exited process's CUDA context.
        time.sleep(0.5)

    def activate(self, kind):
        if self.kind == kind and self.ready and self.child.poll() is None:
            return
        self.stop()  # Wait for exit before starting the other worker.
        spec = self.specs[kind]
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost"})
        env.update(spec.get("env", {}))
        self.log = (self.log_dir / f"{kind}_{self.started:04d}.log").open("wb")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        self.child = subprocess.Popen(spec["command"], cwd=self.root, env=env,
                                      stdout=self.log, stderr=subprocess.STDOUT, **options)
        self.kind, self.ready = kind, False
        self.started += 1
        self.event("worker_started", kind=kind, pid=self.child.pid, live_owned_workers=1)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.child.poll() is not None:
                self.event("worker_failed", kind=kind, return_code=self.child.returncode)
                raise RuntimeError(f"{kind} worker exited; inspect its local log")
            try:
                status, data, _ = http(spec["url"] + "/health", timeout=2)
                health = json.loads(data)
                if status == 200 and health.get("status") == "ok":
                    self.ready = True
                    self.event("worker_ready", kind=kind, pid=self.child.pid,
                               model_loaded=health.get("model_loaded"), quantization=health.get("quantization"))
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{kind} startup timed out")

    def call(self, kind, path, body):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another model request is active; retry after it finishes")
        started = time.monotonic()
        try:
            self.activate(kind)
            result = http(self.specs[kind]["url"] + path, body, timeout=270 if kind == "qwen" else 570)
            stats = self.health(kind, "").get("last_inference_stats") if kind == "qwen" else None
            self.event("request_finished", kind=kind, endpoint=path, http_status=result[0],
                       seconds=round(time.monotonic()-started, 3), inference_stats=stats)
            return result
        finally:
            self.lock.release()

    def health(self, kind, model):
        if self.kind == kind and self.ready and self.child.poll() is None:
            try:
                status, data, _ = http(self.specs[kind]["url"] + "/health", timeout=3)
                if status == 200:
                    return json.loads(data) | {"on_demand": True}
            except (OSError, ValueError):
                pass
        return {"status": "ok", "model_loaded": False, "on_demand": True,
                "model": model, "backend": "transformers_local_cuda" if kind == "qwen" else "sam3"}

    def status(self):
        alive = self.child is not None and self.child.poll() is None
        return {"active_worker": self.kind if alive else None, "worker_ready": self.ready and alive,
                "owned_worker_pid": self.child.pid if alive else None,
                "live_owned_workers": int(alive), "workers_started": self.started,
                "workers_stopped": self.stopped, "events": self.events}


def create_apps(manager, model):
    sam_app, vlm_app = Flask("sam3_on_demand"), Flask("qwen_on_demand")
    for app, kind in ((sam_app, "sam3"), (vlm_app, "qwen")):
        app.config["MAX_CONTENT_LENGTH"] = 64*1024*1024
        app.add_url_rule("/health", "health", lambda kind=kind: jsonify(manager.health(kind, model)))
        app.add_url_rule("/manager/status", "status", lambda: jsonify(manager.status()))

        @app.errorhandler(RuntimeError)
        @app.errorhandler(OSError)
        def unavailable(error):
            return jsonify(error={"message": str(error)[:300]}), 503

    def forward(kind, path):
        status, data, content_type = manager.call(kind, path, request.get_data())
        return Response(data, status=status, content_type=content_type)

    sam_app.add_url_rule("/segment_text", "text", lambda: forward("sam3", "/segment_text"), methods=["POST"])
    sam_app.add_url_rule("/segment_points", "points", lambda: forward("sam3", "/segment_points"), methods=["POST"])
    vlm_app.add_url_rule("/v1/chat/completions", "chat", lambda: forward("qwen", "/v1/chat/completions"), methods=["POST"])

    @vlm_app.get("/v1/models")
    def models():
        return jsonify(object="list", data=[{"id": model, "object": "model", "owned_by": "local"}])

    @vlm_app.post("/activate")
    def activate():
        status, data, _ = manager.call("qwen", "/health", None)
        return Response(data, status=status, content_type="application/json")

    return sam_app, vlm_app


def ensure_ports_free(ports):
    for port in ports:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-python", required=True)
    parser.add_argument("--vlm-python", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--model-id", default="labelagent-vlm")
    parser.add_argument("--log-dir", default="work/local_gpu_manager")
    args = parser.parse_args()
    cleanup_job_handle = attach_cleanup_job()
    root = Path(__file__).resolve().parents[1]
    ensure_ports_free([8766, 18868, 17866, 18869])
    specs = {
        "sam3": {"command": [args.sam_python, "-u", "-m", "source.sam3_service.server"],
                 "url": "http://127.0.0.1:17866", "env": {"SAM3_PORT": "17866", "SAM3_HOST": "127.0.0.1", "CUDA_VISIBLE_DEVICES": args.gpu_uuid}},
        "qwen": {"command": [args.vlm_python, "-u", str(root/"tools/local_vlm_server.py"),
                              "--model-dir", args.model_dir, "--model-id", args.model_id,
                              "--gpu-uuid", args.gpu_uuid, "--profile", "local-8gb",
                              "--quantization", "nf4", "--context-budget", "16384", "--port", "18869"],
                 "url": "http://127.0.0.1:18869"}}
    log_dir = (root / args.log_dir).resolve()
    manager = WorkerManager(specs, root, log_dir, log_dir/"events.json")
    atexit.register(manager.stop)
    sam_app, vlm_app = create_apps(manager, args.model_id)
    sam_http = make_server("127.0.0.1", 8766, sam_app, threaded=True)
    vlm_http = make_server("127.0.0.1", 18868, vlm_app, threaded=True)
    thread = threading.Thread(target=sam_http.serve_forever, daemon=True)
    thread.start()
    print("Local GPU manager ready: SAM3 8766, Qwen 18868; workers load on demand", flush=True)
    try:
        vlm_http.serve_forever()
    finally:
        sam_http.shutdown()
        manager.stop()
