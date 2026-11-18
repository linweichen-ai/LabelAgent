"""Start the local LabelAgent workbench against private services on factory-4090."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.request
import webbrowser

import psutil


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SSH = Path(shutil.which("ssh") or r"C:\Windows\System32\OpenSSH\ssh.exe")
RECORD = LOCAL / "remote_processes.json"
ISOLATE_RECORD = LOCAL / "isolate_processes.json"
REMOTE_HOST = "factory-4090"
FAST_REMOTE_HOST = "factory-5090"
REMOTE_START = "/home/clw/labelagent_remote_20261110/start_remote_services.sh"
SAM_PORT = 28766
VLM_PORT = 28868
FAST_VLM_PORT = 38868
FAST_SAM_PORT = 38766
WEB_PORT = 8765
STORAGE_WARNING_BYTES = 2 * 1024 ** 3
STORAGE_CRITICAL_BYTES = 256 * 1024 ** 2


def optional_fast_wait_seconds(environment=None):
    """Bound optional acceleration probing without delaying the primary workbench."""
    environment = os.environ if environment is None else environment
    raw = environment.get("LABELAGENT_FAST_ROUTE_WAIT_SECONDS", "3")
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = 3.0
    return min(20.0, max(1.0, seconds))


def storage_free_bytes(path):
    """Return free bytes for a path even when its final directory is new."""
    target = Path(path).resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def prepare_runtime_environment(environment, runtime_temp):
    """Keep workbench temporary writes off a low-headroom system volume."""
    prepared = environment.copy()
    system_drive = prepared.get("SystemDrive")
    system_volume = Path(f"{system_drive}\\" if system_drive else Path.home().anchor).resolve()
    system_temp = Path(
        prepared.get("TEMP") or prepared.get("TMP") or tempfile.gettempdir()
    ).resolve()
    runtime_temp = Path(runtime_temp).resolve()
    runtime_temp.mkdir(parents=True, exist_ok=True)
    system_volume_free = storage_free_bytes(system_volume)
    system_temp_free = storage_free_bytes(system_temp)
    runtime_free = storage_free_bytes(runtime_temp)
    if runtime_free < STORAGE_CRITICAL_BYTES:
        raise RuntimeError(
            f"工作台临时目录可用空间不足 256 MiB：{runtime_temp}；"
            "为避免保存损坏，本次不会启动")
    prepared.update(
        TEMP=str(runtime_temp), TMP=str(runtime_temp), TMPDIR=str(runtime_temp),
        LABELAGENT_RUNTIME_TEMP=str(runtime_temp))
    return prepared, {
        "system_volume": str(system_volume),
        "system_volume_free_bytes": system_volume_free,
        "system_temp": str(system_temp),
        "system_temp_free_bytes": system_temp_free,
        "runtime_temp": str(runtime_temp),
        "runtime_free_bytes": runtime_free,
        "redirected": system_temp != runtime_temp,
    }


def print_storage_advisory(report):
    system_volume_free = report["system_volume_free_bytes"]
    runtime_free = report["runtime_free_bytes"]
    if system_volume_free < STORAGE_WARNING_BYTES:
        print(
            f"提示：Windows 系统盘 {report['system_volume']} 可用空间仅 "
            f"{system_volume_free / 1024 ** 3:.2f} GiB；工作台临时文件已固定到 "
            f"{report['runtime_temp']}（可用 {runtime_free / 1024 ** 3:.2f} GiB）")
    if runtime_free < STORAGE_WARNING_BYTES:
        print(
            "提示：工作台数据盘可用空间低于 2 GiB，请先释放可恢复空间，"
            "再导入大批量数据")


def records(path=RECORD):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(items, path=RECORD):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def managed_process(item):
    try:
        proc = psutil.Process(item["pid"])
        command = proc.cmdline()
        marker = item["marker"].casefold()
        executable = item["executable"].casefold()
        if (abs(proc.create_time() - item["created"]) < 1.0
                and any(marker in arg.casefold() for arg in command)
                and any(executable == str(Path(arg)).casefold() for arg in command[:1])):
            return proc
    except (psutil.Error, KeyError, OSError):
        pass
    return None


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def request_ok(port, endpoint, timeout=3):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}{endpoint}", timeout=timeout) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def post_json(port, endpoint, body, timeout=10, accepted_statuses=(200,)):
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        if response.status not in accepted_statuses:
            raise RuntimeError(f"配置标注页面失败：HTTP {response.status}")
        raw = response.read()
        return json.loads(raw) if raw else {}


def prewarm_primary_vlm(attempts=3):
    """Move the first full discovery decode before the browser opens."""
    for attempt in range(attempts):
        result = post_json(
            WEB_PORT, "/api/model_prewarm", {}, timeout=90,
            accepted_statuses=(200, 202))
        if result.get("prewarmed") is True:
            return result
        code = result.get("code")
        if code != "vlm_busy" or attempt + 1 >= attempts:
            return result
        time.sleep(2)
    return {"prewarmed": False, "code": "attempts_exhausted"}


def wait_for(item, checks, seconds, label):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if managed_process(item) is None:
            raise RuntimeError(f"{label}进程已经退出，请查看 .local\\logs 中的日志")
        if all(request_ok(port, endpoint) for port, endpoint in checks):
            return
        time.sleep(1)
    raise RuntimeError(f"等待{label}超时，请查看 .local\\logs 中的日志")


def stop_item(item):
    proc = managed_process(item)
    if proc is None:
        return
    owned = proc.children(recursive=True) + [proc]
    for child in reversed(owned):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=15)
    for child in alive:
        child.kill()
    psutil.wait_procs(alive, timeout=5)


def preflight():
    required = (PYTHON, SSH, ROOT / "source" / "agent" / "server.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("启动文件不完整：" + "；".join(missing))


def remote_start():
    result = subprocess.run(
        [str(SSH), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         REMOTE_HOST, f"bash {REMOTE_START}"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError("4090 服务启动请求失败：" + message)


def remote_fast_ready(timeout=3):
    """Check the optional 5090 VLM before creating a noisy local tunnel."""
    try:
        result = subprocess.run(
            [str(SSH), "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             FAST_REMOTE_HOST,
             "curl -fsS --max-time 2 http://127.0.0.1:18878/v1/models >/dev/null"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout + 2,
            creationflags=subprocess.CREATE_NO_WINDOW)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def remote_5090_sam_ready(timeout=3):
    """Check the already-deployed 5090 SAM service before tunnelling it."""
    try:
        result = subprocess.run(
            [str(SSH), "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             FAST_REMOTE_HOST,
             "curl -fsS --max-time 2 http://127.0.0.1:18766/health >/dev/null"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout + 2,
            creationflags=subprocess.CREATE_NO_WINDOW)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def start_5090_sam_tunnel(items, environment, newly_started):
    """Prefer the dedicated 5090 SAM while retaining 4090 SAM as fallback."""
    if "sam_5090_tunnel" not in items:
        if not remote_5090_sam_ready():
            raise RuntimeError("5090 远端 SAM 尚未就绪，继续使用 4090 SAM")
        command = [
            str(SSH), "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6", "-o", "TCPKeepAlive=yes",
            "-L", f"{FAST_SAM_PORT}:127.0.0.1:18766", FAST_REMOTE_HOST]
        with (LOCAL / "logs" / "sam_5090_tunnel.log").open("ab") as output:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
        item = {"pid": process.pid, "created": psutil.Process(process.pid).create_time(),
                "marker": f"{FAST_SAM_PORT}:127.0.0.1:18766",
                "executable": str(SSH), "ports": [FAST_SAM_PORT]}
        items["sam_5090_tunnel"] = item
        newly_started.append("sam_5090_tunnel")
        save(items)
    wait_for(items["sam_5090_tunnel"], ((FAST_SAM_PORT, "/health"),),
             optional_fast_wait_seconds(), "5090 SAM 通道")


def start_optional_fast_tunnel(items, environment, newly_started):
    """Start the optional tunnel only when its remote service is already ready."""
    if "fast_tunnel" not in items:
        if not remote_fast_ready():
            raise RuntimeError("5090 远端 VLM 尚未就绪，已跳过可选安全通道")
        tunnel_command = [
            str(SSH), "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6", "-o", "TCPKeepAlive=yes",
            "-L", f"{FAST_VLM_PORT}:127.0.0.1:18878", FAST_REMOTE_HOST]
        with (LOCAL / "logs" / "fast_vlm_tunnel.log").open("ab") as output:
            process = subprocess.Popen(
                tunnel_command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
        item = {"pid": process.pid, "created": psutil.Process(process.pid).create_time(),
                "marker": f"{FAST_VLM_PORT}:127.0.0.1:18878",
                "executable": str(SSH), "ports": [FAST_VLM_PORT]}
        items["fast_tunnel"] = item
        newly_started.append("fast_tunnel")
        save(items)
    wait_for(items["fast_tunnel"], ((FAST_VLM_PORT, "/v1/models"),),
             optional_fast_wait_seconds(), "5090 可选加速通道")


def start(open_browser):
    preflight()
    items = records()
    for name in ("tunnel", "sam_5090_tunnel", "fast_tunnel", "workbench"):
        if managed_process(items.get(name, {})) is None:
            items.pop(name, None)
    save(items)

    if "workbench" not in items and port_busy(WEB_PORT):
        raise RuntimeError("本机标注页面仍在运行。请先执行 Stop-LabelAgent.cmd，再启动 4090 版。")
    for port in (SAM_PORT, VLM_PORT):
        if "tunnel" not in items and port_busy(port):
            raise RuntimeError(f"本机端口 {port} 已被其他程序使用，请先确认该程序。")
    if "fast_tunnel" not in items and port_busy(FAST_VLM_PORT):
        raise RuntimeError(f"本机端口 {FAST_VLM_PORT} 已被其他程序使用，请先确认该程序。")
    if "sam_5090_tunnel" not in items and port_busy(FAST_SAM_PORT):
        raise RuntimeError(f"本机端口 {FAST_SAM_PORT} 已被其他程序使用，请先确认该程序。")

    remote_start()
    (LOCAL / "logs").mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        PYTHONUTF8="1", PYTHONUNBUFFERED="1", AUTOLABEL_PORT=str(WEB_PORT),
        SAM3_URL=f"http://127.0.0.1:{SAM_PORT}", NO_PROXY="127.0.0.1,localhost")
    newly_started = []
    try:
        if "tunnel" not in items:
            tunnel_command = [
                str(SSH), "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=6", "-o", "TCPKeepAlive=yes",
                "-L", f"{SAM_PORT}:127.0.0.1:18766",
                "-L", f"{VLM_PORT}:127.0.0.1:18878", REMOTE_HOST]
            with (LOCAL / "logs" / "remote_tunnel.log").open("ab") as output:
                process = subprocess.Popen(
                    tunnel_command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
            item = {"pid": process.pid, "created": psutil.Process(process.pid).create_time(),
                    "marker": f"{SAM_PORT}:127.0.0.1:18766",
                    "executable": str(SSH), "ports": [SAM_PORT, VLM_PORT]}
            items["tunnel"] = item
            newly_started.append("tunnel")
            save(items)
        wait_for(items["tunnel"], ((SAM_PORT, "/health"), (VLM_PORT, "/v1/models")),
                 240, "4090 模型与安全通道")

        sam_endpoint_port = SAM_PORT
        try:
            start_5090_sam_tunnel(items, environment, newly_started)
            sam_endpoint_port = FAST_SAM_PORT
        except Exception as error:
            stop_item(items.pop("sam_5090_tunnel", {}))
            newly_started = [name for name in newly_started if name != "sam_5090_tunnel"]
            save(items)
            print(f"提示：5090 SAM 通道暂不可用，将继续使用 4090：{error}")
        environment["SAM3_URL"] = f"http://127.0.0.1:{sam_endpoint_port}"

        fast_ready = False
        try:
            start_optional_fast_tunnel(items, environment, newly_started)
            fast_ready = True
        except Exception as error:
            stop_item(items.pop("fast_tunnel", {}))
            newly_started = [name for name in newly_started if name != "fast_tunnel"]
            save(items)
            print(f"提示：5090 可选加速通道暂不可用，将继续使用 4090：{error}")

        if "workbench" not in items:
            command = [str(PYTHON), "-u", "-m", "source.agent.server"]
            workbench_environment, storage = prepare_runtime_environment(
                environment, LOCAL / "runtime-temp" / f"workbench-{WEB_PORT}")
            print_storage_advisory(storage)
            with (LOCAL / "logs" / "remote_workbench.log").open("ab") as output:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=workbench_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
            item = {"pid": process.pid, "created": psutil.Process(process.pid).create_time(),
                    "marker": "source.agent.server", "executable": str(PYTHON),
                    "ports": [WEB_PORT]}
            items["workbench"] = item
            newly_started.append("workbench")
            save(items)
        wait_for(items["workbench"], ((WEB_PORT, "/api/state"),), 90, "标注页面")
        model_config = {
            "llm_url": f"http://127.0.0.1:{VLM_PORT}/v1",
            "llm_model": "labelagent-vlm8b",
            "sam3_url": f"http://127.0.0.1:{sam_endpoint_port}",
            "llm_fast_url": (f"http://127.0.0.1:{FAST_VLM_PORT}/v1"
                             if fast_ready else ""),
            "llm_fast_model": "labelagent-vlm8b" if fast_ready else "",
            "llm_fast_max_pixels": 640 * 512,
            "llm_fast_max_long_edge": 640,
            "llm_fast_max_extra_pixels": 4 * 640 * 512,
            "llm_fast_max_extra_long_edge": 1280,
            "llm_fast_timeout_seconds": 60,
            "llm_fast_cooldown_seconds": 60,
        }
        post_json(WEB_PORT, "/api/config", model_config)
        try:
            print("正在准备主 VLM 首次推理路径，请稍候……", flush=True)
            prewarm = prewarm_primary_vlm()
            if prewarm.get("prewarmed") is True:
                print(f"主 VLM 完整预热已就绪：{prewarm.get('elapsed_ms', 0) / 1000:.3f} 秒")
            else:
                print(f"提示：主 VLM 完整预热已跳过：{prewarm.get('code', 'unknown')}")
        except Exception as error:
            print(f"提示：主 VLM 完整预热失败，工作台仍可使用：{error}")
        print(f"双服务器 LabelAgent 已启动：http://127.0.0.1:{WEB_PORT}")
        print("SAM3 优先使用 5090、失败时回退 4090；主 VLM 使用 4090，5090 VLM 就绪后自动加速小图。")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{WEB_PORT}")
    except Exception:
        for name in reversed(newly_started):
            stop_item(items.pop(name, {}))
        save(items)
        raise


def stop():
    items = records()
    for name in ("workbench", "fast_tunnel", "sam_5090_tunnel", "tunnel"):
        stop_item(items.get(name, {}))
    save({})
    print("4090 版 LabelAgent 本机页面和安全通道已停止；服务器模型保持热启动。")


def status():
    items = records()
    print(f"安全通道：{'正常' if managed_process(items.get('tunnel', {})) else '未运行'}")
    print(f"远端 SAM3：{'正常' if request_ok(SAM_PORT, '/health') else '不可用'}")
    print(f"5090 SAM 通道：{'正常' if request_ok(FAST_SAM_PORT, '/health') else '未启用或不可用'}")
    print(f"远端 Qwen：{'正常' if request_ok(VLM_PORT, '/v1/models') else '不可用'}")
    print(f"5090 加速通道：{'正常' if request_ok(FAST_VLM_PORT, '/v1/models') else '未启用或不可用'}")
    print(f"标注页面：{'正常' if request_ok(WEB_PORT, '/api/state') else '未运行'}")


def isolate_options(port, work_dir):
    if port is None or work_dir is None:
        raise ValueError("隔离模式必须同时指定 --port 和 --work-dir")
    if not 1 <= port <= 65535 or port in (8756, 8765, 18766, 18878, SAM_PORT, VLM_PORT, FAST_SAM_PORT):
        raise ValueError("隔离端口必须有效且不能使用标准工作台、manager 或模型通道端口")
    work = Path(work_dir).resolve()
    standard = (ROOT / "work").resolve()
    if work == ROOT or work in ROOT.parents or work == standard or standard in work.parents or work in standard.parents:
        raise ValueError("隔离工作目录不能覆盖项目目录或标准工作目录")
    return port, work


def isolate_executables():
    allowed = {PYTHON.resolve()}
    config = PYTHON.parent.parent / "pyvenv.cfg"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "executable":
                allowed.add(Path(value.strip()).resolve())
    return allowed


def isolate_model_config():
    """Bind isolated acceptance to the same verified tunnels as the workbench."""
    return {
        "llm_url": f"http://127.0.0.1:{VLM_PORT}/v1",
        "llm_model": "labelagent-vlm8b",
        "sam3_url": f"http://127.0.0.1:{SAM_PORT}",
    }


def isolate_labels(state_path=None):
    """Read the authoritative label registry without copying workbench state."""
    path = Path(state_path) if state_path is not None else ROOT / "work" / "state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("正式标签表不可读取，隔离验收不会使用猜测标签") from error
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise RuntimeError("正式标签表为空或无效，隔离验收已停止")
    return list(labels)


def isolate_identity(proc, token):
    command = proc.cmdline()
    exe = Path(proc.exe()).resolve()
    if (not token or len(command) != 5
            or command[1:] != ["-u", "-c", ISOLATE_LAUNCHER, token]
            or Path(command[0]).resolve() != exe or exe not in isolate_executables()):
        raise ValueError("隔离进程命令或解释器身份不符")
    return {"pid": proc.pid, "created": proc.create_time(), "executable": str(exe)}


def isolate_owned(item, port, work, discover=False):
    try:
        if (item["kind"] != "isolated-workbench" or item["port"] != port
                or item["work_dir"] != str(work)):
            return []
        identities = item.get("processes") or [
            {"pid": item["pid"], "created": item["created"], "executable": str(PYTHON.resolve())}]
        owned = []
        for identity in identities:
            try:
                proc = psutil.Process(identity["pid"])
            except psutil.NoSuchProcess:
                continue
            actual = isolate_identity(proc, item["token"])
            if (abs(actual["created"] - identity["created"]) >= 0.01
                    or actual["executable"] != identity["executable"]):
                return []
            owned.append(proc)
        if discover:
            known = {proc.pid for proc in owned}
            for parent in list(owned):
                for child in parent.children(recursive=True):
                    if child.pid not in known:
                        try:
                            identity = isolate_identity(child, item["token"])
                        except (psutil.Error, ValueError):
                            continue
                        identities.append(identity)
                        owned.append(child)
                        known.add(child.pid)
            item["processes"] = identities
        return owned
    except (psutil.Error, KeyError, TypeError, OSError, ValueError):
        return []


def isolate_process(item, port, work):
    owned = isolate_owned(item, port, work)
    return owned[0] if owned else None


def port_listening_pids(port):
    return {connection.pid for connection in psutil.net_connections(kind="inet")
            if (connection.status == psutil.CONN_LISTEN and connection.laddr
                and connection.laddr.port == port)}


def port_released(port, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket() as sock:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            time.sleep(0.1)
    return False


ISOLATE_LAUNCHER = "import runpy; runpy.run_module('source.agent.server', run_name='__main__')"


def stop_isolate_item(item, port, work):
    owned = isolate_owned(item, port, work, discover=True)
    if not owned:
        raise RuntimeError("隔离工作台进程身份无法确认；为避免停止其他程序，保留进程记录")
    listeners = port_listening_pids(port)
    if not listeners.issubset({proc.pid for proc in owned}):
        raise RuntimeError(f"隔离端口 {port} 被其他进程占用；保留进程记录")
    for proc in reversed(owned):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=15)
    for proc in alive:
        identity = next((entry for entry in item["processes"] if entry["pid"] == proc.pid), None)
        if identity and isolate_owned({**item, "processes": [identity]}, port, work):
            proc.kill()
    if alive:
        psutil.wait_procs(alive, timeout=5)
    if not port_released(port):
        raise RuntimeError(f"隔离端口 {port} 仍被占用；保留进程记录")


def isolate(action, port, work, open_browser=False):
    key = f"{port}:{work}"
    items = records(ISOLATE_RECORD)
    item = items.get(key, {})
    proc = isolate_process(item, port, work)
    if action == "stop":
        if item:
            stop_isolate_item(item, port, work)
            items.pop(key, None)
            save(items, ISOLATE_RECORD)
        elif not port_released(port):
            raise RuntimeError(f"隔离端口 {port} 被未记录进程占用；不会停止端口占用者")
        print("隔离工作台已停止；现有通道和模型保持不变。")
        return
    if action == "status":
        listeners = port_listening_pids(port)
        owned = {process.pid for process in isolate_owned(item, port, work)}
        healthy = proc and listeners and listeners.issubset(owned) and request_ok(port, '/api/state')
        print(f"隔离工作台：{'正常' if healthy else '未运行'}")
        return

    if not PYTHON.is_file() or not (ROOT / "source" / "agent" / "server.py").is_file():
        raise RuntimeError("隔离工作台启动文件不完整")
    if item and proc is None:
        raise RuntimeError("隔离工作台进程身份无法确认；保留进程记录")
    if proc is None:
        with socket.socket() as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as error:
                raise RuntimeError(f"隔离端口 {port} 不可用；不会停止端口占用者") from error
    for model_port, endpoint in ((SAM_PORT, "/health"),
                                 (VLM_PORT, "/v1/models")):
        if not request_ok(model_port, endpoint):
            raise RuntimeError(f"现有远端通道 {model_port} 不可用；请先准备通道，隔离模式不会启动模型或隧道")

    newly_started = False
    try:
        if proc is None:
            work.mkdir(parents=True, exist_ok=True)
            log_dir = LOCAL / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                PYTHONUTF8="1", PYTHONUNBUFFERED="1", LABELAGENT_WORK_DIR=str(work),
                AUTOLABEL_PORT=str(port),
                SAM3_URL=f"http://127.0.0.1:{SAM_PORT}",
                NO_PROXY="127.0.0.1,localhost")
            environment, storage = prepare_runtime_environment(
                environment, work / ".runtime-temp")
            print_storage_advisory(storage)
            token = uuid.uuid4().hex
            with (log_dir / f"isolated_workbench_{port}.log").open("ab") as output:
                process = subprocess.Popen(
                    [str(PYTHON), "-u", "-c", ISOLATE_LAUNCHER, token],
                    cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
            item = {"kind": "isolated-workbench", "pid": process.pid,
                    "created": psutil.Process(process.pid).create_time(),
                    "token": token, "port": port, "work_dir": str(work)}
            newly_started = True
            items[key] = item
            save(items, ISOLATE_RECORD)
        deadline = time.monotonic() + 90
        while True:
            if isolate_process(item, port, work) is None:
                raise RuntimeError("隔离工作台进程已退出，请查看隔离工作台日志")
            if request_ok(port, "/api/state"):
                listener_pids = port_listening_pids(port)
                owned_pids = {proc.pid for proc in isolate_owned(item, port, work, discover=True)}
                if not listener_pids or not listener_pids.issubset(owned_pids):
                    raise RuntimeError("隔离端口的监听进程身份与已记录进程不符")
                save(items, ISOLATE_RECORD)
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("等待隔离工作台超时")
            time.sleep(1)
        post_json(port, "/api/config", isolate_model_config())
        post_json(port, "/api/labels", {"labels": isolate_labels()})
        print(f"隔离工作台已启动：http://127.0.0.1:{port}")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception:
        if newly_started:
            stop_isolate_item(item, port, work)
            items.pop(key, None)
            save(items, ISOLATE_RECORD)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--work-dir")
    args = parser.parse_args()
    if args.port is not None or args.work_dir is not None:
        port, work = isolate_options(args.port, args.work_dir)
        isolate(args.action, port, work, args.action == "start" and not args.no_browser)
        return
    if args.action == "start":
        start(not args.no_browser)
    elif args.action == "stop":
        stop()
    else:
        status()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
