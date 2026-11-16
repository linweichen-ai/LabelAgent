"""Linux process identity and stop conditions for a bounded owned GPU worker."""
from pathlib import Path
import os
import signal


def process_identity(pid, proc_root="/proc"):
    directory = Path(proc_root) / str(pid)
    try:
        stat = (directory / "stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        argv = tuple(part.decode() for part in (directory / "cmdline").read_bytes().split(b"\0") if part)
        if not argv:
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]), "argv": list(argv), "uid": directory.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_owned_process(current, expected):
    return bool(current and expected and current == expected)


def signal_owned(pid, expected, signal_number, send_signal=None, proc_root="/proc"):
    if send_signal is not None:
        if not same_owned_process(process_identity(pid, proc_root), expected):
            return False
        send_signal(pid, signal_number)
        return True
    # Hold the kernel process handle before checking identity; PID reuse after
    # the check cannot redirect a signal to a different process.
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        if not same_owned_process(process_identity(pid, proc_root), expected):
            return False
        try:
            signal.pidfd_send_signal(descriptor, signal_number)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def lease_stop_reason(now, deadline, free_mib, reserve_mib=4096, foreign_compute=False):
    if now >= deadline:
        return "deadline"
    if foreign_compute:
        return "foreign_compute_started"
    if free_mib < reserve_mib:
        return "headroom_below_reserve"
    return None
