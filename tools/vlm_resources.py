"""Read-only GPU inventory and conservative shared-server allocation guard."""
import argparse
import csv
import io
import json
import math
import subprocess


def parse_inventory(gpu_text, process_text):
    busy = set()
    for row in csv.reader(io.StringIO(process_text)):
        if not row or not row[0].strip():
            continue
        if not row[0].strip().startswith("GPU-"):
            raise ValueError("GPU process inventory is unavailable")
        busy.add(row[0].strip())
    result = []
    for row in csv.reader(io.StringIO(gpu_text)):
        if not row or len(row) not in (6, 7):
            raise ValueError("Unexpected GPU inventory format")
        uuid, name, total, used, utilization, driver = [value.strip() for value in row[:6]]
        free = int(row[6].strip()) if len(row)==7 else int(total)-int(used)
        if not 0<=free<=int(total):raise ValueError("Invalid free GPU memory")
        result.append({"uuid": uuid, "name": name, "total_mib": int(total), "used_mib": int(used),
                       "free_mib": free, "free_memory_reported":len(row)==7,"utilization_percent": int(utilization),
                       "driver": driver, "has_compute_process": uuid in busy})
    return result


def inventory():
    def query(fields, kind):
        return subprocess.run(["nvidia-smi", "--query-" + kind + "=" + fields,
                               "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True, timeout=15).stdout
    return parse_inventory(query("uuid,name,memory.total,memory.used,utilization.gpu,driver_version,memory.free", "gpu"),
                           query("gpu_uuid,pid", "compute-apps"))


def require_idle_gpu(rows, uuid, minimum_free_mib=22000):
    matches = [row for row in rows if row["uuid"] == uuid]
    if len(matches) != 1:
        raise ValueError("Selected GPU UUID was not found")
    selected = matches[0]
    if "RTX 4090" not in selected["name"]:
        raise ValueError("Selected hardware is not the requested RTX 4090")
    if selected["has_compute_process"] or selected["utilization_percent"] > 10:
        raise ValueError("Selected GPU has existing compute work; no processes were stopped")
    if selected["free_mib"] < minimum_free_mib:
        raise ValueError("Selected GPU has insufficient free memory for this deployment profile")
    return selected


def require_local_gpu(rows, uuid, minimum_free_mib=3500):
    """Personal GPU profile: display processes are expected under Windows WDDM."""
    matches = [row for row in rows if row["uuid"] == uuid]
    if len(matches) != 1 or matches[0]["total_mib"] < 7500:
        raise ValueError("An explicitly selected local GPU with at least 8GB is required")
    if matches[0]["free_mib"] < minimum_free_mib:
        raise ValueError("Insufficient free local GPU memory; no existing processes were stopped")
    return matches[0]


def require_shared_probe_gpu(rows,uuid,memory_fraction=.35,reserve_mib=4096):
    """Opt-in short NF4 probe: bound our allocator and retain headroom for residents."""
    if not isinstance(memory_fraction,(int,float)) or isinstance(memory_fraction,bool) or not .1<=memory_fraction<=.35:
        raise ValueError('Probe memory fraction must be between .1 and .35')
    if type(reserve_mib) is not int or reserve_mib<4096:
        raise ValueError('A shared probe must reserve at least 4096 MiB')
    matches=[row for row in rows if row['uuid']==uuid]
    if len(matches)!=1 or 'RTX 4090' not in matches[0]['name']:
        raise ValueError('An explicitly selected RTX 4090 is required')
    selected=matches[0]
    if not selected.get('free_memory_reported'):
        raise ValueError('A direct free-memory inventory is required for this probe')
    if selected['utilization_percent']>5:
        raise ValueError('Selected GPU has active compute; wait before starting the probe')
    allocator_mib=math.ceil(selected['total_mib']*memory_fraction)
    overhead_mib=512
    if selected['free_mib']<allocator_mib+reserve_mib+overhead_mib:
        raise ValueError('Insufficient free memory for the capped probe and resident headroom')
    return {**selected,'process_allocator_fraction':float(memory_fraction),
        'process_allocator_budget_mib':allocator_mib,'reserved_free_mib':reserve_mib,
        'context_overhead_allowance_mib':overhead_mib,'existing_processes_stopped':0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--minimum-free-mib", type=int, default=22000)
    args = parser.parse_args()
    rows = inventory()
    result = {"gpus": rows, "read_only": True}
    if args.gpu_uuid:
        result["selected"] = require_idle_gpu(rows, args.gpu_uuid, args.minimum_free_mib)
    print(json.dumps(result, indent=2))
