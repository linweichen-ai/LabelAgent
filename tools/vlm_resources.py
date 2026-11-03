"""Read-only GPU inventory and conservative shared-server allocation guard."""
import argparse
import csv
import io
import json
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
        if not row or len(row) != 6:
            raise ValueError("Unexpected GPU inventory format")
        uuid, name, total, used, utilization, driver = [value.strip() for value in row]
        result.append({"uuid": uuid, "name": name, "total_mib": int(total), "used_mib": int(used),
                       "free_mib": int(total)-int(used), "utilization_percent": int(utilization),
                       "driver": driver, "has_compute_process": uuid in busy})
    return result


def inventory():
    def query(fields, kind):
        return subprocess.run(["nvidia-smi", "--query-" + kind + "=" + fields,
                               "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True, timeout=15).stdout
    return parse_inventory(query("uuid,name,memory.total,memory.used,utilization.gpu,driver_version", "gpu"),
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
