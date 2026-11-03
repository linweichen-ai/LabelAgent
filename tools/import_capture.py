"""Safely convert a four-camera/PCD capture ZIP to an indexed LabelAgent MCAP.

3-D annotations remain sidecars. They are never promoted to 2-D mask truth.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader
from mcap.writer import Writer


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_member(name, root):
    normalized = name.replace("\\", "/")
    part = PurePosixPath(normalized)
    if part.is_absolute() or not part.parts or any(
            item in {"..", "."} or ":" in item or item.endswith((" ", "."))
            for item in part.parts):
        raise ValueError("Unsafe archive member")
    target = root.joinpath(*part.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive member escapes output directory")
    return target


def extract(archive, root):
    hashes = {}
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        targets = set()
        for item in entries:
            target = safe_member(item.filename, root)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            if str(target).casefold() in targets:
                raise ValueError("Duplicate archive path")
            targets.add(str(target).casefold())
        for index, item in enumerate(entries):
            target = safe_member(item.filename, root)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            hashes[item.filename] = digest(target)
            if index % 100 == 0:
                print(f"extract {index + 1}/{len(entries)}", flush=True)
    return hashes


PCD_TYPES = {("F", 4): ("<f4", 7), ("F", 8): ("<f8", 8),
             ("U", 1): ("u1", 1), ("I", 1): ("i1", 2),
             ("U", 2): ("<u2", 3), ("I", 2): ("<i2", 4),
             ("U", 4): ("<u4", 5), ("I", 4): ("<i4", 6)}


def read_pcd(path):
    header = {}
    with Path(path).open("rb") as stream:
        for _ in range(100):
            raw = stream.readline()
            if not raw:
                raise ValueError("Incomplete PCD header")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *value = line.split()
            header[key] = value
            if key == "DATA":
                break
        else:
            raise ValueError("PCD header too large")
        if header.get("DATA") != ["binary"]:
            raise ValueError("Only binary PCD is supported; data was not converted")
        data = stream.read()
    names = header["FIELDS"]
    sizes = list(map(int, header["SIZE"]))
    counts = list(map(int, header.get("COUNT", ["1"] * len(names))))
    if not len(names) == len(sizes) == len(counts) == len(header["TYPE"]) or any(c != 1 for c in counts):
        raise ValueError("Unsupported PCD field layout")
    fields, offset = [], 0
    dtypes = []
    for name, size, kind in zip(names, sizes, header["TYPE"]):
        dtype, code = PCD_TYPES[kind, size]
        fields.append({"name": name, "offset": offset, "type": code})
        dtypes.append((name, dtype))
        offset += size
    count = int(header["POINTS"][0])
    if len(data) != count * offset or count != int(header["WIDTH"][0]) * int(header["HEIGHT"][0]):
        raise ValueError("PCD point count does not match payload")
    points = np.frombuffer(data, dtype=np.dtype(dtypes))
    if not all(axis in points.dtype.names for axis in ("x", "y", "z")):
        raise ValueError("PCD lacks XYZ fields")
    finite = np.isfinite(np.column_stack([points[axis] for axis in ("x", "y", "z")])).all(axis=1)
    return data, fields, offset, count, int((~finite).sum())


def pointcloud_schema():
    file = descriptor_pb2.FileDescriptorProto(name="labelagent_capture.proto", package="labelagent.capture", syntax="proto3")
    for name, fields in (
        ("Stamp", [("seconds", 1, 3, None, False), ("nanos", 2, 5, None, False)]),
        ("PackedField", [("name", 1, 9, None, False), ("offset", 2, 13, None, False), ("type", 3, 5, None, False)]),
        ("PointCloud", [("timestamp", 1, 11, ".labelagent.capture.Stamp", False),
                        ("point_stride", 2, 13, None, False),
                        ("fields", 3, 11, ".labelagent.capture.PackedField", True),
                        ("data", 4, 12, None, False)])):
        message = file.message_type.add(name=name)
        for field_name, number, kind, type_name, repeated in fields:
            field = message.field.add(name=field_name, number=number, type=kind, label=3 if repeated else 1)
            if type_name:
                field.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("labelagent.capture.PointCloud"))
    return descriptor_pb2.FileDescriptorSet(file=[file]).SerializeToString(), cls


def profile_annotations(root, stamps):
    result = {}
    for name in ("labels", "ali_label", "wilddet3d-labels"):
        files = [path for path in (root / name).glob("*.json") if path.stem.isdigit()]
        classes, rows, invalid, duplicate_ids = Counter(), [], 0, 0
        for path in sorted(files):
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            objects = value.get("objects", [])
            if str(value.get("frame_id")) != path.stem:
                raise ValueError("Annotation frame ID differs from filename")
            seen = set()
            for obj in objects:
                key = (obj.get("obj_type"), obj.get("obj_id"))
                duplicate_ids += key in seen
                seen.add(key)
                psr = obj.get("obj_psr", {})
                try:
                    numbers = np.array([psr[group][axis] for group, axes in (
                        ("position", ("x", "y", "z")), ("rotation", ("roll", "pitch", "yaw")),
                        ("scale", ("x", "y", "z"))) for axis in axes], dtype=float)
                    valid = np.isfinite(numbers).all() and (numbers[6:] > 0).all()
                except (KeyError, TypeError, ValueError):
                    valid = False
                invalid += not valid
                classes[str(obj.get("obj_type"))] += 1
            rows.append({"timestamp_ns": int(path.stem), "objects": len(objects)})
        available = {item["timestamp_ns"] for item in rows}
        result[name] = {"frames": len(rows), "object_count": sum(classes.values()),
                        "class_counts": dict(classes), "invalid_geometry": invalid,
                        "duplicate_class_object_ids": duplicate_ids,
                        "missing_capture_frames": len(set(stamps) - available),
                        "orphan_annotation_frames": len(available - set(stamps)), "rows": rows}
    return result


def build(archive, output):
    archive, output = Path(archive), Path(output)
    archive_hash = digest(archive)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    hashes = extract(archive, source)
    cameras = {path.name: sorted(path.glob("*.png"), key=lambda p: int(p.stem))
               for path in sorted((source / "cameras").iterdir()) if path.is_dir()}
    if not cameras:
        raise ValueError("No camera images found")
    first_stamps = [int(path.stem) for path in next(iter(cameras.values()))]
    if len(set(first_stamps)) != len(first_stamps) or any(
            [int(path.stem) for path in paths] != first_stamps for paths in cameras.values()):
        raise ValueError("Camera streams have different or duplicate timestamps")
    lidar_paths = sorted((source / "lidar").glob("*.pcd"), key=lambda p: int(p.stem))
    if [int(path.stem) for path in lidar_paths] != first_stamps:
        raise ValueError("Camera and LiDAR timestamps do not match")
    annotations = profile_annotations(source, first_stamps)
    if annotations["labels"]["invalid_geometry"] or annotations["labels"]["missing_capture_frames"]:
        raise ValueError("Reference labels have missing frames or invalid geometry")
    descriptor, cloud_cls = pointcloud_schema()
    mcap_path = output / "capture.mcap"
    records, dimensions, point_counts, invalid_xyz = [], Counter(), [], 0
    with mcap_path.open("xb") as stream:
        writer = Writer(stream)
        writer.start(profile="LabelAgent-capture-images-and-PCD")
        image_schema = writer.register_schema("LabelAgent.CompressedImage", "", b"")
        cloud_schema = writer.register_schema("labelagent.capture.PointCloud", "protobuf", descriptor)
        image_channels = {name: writer.register_channel("/capture/" + name + "/image", "", image_schema) for name in cameras}
        cloud_channel = writer.register_channel("/capture/lidar", "protobuf", cloud_schema)
        writer.add_metadata("capture_import", {"archive_sha256": archive_hash,
            "reference_source": "labels", "timestamps": "original filename nanoseconds",
            "annotations": "3-D cuboids preserved as sidecars; no 2-D mask truth"})
        for index, timestamp in enumerate(first_stamps):
            for name, paths in cameras.items():
                raw = paths[index].read_bytes()
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                writer.add_message(image_channels[name], timestamp, raw, publish_time=timestamp, sequence=index)
                records.append({"topic": "/capture/" + name + "/image", "frame": index,
                                "timestamp_ns": timestamp, "sha256": hashlib.sha256(raw).hexdigest(),
                                "source_file": str(paths[index].relative_to(output))})
            data, fields, stride, count, invalid = read_pcd(lidar_paths[index])
            cloud = cloud_cls(point_stride=stride, data=data)
            cloud.timestamp.seconds, cloud.timestamp.nanos = divmod(timestamp, 10**9)
            for field in fields:
                cloud.fields.add(**field)
            encoded = cloud.SerializeToString()
            writer.add_message(cloud_channel, timestamp, encoded, publish_time=timestamp, sequence=index)
            records.append({"topic": "/capture/lidar", "frame": index, "timestamp_ns": timestamp,
                            "sha256": hashlib.sha256(encoded).hexdigest(), "points": count})
            point_counts.append(count)
            invalid_xyz += invalid
            if index % 25 == 0:
                print(f"convert {index + 1}/{len(first_stamps)}", flush=True)
        writer.finish()
    expected = {(item["topic"], item["timestamp_ns"]): item for item in records}
    checked = 0
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary.statistics.message_count != len(records):
            raise ValueError("MCAP summary message count mismatch")
        for schema, channel, message in reader.iter_messages():
            item = expected[channel.topic, message.log_time]
            if hashlib.sha256(message.data).hexdigest() != item["sha256"]:
                raise ValueError("MCAP payload changed during conversion")
            checked += 1
    contact = Image.new("RGB", (4 * 256, len(cameras) * 225), "white")
    draw = ImageDraw.Draw(contact)
    indices = np.linspace(0, len(first_stamps) - 1, 4, dtype=int)
    for row, (name, paths) in enumerate(cameras.items()):
        for col, index in enumerate(indices):
            with Image.open(paths[index]) as image:
                image.thumbnail((256, 204))
                contact.paste(image.convert("RGB"), (col * 256, row * 225 + 20))
            draw.text((col * 256 + 4, row * 225 + 3), f"{name} frame {index}", fill="black")
    contact.save(output / "contact_sheet.jpg", quality=88)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size, "extracted_file_count": len(hashes),
        "extracted_sha256": hashes, "reference_source": "labels", "reference_is_independent_ground_truth": False,
        "frame_groups": len(first_stamps), "camera_count": len(cameras), "image_count": sum(map(len, cameras.values())),
        "lidar_frames": len(lidar_paths), "image_dimensions": dict(dimensions),
        "timestamps_ns": first_stamps, "duration_seconds": (first_stamps[-1] - first_stamps[0]) / 1e9,
        "camera_lidar_timestamp_join_exact": True, "point_counts": point_counts, "nonfinite_xyz_points": invalid_xyz,
        "annotations": annotations, "mcap_messages_verified": checked, "mcap_sha256": digest(mcap_path),
        "mcap_file": str(mcap_path.resolve()), "records": records,
        "pixel_ground_truth_available": False, "annotations_3d_promoted_to_2d_truth": False}
    (output / "import_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    public = {key: manifest[key] for key in ("created_utc", "archive_sha256", "extracted_file_count", "reference_source",
        "reference_is_independent_ground_truth", "frame_groups", "camera_count", "image_count", "lidar_frames",
        "image_dimensions", "duration_seconds", "camera_lidar_timestamp_join_exact", "nonfinite_xyz_points",
        "mcap_messages_verified", "pixel_ground_truth_available", "annotations_3d_promoted_to_2d_truth")}
    public["annotation_sources"] = {name: {key: value for key, value in stats.items() if key != "rows"}
                                    for name, stats in annotations.items()}
    (output / "public_import_summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True, help="New dataset directory; existing data is not overwritten")
    args = parser.parse_args()
    build(args.archive, args.output)
