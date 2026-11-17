"""Label-free LiDAR geometry used to propose ramp point prompts.

The routines in this module never read annotations.  They fit a dominant
support plane, find locally planar surfaces that are tilted relative to it,
cluster those surface samples in LiDAR space, and project the resulting
components into calibrated cameras.  A component is only a proposal: it is
not a slope label or pixel truth.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PCD_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("U", 1): "u1",
    ("I", 1): "i1",
    ("U", 2): "<u2",
    ("I", 2): "<i2",
    ("U", 4): "<u4",
    ("I", 4): "<i4",
}

PLANE_RECIPE = {
    "minimum_radius": 0.5,
    "maximum_radius": 20.0,
    "minimum_z": -4.0,
    "maximum_z": -0.15,
    "lower_quantile": 0.5,
    "sample_limit": 6000,
    "trials": 256,
    "seed": 0,
    "distance_threshold": 0.04,
    "maximum_tilt_degrees": 15.0,
    "minimum_origin_depth": 0.2,
    "maximum_origin_depth": 4.0,
    "minimum_pool_points": 100,
    "minimum_inlier_fraction": 0.55,
    "minimum_xy_span": 2.0,
}

TILTED_RECIPE = {
    "minimum_radius": 0.5,
    "maximum_radius": 12.0,
    "minimum_height": -0.03,
    "maximum_height": 0.65,
    "voxel_xy": 0.06,
    "neighborhood_radius": 0.24,
    "minimum_neighbors": 8,
    "maximum_surface_residual": 0.025,
    "minimum_planarity": 0.35,
    "minimum_tilt_degrees": 5.0,
    "maximum_tilt_degrees": 35.0,
    "cluster_radius": 0.16,
    "minimum_cluster_points": 6,
    "minimum_major_span": 0.24,
    "minimum_minor_span": 0.08,
    "minimum_height_span": 0.025,
    "maximum_cluster_area": 4.0,
}


def read_xyz_pcd(path: str | Path) -> np.ndarray:
    """Read finite, nonzero XYZ from a binary PCD without converting it."""
    header: dict[str, list[str]] = {}
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
            raise ValueError("Only binary PCD is supported")
        payload = stream.read()

    names = header.get("FIELDS", [])
    sizes = list(map(int, header.get("SIZE", [])))
    kinds = header.get("TYPE", [])
    counts = list(map(int, header.get("COUNT", ["1"] * len(names))))
    if not names or not len(names) == len(sizes) == len(kinds) == len(counts):
        raise ValueError("Invalid PCD field layout")
    if any(count != 1 for count in counts):
        raise ValueError("PCD vector fields are not supported")
    try:
        dtype = np.dtype([(name, PCD_TYPES[(kind, size)]) for name, size, kind in zip(names, sizes, kinds)])
    except KeyError as error:
        raise ValueError("Unsupported PCD field type") from error
    count = int(header["POINTS"][0])
    width = int(header["WIDTH"][0])
    height = int(header["HEIGHT"][0])
    if count != width * height or len(payload) != count * dtype.itemsize:
        raise ValueError("PCD point count does not match payload")
    points = np.frombuffer(payload, dtype=dtype)
    if not all(axis in points.dtype.names for axis in ("x", "y", "z")):
        raise ValueError("PCD lacks XYZ fields")
    xyz = np.column_stack([points[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.001)]


def camera_parameters(intrinsic: dict, extrinsic: dict) -> dict:
    """Validate the supplied PINHOLE calibration and return numeric arrays."""
    if intrinsic.get("model_type") != "PINHOLE" or extrinsic.get("calibrated") is not True:
        raise ValueError("Requires an explicitly calibrated PINHOLE camera")
    distortions = np.array([float(value) for value in intrinsic.get("distortion_parameters", {}).values()])
    if not np.isfinite(distortions).all() or np.any(distortions != 0):
        raise ValueError("Nonzero distortion is not silently ignored")
    width, height = int(intrinsic["image_width"]), int(intrinsic["image_height"])
    params = intrinsic["projection_parameters"]
    vector = np.array([params[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
    transform_keys = [key for key in ("transform:", "transform") if key in extrinsic]
    if len(transform_keys) != 1:
        raise ValueError("Extrinsic must contain exactly one supported transform field")
    transform = np.array(extrinsic[transform_keys[0]], dtype=float).reshape(4, 4)
    if width <= 0 or height <= 0 or not np.isfinite(vector).all() or (vector[:2] <= 0).any():
        raise ValueError("Invalid camera parameters")
    rotation = transform[:3, :3]
    if (
        not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or abs(np.linalg.det(rotation) - 1) > 0.01
        or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 0.01
    ):
        raise ValueError("Extrinsic is not a rigid transform")
    return {"width": width, "height": height, "intrinsics": vector, "transform": transform}


def project_visible(points: np.ndarray, camera: dict, cell_size: int = 8, depth_slack: float = 0.15, near: float = 0.2):
    """Project points and keep a conservative z-buffered visible cohort."""
    if cell_size < 1 or depth_slack < 0 or near <= 0:
        raise ValueError("Invalid visibility parameters")
    transform = camera["transform"]
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    good = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > near)
    source = np.flatnonzero(good)
    xyz = xyz[good]
    fx, fy, cx, cy = camera["intrinsics"]
    uv = xyz[:, :2] / xyz[:, 2, None] * [fx, fy] + [cx, cy]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < camera["width"] - 0.5)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera["height"] - 0.5)
    )
    source, xyz, uv = source[inside], xyz[inside], uv[inside]
    pixels = np.floor(uv + 0.5).astype(np.int32)
    columns = (camera["width"] + cell_size - 1) // cell_size
    rows = (camera["height"] + cell_size - 1) // cell_size
    cells = (pixels[:, 1] // cell_size) * columns + pixels[:, 0] // cell_size
    nearest = np.full(rows * columns, np.inf)
    np.minimum.at(nearest, cells, xyz[:, 2])
    visible = xyz[:, 2] <= nearest[cells] + depth_slack
    return source[visible], pixels[visible]


def fit_support_plane(points: np.ndarray, recipe: dict | None = None) -> dict:
    """Fit the deterministic lower dominant near-horizontal support plane."""
    p = deepcopy(PLANE_RECIPE if recipe is None else recipe)
    if set(p) != set(PLANE_RECIPE):
        raise ValueError("Invalid plane fitting recipe")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected XYZ points")
    points = points[np.isfinite(points).all(axis=1)]
    radius = np.linalg.norm(points[:, :2], axis=1)
    domain = points[
        (radius >= p["minimum_radius"])
        & (radius <= p["maximum_radius"])
        & (points[:, 2] >= p["minimum_z"])
        & (points[:, 2] <= p["maximum_z"])
    ]
    unavailable = {"usable": False, "domain_points": len(domain), "plane": None}
    if len(domain) < p["minimum_pool_points"]:
        return {**unavailable, "reason": "insufficient_lower_points"}
    pool = domain[domain[:, 2] <= np.quantile(domain[:, 2], p["lower_quantile"])]
    if len(pool) < p["minimum_pool_points"]:
        return {**unavailable, "reason": "insufficient_fitting_pool"}
    rng = np.random.default_rng(p["seed"])
    if len(pool) > p["sample_limit"]:
        pool = pool[rng.choice(len(pool), p["sample_limit"], replace=False)]
    cosine = math.cos(math.radians(p["maximum_tilt_degrees"]))
    best = None
    for _ in range(p["trials"]):
        a, b, c = pool[rng.choice(len(pool), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        offset = -float(normal @ a)
        if normal[2] < cosine or not p["minimum_origin_depth"] <= offset / normal[2] <= p["maximum_origin_depth"]:
            continue
        distances = np.abs(pool @ normal + offset)
        inliers = distances <= p["distance_threshold"]
        key = (int(inliers.sum()), -float(np.sum(distances[inliers] ** 2)))
        if best is None or key > best[0]:
            best = (key, inliers)
    if best is None:
        return {**unavailable, "reason": "no_supported_horizontal_plane"}
    if best[0][0] / len(pool) < p["minimum_inlier_fraction"]:
        return {**unavailable, "reason": "insufficient_plane_fraction", "fit_fraction": best[0][0] / len(pool)}
    inliers = pool[best[1]]
    if np.min(np.ptp(inliers[:, :2], axis=0)) < p["minimum_xy_span"]:
        return {**unavailable, "reason": "insufficient_spatial_span"}
    coefficients = np.linalg.lstsq(
        np.column_stack((inliers[:, :2], np.ones(len(inliers)))), -inliers[:, 2], rcond=None
    )[0]
    normal = np.array([*coefficients[:2], 1.0])
    normal /= np.linalg.norm(normal)
    offset = float(coefficients[2] / np.linalg.norm(np.array([*coefficients[:2], 1.0])))
    distances = np.abs(pool @ normal + offset)
    good = distances <= p["distance_threshold"]
    if normal[2] < cosine or float(np.mean(good)) < p["minimum_inlier_fraction"]:
        return {**unavailable, "reason": "refinement_failed"}
    return {
        "usable": True,
        "plane": [*normal.tolist(), offset],
        "domain_points": len(domain),
        "sampled_pool_points": len(pool),
        "fit_fraction": float(np.mean(good)),
        "origin_depth": offset / float(normal[2]),
        "tilt_degrees": math.degrees(math.acos(float(normal[2]))),
        "median_inlier_residual": float(np.median(distances[good])),
        "xy_span": np.ptp(pool[good, :2], axis=0).tolist(),
        "interpretation": "lower dominant near-horizontal plane candidate; not confirmed ground truth",
    }


def _voxel_means(points: np.ndarray, voxel_xy: float):
    keys = np.floor(points[:, :2] / voxel_xy).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None], counts


def tilted_planar_samples(points: np.ndarray, support: dict, recipe: dict | None = None) -> dict:
    """Return label-free locally planar samples tilted from the support plane."""
    p = deepcopy(TILTED_RECIPE if recipe is None else recipe)
    if set(p) != set(TILTED_RECIPE) or not support.get("usable"):
        raise ValueError("A usable support plane and complete tilted recipe are required")
    normal = np.asarray(support["plane"][:3], dtype=float)
    offset = float(support["plane"][3])
    radius = np.linalg.norm(points[:, :2], axis=1)
    height = points @ normal + offset
    domain_mask = (
        (radius >= p["minimum_radius"])
        & (radius <= p["maximum_radius"])
        & (height >= p["minimum_height"])
        & (height <= p["maximum_height"])
    )
    domain = points[domain_mask]
    centers, voxel_counts = _voxel_means(domain, p["voxel_xy"])
    tree = cKDTree(centers[:, :2])
    selected = []
    for index, center in enumerate(centers):
        neighbors = tree.query_ball_point(center[:2], p["neighborhood_radius"])
        if len(neighbors) < p["minimum_neighbors"]:
            continue
        local = centers[neighbors]
        covariance = np.cov(local - local.mean(axis=0), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        if values[-1] <= 0:
            continue
        local_normal = vectors[:, 0]
        if local_normal @ normal < 0:
            local_normal = -local_normal
        residual = float(np.sqrt(max(values[0], 0)))
        planarity = float((values[1] - values[0]) / values[-1])
        tilt = math.degrees(math.acos(float(np.clip(local_normal @ normal, -1, 1))))
        if (
            residual <= p["maximum_surface_residual"]
            and planarity >= p["minimum_planarity"]
            and p["minimum_tilt_degrees"] <= tilt <= p["maximum_tilt_degrees"]
        ):
            selected.append((index, tilt, residual, planarity, len(neighbors)))
    selected_indices = np.array([row[0] for row in selected], dtype=np.int64)
    return {
        "domain_points": domain,
        "voxel_centers": centers,
        "voxel_counts": voxel_counts,
        "selected_indices": selected_indices,
        "selected_points": centers[selected_indices] if len(selected_indices) else np.empty((0, 3)),
        "tilt_degrees": np.array([row[1] for row in selected], dtype=float),
        "residuals": np.array([row[2] for row in selected], dtype=float),
        "planarity": np.array([row[3] for row in selected], dtype=float),
        "neighbor_counts": np.array([row[4] for row in selected], dtype=np.int32),
        "recipe": p,
    }


def cluster_tilted_samples(samples: dict, support: dict, recipe: dict | None = None) -> list[dict]:
    """Cluster tilted samples and reject tiny or implausibly large fragments."""
    p = deepcopy(TILTED_RECIPE if recipe is None else recipe)
    points = samples["selected_points"]
    if not len(points):
        return []
    tree = cKDTree(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    normal = np.asarray(support["plane"][:3], dtype=float)
    offset = float(support["plane"][3])
    components = []
    for start in range(len(points)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for nxt in tree.query_ball_point(points[current, :2], p["cluster_radius"]):
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        if len(members) < p["minimum_cluster_points"]:
            continue
        indices = np.asarray(members, dtype=np.int64)
        xyz = points[indices]
        centered = xyz[:, :2] - xyz[:, :2].mean(axis=0)
        values = np.linalg.eigvalsh(np.cov(centered, rowvar=False)) if len(xyz) > 1 else np.zeros(2)
        spans = 2 * np.sqrt(np.maximum(values, 0))
        major, minor = float(spans[-1]), float(spans[0])
        heights = xyz @ normal + offset
        height_span = float(np.ptp(heights))
        area = major * minor
        if (
            major < p["minimum_major_span"]
            or minor < p["minimum_minor_span"]
            or height_span < p["minimum_height_span"]
            or area > p["maximum_cluster_area"]
        ):
            continue
        score = float(len(xyz) * np.median(samples["planarity"][indices]) * min(height_span / 0.15, 1.0))
        components.append({
            "sample_indices": indices,
            "points": xyz,
            "centroid": xyz.mean(axis=0),
            "point_count": len(xyz),
            "major_span": major,
            "minor_span": minor,
            "height_span": height_span,
            "minimum_height": float(np.min(heights)),
            "median_height": float(np.median(heights)),
            "maximum_height": float(np.max(heights)),
            "median_tilt_degrees": float(np.median(samples["tilt_degrees"][indices])),
            "median_planarity": float(np.median(samples["planarity"][indices])),
            "score": score,
        })
    components.sort(key=lambda row: (-row["score"], tuple(row["centroid"])))
    return components


def select_traversable_components(components: list[dict], maximum_median_tilt_degrees: float = 10.0) -> list[dict]:
    """Keep shallow components compatible with the current traversable-ramp policy.

    The threshold is a fixed physical candidate gate, not an image-color rule.
    It is intentionally exposed in reports and must be validated on a new set.
    """
    if not 5.0 <= maximum_median_tilt_degrees <= 20.0:
        raise ValueError("Traversable tilt limit must be between 5 and 20 degrees")
    return [row for row in components if row["median_tilt_degrees"] <= maximum_median_tilt_degrees]


def select_ground_connected_ramp_components(
    components: list[dict],
    maximum_median_tilt_degrees: float = 10.0,
    maximum_minimum_height: float = 0.05,
    minimum_median_height: float = 0.02,
    minimum_height_span: float = 0.10,
    maximum_satellite_distance: float = 1.5,
) -> list[dict]:
    """Keep shallow components that start near support and rise materially.

    This rejects elevated cargo/equipment faces and tiny floor-edge fragments.
    The values are physical candidate gates and remain subject to independent
    validation; they are not image semantics or human truth.
    """
    if not 5.0 <= maximum_median_tilt_degrees <= 20.0:
        raise ValueError("Traversable tilt limit must be between 5 and 20 degrees")
    if not 0.0 <= maximum_minimum_height <= 0.20:
        raise ValueError("Ground connection height must be between 0 and 0.20 metres")
    if not 0.0 <= minimum_median_height <= 0.20:
        raise ValueError("Ramp median height must be between 0 and 0.20 metres")
    if not 0.03 <= minimum_height_span <= 0.50:
        raise ValueError("Ramp rise must be between 0.03 and 0.50 metres")
    if not 0.2 <= maximum_satellite_distance <= 3.0:
        raise ValueError("Ramp satellite distance must be between 0.2 and 3.0 metres")
    shallow_grounded = []
    for row in components:
        minimum = row.get("minimum_height")
        median = row.get("median_height")
        span = row.get("height_span")
        if (
            not isinstance(minimum, (int, float))
            or not isinstance(median, (int, float))
            or not isinstance(span, (int, float))
        ):
            raise ValueError("Ground-connected selection requires component height bounds")
        if (
            row["median_tilt_degrees"] <= maximum_median_tilt_degrees
            and minimum <= maximum_minimum_height
        ):
            shallow_grounded.append(row)
    anchors = [
        row
        for row in shallow_grounded
        if row["height_span"] >= minimum_height_span and row["median_height"] >= minimum_median_height
    ]
    if not anchors:
        return []
    for row in shallow_grounded:
        centroid = np.asarray(row.get("centroid"), dtype=float)
        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            raise ValueError("Ground-connected selection requires finite component centroids")
    anchor_centroids = [np.asarray(row["centroid"], dtype=float) for row in anchors]
    anchor_ids = {id(row) for row in anchors}
    return [
        row
        for row in shallow_grounded
        if id(row) in anchor_ids
        or (
            row["height_span"] < minimum_height_span
            and min(np.linalg.norm(np.asarray(row["centroid"], dtype=float) - center) for center in anchor_centroids)
            <= maximum_satellite_distance
        )
    ]


def _representative_pixels(pixels: np.ndarray, maximum_points: int = 3) -> np.ndarray:
    """Choose deterministic in-component points with center and extent coverage."""
    if not len(pixels) or maximum_points < 1:
        return np.empty((0, 2), dtype=np.int32)
    center = np.median(pixels, axis=0)
    selected = [int(np.argmin(np.linalg.norm(pixels - center, axis=1)))]
    while len(selected) < min(maximum_points, len(pixels)):
        distances = np.min(
            np.stack([np.linalg.norm(pixels - pixels[index], axis=1) for index in selected]), axis=0
        )
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return pixels[selected]


def component_camera_hints(
    points: np.ndarray,
    components: list[dict],
    camera: dict,
    maximum_hints: int = 6,
    maximum_points_per_hint: int = 5,
) -> list[dict]:
    """Project components through the full-cloud visibility cohort into a camera."""
    source, pixels = project_visible(points, camera, cell_size=4, depth_slack=0.12, near=0.2)
    visible_points = points[source]
    if not len(visible_points):
        return []
    tree = cKDTree(visible_points)
    hints = []
    for component_index, component in enumerate(components):
        # A visible point within one voxel diagonal binds a projection to the
        # selected component without using any image annotation.
        member_lists = tree.query_ball_point(component["points"], r=TILTED_RECIPE["voxel_xy"] * 1.8)
        visible_rows = sorted({row for rows in member_lists for row in rows})
        if len(visible_rows) < 2:
            continue
        uv = pixels[visible_rows]
        prompt_pixels = _representative_pixels(uv, maximum_points=maximum_points_per_hint)
        seed = prompt_pixels[0]
        hints.append({
            "component_index": component_index,
            "x": int(seed[0]),
            "y": int(seed[1]),
            "visible_points": len(visible_rows),
            "pixel_bbox": [int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max())],
            "component_score": component["score"],
            "points": [{"x": int(pixel[0]), "y": int(pixel[1]), "label": 1} for pixel in prompt_pixels],
        })
    hints.sort(key=lambda row: (-row["component_score"], -row["visible_points"], row["y"], row["x"]))
    return hints[:maximum_hints]
