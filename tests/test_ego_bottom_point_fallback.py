from tools.diagnose_ego_bottom_point_fallback import fixed_bottom_points, guard_profile


def record(bbox, area):
    return {"bbox": bbox, "area": area}


def test_guard_accepts_wide_bottom_chassis_geometry():
    assert guard_profile(
        record([0, 399, 523, 512], 41_214), 0.875, 640, 512
    ) == "wide_bottom_chassis"


def test_guard_accepts_centered_fork_carriage_geometry():
    assert guard_profile(
        record([583, 947, 1435, 1536], 365_700), 0.92, 1920, 1536
    ) == "centered_fork_carriage"


def test_guard_rejects_floor_like_large_mask():
    assert guard_profile(
        record([0, 200, 640, 512], 180_000), 0.99, 640, 512
    ) is None


def test_guard_rejects_tall_side_occlusion():
    assert guard_profile(
        record([1350, 700, 1920, 1536], 250_000), 0.95, 1920, 1536
    ) is None


def test_guard_rejects_small_bottom_fragment():
    assert guard_profile(
        record([250, 470, 330, 512], 2_000), 0.99, 640, 512
    ) is None


def test_fixed_bottom_points_are_deterministic_unique_and_in_bounds():
    first = fixed_bottom_points(640, 512)
    second = fixed_bottom_points(640, 512)

    assert first == second
    assert len(first) == 18
    assert len({(point["x"], point["y"]) for point in first}) == 18
    assert all(0 <= point["x"] < 640 and 0 <= point["y"] < 512
               for point in first)
