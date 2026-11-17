"""Regression tests for bounded editing and persisted agent decisions."""
import io
import base64
import copy
import json
import tempfile
import threading
import unittest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from source.agent import mcap_video, server, workflow
from source.agent.edit_agent import (_hole_regions, _candidate_gap_regions, _shape_from_mask,
                                     _rasterize, _valid_region, apply_operations, valid_shape)


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(output, "JPEG")
    return output.getvalue()


def shape(shape_id=1, label="a", left=2, right=18):
    points = [[left, 2], [right, 2], [right, 18], [left, 18]]
    part = {"points": points, "holes": []}
    return {"id": shape_id, "label": label, "points": points,
            "holes": [], "parts": [part]}


def sam_result():
    return {"parts": shape()["parts"], "score": 0.9}


class EditTests(unittest.TestCase):
    def test_open_line_gap_is_detected_and_fill_preserves_excluded_object(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:95, 5:95] = 1
        mask[35:65, 35:65] = 0  # Excluded robot.
        mask[5:40, 48:52] = 0  # Thin line connected to robot and background.
        candidate = _shape_from_mask(mask, "地面", 1)
        self.assertFalse(_hole_regions(candidate, 100, 100))
        gaps = _candidate_gap_regions(candidate, 100, 100)
        gap = next(region for region in gaps if region["mask"][20, 50])
        self.assertEqual(gap["kind"], "open_gap")
        updated, applied, errors = apply_operations([candidate], [{
            "operation": "fill_gap", "shape_ids": [1], "hole_id": gap["id"],
            "polygon": [[0, 0], [99, 0], [99, 99], [0, 99]]}], ["地面"], 100, 100)
        self.assertFalse(errors)
        result = _rasterize(updated[0], 100, 100)
        self.assertEqual(result[20, 50], 1)
        self.assertEqual(result[50, 50], 0)
        added = (result != 0) & (mask == 0)
        self.assertTrue(np.all(~added | (gap["mask"] != 0)))
        self.assertTrue(np.all(result[mask != 0] == 1))
        self.assertEqual(applied[0]["filled_pixels"], int(added.sum()))

    def test_partial_gap_edits_share_observation_ids_and_preserve_other_holes(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[15:30, 15:30] = 0
        mask[60:80, 60:80] = 0
        candidate = _shape_from_mask(mask, "地面", 1)
        self.assertTrue(np.array_equal(mask, _rasterize(candidate, 100, 100)))
        gap = _hole_regions(candidate, 100, 100)[0]
        operations = [{"operation": "fill_gap", "shape_ids": [1], "hole_id": gap["id"],
                       "polygon": polygon} for polygon in (
                           [[15, 15], [21, 15], [21, 29], [15, 29]],
                           [[22, 15], [29, 15], [29, 29], [22, 29]])]
        result, applied, errors = apply_operations([candidate], operations, ["地面"], 100, 100)
        self.assertFalse(errors)
        self.assertEqual(len(applied), 2)
        expected = mask.copy()
        expected[15:30, 15:30] = 1
        self.assertTrue(np.array_equal(expected, _rasterize(result[0], 100, 100)))

    def test_shape_and_polygon_validation(self):
        self.assertTrue(valid_shape(shape(), 40, 40))
        invalid = shape()
        invalid["parts"][0]["holes"] = [[[0, 0], [float("inf"), 0], [0, 2]]]
        self.assertFalse(valid_shape(invalid, 40, 40))
        updated, applied, queue = apply_operations([shape()], [
            {"operation": "add_region", "shape_ids": [1],
             "polygon": [[0, 0], [40, 0], [0, 2]]}], ["a"], 40, 40)
        self.assertEqual(updated, [shape()])
        self.assertFalse(applied)
        self.assertTrue(queue)
        self.assertFalse(_valid_region([[5, 5], [5, 5], [5, 5]], 40, 40))
        self.assertFalse(_valid_region([[2, 2], [5, 5], [8, 8]], 40, 40))
        self.assertTrue(_valid_region([[2, 2], [8, 2], [8, 8]], 40, 40))

    def test_merge_and_split_preserve_pixels(self):
        original = [shape(), shape(2, left=22, right=35)]
        merged, applied, queue = apply_operations(original, [
            {"operation": "merge", "shape_ids": [1, 2], "label": "a"}],
            ["a"], 40, 40, next_id=3)
        self.assertFalse(queue)
        self.assertEqual([item["id"] for item in merged], [3])
        self.assertEqual(int(_rasterize(merged[0], 40, 40).sum()),
                         sum(int(_rasterize(item, 40, 40).sum()) for item in original))
        split, applied, queue = apply_operations([shape()], [
            {"operation": "split", "shape_ids": [1],
             "polygon": [[0, 0], [10, 0], [10, 39], [0, 39]]}],
            ["a"], 40, 40, next_id=2)
        self.assertFalse(queue)
        self.assertEqual([item["id"] for item in split], [1, 2])
        self.assertEqual(sum(int(_rasterize(item, 40, 40).sum()) for item in split),
                         int(_rasterize(shape(), 40, 40).sum()))

    def test_fill_gap_polygon_is_clipped_to_the_named_hole(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [], "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                                           "holes": []}]}
        donut["holes"] = [[[10, 10], [20, 10], [20, 20], [10, 20]]]
        donut["parts"][0]["holes"] = donut["holes"]
        hole = _hole_regions(donut, 40, 40)[0]
        updated, applied, queue = apply_operations([donut], [{
            "operation": "fill_gap", "shape_ids": [1], "hole_id": hole["id"],
            "polygon": [[5, 5], [25, 5], [25, 25], [5, 25]]}], ["地面"], 40, 40)
        self.assertFalse(queue)
        self.assertEqual(applied[0]["filled_pixels"], int(hole["mask"].sum()))
        result = _rasterize(updated[0], 40, 40)
        self.assertEqual(result[15, 15], 1)
        before = _rasterize(donut, 40, 40)
        added = (result != 0) & (before == 0)
        expected_added = (hole["mask"] != 0) & (before == 0)
        self.assertEqual(int(added.sum()), int(expected_added.sum()))
        self.assertTrue((added <= (hole["mask"] != 0)).all())

    def test_rejected_edits_leave_original(self):
        for operation in (
                {"operation": "review", "shape_ids": [1]},
                {"operation": "merge", "shape_ids": [1, 1], "label": "a"},
                {"operation": "erase_region", "shape_ids": [1],
                 "polygon": [[0, 0], [39, 0], [39, 39], [0, 39]]},
                {"operation": "remove", "shape_ids": [99]}):
            with self.subTest(operation=operation):
                updated, applied, queue = apply_operations([shape()], [operation],
                    ["a"], 40, 40)
                self.assertEqual(updated, [shape()])
                self.assertFalse(applied)
                self.assertTrue(queue)

    def test_large_traversable_surface_is_not_reclassified_as_instance(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "Foreground chassis is visible",
        }], ["floor", "ego"], 40, 40)
        self.assertEqual(updated[0]["label"], "floor")
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "global_surface_reclassify_requires_human_review")
        self.assertGreater(queue[0]["area_ratio"], 0.35)

    def test_small_surface_candidate_can_still_be_reclassified(self):
        candidate = shape(label="floor", left=2, right=12)
        updated, applied, queue = apply_operations([candidate], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "The entire small candidate is host chassis",
        }], ["floor", "ego"], 40, 40)
        self.assertEqual(updated[0]["label"], "ego")
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_large_floor_erase_is_preserved_for_human_review(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "erase_region", "shape_ids": [1],
            "polygon": [[0, 0], [39, 0], [39, 20], [0, 20]],
            "reason": "remove a suspected foreground object",
        }], ["floor"], 40, 40)
        self.assertEqual(updated, [large_floor])
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "global_surface_large_erase_requires_human_review")
        self.assertGreater(queue[0]["removed_fraction"], 0.25)

    def test_small_floor_erase_remains_available(self):
        large_floor = shape(label="floor", left=1, right=38)
        large_floor["points"] = [[1, 1], [38, 1], [38, 38], [1, 38]]
        large_floor["parts"] = [{"points": large_floor["points"], "holes": []}]
        updated, applied, queue = apply_operations([large_floor], [{
            "operation": "erase_region", "shape_ids": [1],
            "polygon": [[1, 1], [5, 1], [5, 5], [1, 5]],
            "reason": "remove a bounded local foreground object",
        }], ["floor"], 40, 40)
        self.assertNotEqual(updated, [large_floor])
        self.assertTrue(applied)
        self.assertFalse(queue)

    def test_unreferenced_high_risk_reclassification_requires_human_review(self):
        person = shape(label="person")
        updated, applied, queue = apply_operations([person], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "foreground host structure",
        }], ["person", "ego"], 40, 40, allow_high_risk_reclassify=False)
        self.assertEqual(updated, [person])
        self.assertFalse(applied)
        self.assertEqual(queue[0]["reason_code"],
                         "unreferenced_high_risk_reclassify_requires_human_review")

    def test_reference_backed_high_risk_reclassification_remains_available(self):
        person = shape(label="person")
        updated, applied, queue = apply_operations([person], [{
            "operation": "reclassify", "shape_ids": [1], "label": "ego",
            "reason": "human reference identifies host chassis",
        }], ["person", "ego"], 40, 40, allow_high_risk_reclassify=True)
        self.assertEqual(updated[0]["label"], "ego")
        self.assertTrue(applied)
        self.assertFalse(queue)


class WorkflowTests(unittest.TestCase):
    def test_local_gap_evidence_shows_original_binary_and_bounded_image_count(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[5:95, 48:52] = 0
        candidate = _shape_from_mask(mask, "地面", 1)
        gaps = workflow.candidate_holes([candidate], ["地面"], 100, 100)
        output = io.BytesIO()
        Image.new("RGB", (100, 100), "yellow").save(output, "JPEG")
        sheets = workflow.gap_detail_images(output.getvalue(), [candidate], gaps)
        self.assertEqual(len(sheets), 1)
        sheet, regions = sheets[0]
        self.assertTrue(regions)
        self.assertEqual(regions[0]["hole_id"], gaps[0]["id"])
        self.assertLessEqual(len(regions), 4)
        picture = Image.open(io.BytesIO(sheet))
        self.assertEqual(picture.width, 1080)

    def test_description_uses_original_image_and_separate_outline(self):
        original = image_bytes()
        marked = io.BytesIO()
        Image.new("RGB", (40, 40), (255, 0, 255)).save(marked, "JPEG")
        with patch.object(workflow, "chat", return_value={"description": "地面可见边界",
                "prompts": [{"label": "地面", "text": "floor"}]}) as chat:
            workflow.describe({}, original, [shape(label="地面")], ["地面"], marked.getvalue())
        self.assertEqual(chat.call_args.args[2], original)
        self.assertEqual(chat.call_args.kwargs["extra_images"], (marked.getvalue(),))
        self.assertIn("轮廓颜色只是界面标记", chat.call_args.args[1])
        self.assertIn("仅依据图像 1", chat.call_args.args[1])

    def test_visible_label_blind_review_keeps_operator_person_and_viewpoint_non_ego(self):
        labels = ["ego", "other_robot", "person"]
        answer = {"assessments": [
            {"label": label, "evidence": "absent", "confidence": 0.9,
             "reason": "No visible object."} for label in labels], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            workflow.discover_visible_labels({}, image_bytes(), labels)
        instruction = chat.call_args.args[1]
        self.assertIn("distinct from any human operator", instruction)
        self.assertIn("a visible human body", instruction)
        self.assertIn("camera viewpoint or camera position alone is never evidence", instruction)

    def test_visible_label_discovery_filters_known_labels_and_routes_unknown_to_review(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "present", "confidence": 0.94,
             "reason": "A standing person is visible."},
            {"label": "pallet", "evidence": "absent", "confidence": 0.91,
             "reason": "No pallet is visible."},
            {"label": "slope", "evidence": "uncertain", "confidence": 0.52,
             "reason": "The distant surface is partly occluded."}],
            "unknown_candidates": [{"candidate_name": "rack", "confidence": 0.72,
                                    "evidence": "A large storage rack is visible."},
                                   {"candidate_name": "person", "confidence": 0.91,
                                    "evidence": "A person is visible."}]}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet", "slope"])
        self.assertEqual(result["visible_labels"], ["person"])
        self.assertEqual(result["uncertain_labels"], ["slope"])
        self.assertEqual(result["candidate_labels"], ["person", "slope"])
        self.assertEqual(result["consistency_flags"], [])
        self.assertFalse(result["registry_mutated"])
        self.assertEqual(result["unknown_candidates"][0]["route"], "human_review_required")
        self.assertFalse(result["unknown_candidates"][0]["ontology_mutation_allowed"])
        self.assertEqual(result["registry_conflicts"][0]["matching_known_label"], "person")
        self.assertEqual(result["registry_conflicts"][0]["reason_code"], "known_label_alias_conflict")
        self.assertIn("assess every known label", chat.call_args.args[1])
        discovery_schema = chat.call_args.kwargs["response_schema"]
        self.assertEqual(discovery_schema["properties"]["assessments"]["minItems"], 3)
        self.assertEqual(discovery_schema["properties"]["assessments"]["items"]
                         ["properties"]["reason"]["maxLength"], 72)

    def test_visible_label_discovery_uses_registry_sized_compact_budget(self):
        labels = ["车辆", "pallet", "person", "floor", "slope", "ego",
                  "other_robot", "cargo_floor"]
        answer = {"assessments": [
            {"label": label, "evidence": "absent", "confidence": 0.9,
             "reason": "No visible instance."} for label in labels],
            "unknown_candidates": []}
        config = {"llm_prompt_profile": "compact"}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            workflow.discover_visible_labels(config, image_bytes(), labels)
        self.assertEqual(chat.call_args.args[0]["llm_max_tokens"], 640)
        self.assertNotIn("llm_max_tokens", config)
        self.assertIn("at most 10 words", chat.call_args.args[1])
        self.assertIn("no indentation or extra whitespace", chat.call_args.args[1])
        self.assertEqual(workflow.compact_discovery_token_budget(config, 8), 640)
        self.assertEqual(workflow.compact_discovery_token_budget(config, 20), 1024)
        self.assertEqual(workflow.compact_discovery_token_budget(
            {**config, "llm_max_tokens": 704}, 8), 704)

    def test_visible_label_discovery_routes_contradictory_absent_to_uncertain_candidate(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "absent", "confidence": 0.8,
             "reason": "A person is visible in the image, but the label is not present in the list of known labels."},
            {"label": "pallet", "evidence": "absent", "confidence": 0.9,
             "reason": "No pallet is visible."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet"])
        self.assertEqual(result["visible_labels"], [])
        self.assertEqual(result["uncertain_labels"], ["person"])
        self.assertEqual(result["candidate_labels"], ["person"])
        self.assertEqual(result["assessments"][0]["evidence"], "uncertain")
        self.assertEqual(result["consistency_flags"], [{
            "label": "person", "original_evidence": "absent", "normalized_evidence": "uncertain",
            "reason_code": "vlm_visibility_contradiction", "route": "human_review_required",
            "reason": "A person is visible in the image, but the label is not present in the list of known labels."}])

    def test_visible_label_discovery_routes_duplicate_and_missing_registry_labels(self):
        answer = {"assessments": [
            {"label": "person", "evidence": "present", "confidence": 0.9, "reason": "visible"},
            {"label": "person", "evidence": "absent", "confidence": 0.8, "reason": "duplicate"}],
            "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["person", "pallet"])
        self.assertEqual(result["uncertain_labels"], ["person", "pallet"])
        self.assertEqual(result["visible_labels"], [])
        self.assertTrue(any(flag["reason_code"] == "vlm_duplicate_registry_assessment"
                            for flag in result["consistency_flags"]))

    def test_visible_label_discovery_downgrades_shared_vehicle_robot_anchor(self):
        labels = ["车辆", "ego", "other_robot"]
        answer = {"assessments": [
            {"label": "车辆", "evidence": "present", "confidence": 0.9,
             "reason": "A robot is visible near [700,450]."},
            {"label": "ego", "evidence": "absent", "confidence": 0.9,
             "reason": "No ego robot is visible."},
            {"label": "other_robot", "evidence": "present", "confidence": 0.9,
             "reason": "Another robot is visible at [700,450]."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), labels)
        self.assertEqual(result["visible_labels"], [])
        self.assertEqual(result["uncertain_labels"], ["车辆", "other_robot"])
        flags = [flag for flag in result["consistency_flags"]
                 if flag["reason_code"] == "vlm_cross_label_entity_conflict"]
        self.assertEqual({flag["label"] for flag in flags}, {"车辆", "other_robot"})
        self.assertEqual(flags[0]["conflicting_labels"], ["车辆", "other_robot"])

    def test_visible_label_discovery_ignores_other_object_and_negated_anchors(self):
        labels = ["车辆", "other_robot"]
        answers = [
            ["A vehicle is visible at [300,450], not the robot at [700,450].",
             "A robot is visible at [700,450]."],
            ["A vehicle at [300,450] is left of the robot at [700,450].",
             "A robot is visible at [700,450]."],
        ]
        for reasons in answers:
            with self.subTest(reasons=reasons):
                answer = {"assessments": [
                    {"label": label, "evidence": "present", "confidence": 0.9, "reason": reason}
                    for label, reason in zip(labels, reasons)], "unknown_candidates": []}
                with patch.object(workflow, "chat", return_value=answer):
                    result = workflow.discover_visible_labels({}, image_bytes(), labels)
                self.assertEqual(result["visible_labels"], labels)
                self.assertFalse(any(flag["reason_code"] == "vlm_cross_label_entity_conflict"
                                     for flag in result["consistency_flags"]))

    def test_visible_label_discovery_keeps_distinct_vehicle_robot_anchors(self):
        labels = ["车辆", "other_robot"]
        answer = {"assessments": [
            {"label": "车辆", "evidence": "present", "confidence": 0.9,
             "reason": "A vehicle is visible at [300,450]."},
            {"label": "other_robot", "evidence": "present", "confidence": 0.9,
             "reason": "A robot is visible at [700,450]."}], "unknown_candidates": []}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), labels)
        self.assertEqual(result["visible_labels"], labels)
        self.assertFalse(any(flag["reason_code"] == "vlm_cross_label_entity_conflict"
                             for flag in result["consistency_flags"]))

    def test_visible_label_discovery_routes_english_vehicle_alias_to_registry_conflict(self):
        answer = {"assessments": [{"label": "车辆", "evidence": "present",
            "confidence": 0.96, "reason": "A vehicle is visible."}],
            "unknown_candidates": [{"candidate_name": "vehicle", "confidence": 0.95,
                                    "evidence": "A vehicle is visible."}]}
        with patch.object(workflow, "chat", return_value=answer):
            result = workflow.discover_visible_labels({}, image_bytes(), ["车辆"])
        self.assertEqual(result["unknown_candidates"], [])
        self.assertEqual(result["registry_conflicts"], [{
            "candidate_name": "vehicle", "confidence": 0.95, "evidence": "A vehicle is visible.",
            "route": "human_review_required", "ontology_mutation_allowed": False,
            "matching_known_label": "车辆", "reason_code": "known_label_alias_conflict"}])
        self.assertFalse(result["registry_mutated"])

    def test_description_only_uses_reference_truth_label_and_short_prompt(self):
        answer = {"description": "地面边界，不包括障碍物。", "prompts": [
            {"label": "person", "text": "A warehouse person in clothing"},
            {"label": "pallet", "text": "wooden warehouse pallet"},
            {"label": "地面", "text": "The visible warehouse floor surface around pallets and people"}]}
        with patch.object(workflow, "chat", return_value=answer) as chat:
            result = workflow.describe({}, image_bytes(), [shape(label="地面")],
                                       ["person", "pallet", "地面"])
        self.assertEqual(result["prompts"], [{"label": "地面", "text": "floor"}])
        self.assertEqual(result["description"], "地面边界，不包括障碍物。")
        instruction = chat.call_args.args[1]
        self.assertIn('允许类别：["地面"]', instruction)
        self.assertNotIn('允许类别：["person"', instruction)

    def test_description_grounds_rules_in_confirmed_reference_geometry(self):
        reference = shape(label="地面")
        reference["source"] = "automatic"
        reference["parts"][0]["holes"] = [[[5, 5], [8, 5], [8, 8], [5, 8]]]
        with patch.object(workflow, "chat", return_value={"description": "参考地面区域",
                "prompts": [{"label": "地面", "text": "floor"}]}) as chat:
            workflow.describe({}, image_bytes(), [reference], ["地面"])
        instruction = chat.call_args.args[1]
        for principle in ("参考帧真值优先", "先比对，再描述", "必须有标注证据",
                          "不擅自改变区域归属", "谨慎解释未标注区域", "提炼可迁移规则",
                          "输出前检查一致性", "分别生成完整描述和 SAM 3 关键词"):
            self.assertIn(principle, instruction)
        self.assertIn("地面框线归属规则", instruction)
        self.assertIn("无论颜色", instruction)
        self.assertIn("当前可见且没有被物体遮挡", instruction)
        self.assertIn("无论最初来自人工标注还是自动标注", instruction)
        self.assertIn("不得预设所有标线都包含或排除", instruction)
        self.assertIn("证据不足时省略未经证实的规则", instruction)
        self.assertIn("参考帧未出现的独立物体默认按非地面语义处理", instruction)
        self.assertIn("黄色地面标线纳入地面、将机器人排除在地面之外", instruction)
        self.assertIn(json.dumps([reference], ensure_ascii=False), instruction)

    def test_review_has_original_reference_and_geometry_on_every_round(self):
        marked = io.BytesIO()
        Image.new("RGB", (40, 40), "magenta").save(marked, "JPEG")
        reference = {"image": image_bytes(), "marked_image": marked.getvalue(),
                     "annotations": [shape(label="a")]}
        decisions = [{"status": "continue", "operations": [{"operation": "reclassify",
            "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=decisions) as chat:
            result = workflow.run({}, image_bytes(), {"description": "reference rule",
                "prompts": [{"label": "a", "text": "target"}]}, ["a", "b"],
                reference=reference)
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(chat.call_count, 2)
        for call in chat.call_args_list:
            self.assertEqual(call.kwargs["extra_images"][2:],
                             (reference["image"], reference["marked_image"]))
            self.assertIn(json.dumps(workflow.reference_geometry(reference["annotations"]), ensure_ascii=False), call.args[1])
            self.assertIn("如与参考真值证据冲突，以参考真值为准", call.args[1])
            self.assertIn("不得直接照搬参考帧坐标", call.args[1])

    def test_review_prompt_allows_local_sam_points_and_opencv(self):
        with patch.object(workflow, "chat", return_value={"status": "done", "operations": []}) as chat:
            workflow.review({}, image_bytes(), image_bytes(),
                {"description": "floor only", "prompts": [{"label": "a", "text": "floor"}]},
                [], ["a"], [])
        instruction = chat.call_args.args[1]
        for principle in ("复查阶段允许使用 SAM 3", "只能对现有候选的局部问题使用 sam_points",
                          "禁止使用 sam_text 对整幅图像重新分割", "保留已正确标注的区域",
                          "不得仅凭参考帧与当前帧坐标或面积不同",
                          "修改范围限于需要修复的区域", "无法可靠判断或准确定位时",
                          "SAM 3 点提示结果必须由下一轮 GPT"):
            self.assertIn(principle, instruction)
        self.assertNotIn("- sam_text:", instruction)
        self.assertIn("- sam_points:", instruction)
        self.assertIn("不修改候选并返回 done", instruction)
        self.assertIn("发现漏标时使用局部 sam_points 或 add_region 修复，不能返回 done", instruction)
        self.assertIn("线条颜色不能作为扣除依据", instruction)
        self.assertNotIn('"status":"continue|done|needs_review"', instruction)

    def test_review_requires_assessment_for_each_detected_gap(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]],
                 "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                            "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
        holes = workflow.candidate_holes([donut], ["地面"], 40, 40)
        with patch.object(workflow, "chat", return_value={"status": "done", "operations": []}) as chat:
            with self.assertRaisesRegex(ValueError, "逐一判断"):
                workflow.review({}, image_bytes(), image_bytes(),
                    {"description": "floor", "prompts": []}, [], ["地面"], [], holes=holes)
        self.assertEqual(chat.call_count, 3)
        self.assertIn("上一份输出未通过 Agent 协议校验", chat.call_args.args[1])

    def test_review_retries_protocol_error_and_returns_corrected_decision(self):
        donut = {"id": 1, "label": "地面", "points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                 "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]],
                 "parts": [{"points": [[2, 2], [30, 2], [30, 30], [2, 30]],
                            "holes": [[[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
        holes = workflow.candidate_holes([donut], ["地面"], 40, 40)
        invalid = {"status": "done", "hole_assessment": [{"hole_id": "#5",
            "semantic": "ground", "reason": "floor"}], "operations": []}
        valid = {"status": "done", "hole_assessment": [{"hole_id": hole["id"],
            "semantic": "non_ground", "reason": "visible object"} for hole in holes],
            "operations": []}
        with patch.object(workflow, "chat", side_effect=[invalid, valid]) as chat:
            decision = workflow.review({}, image_bytes(), image_bytes(),
                {"description": "floor", "prompts": []}, [], ["地面"], [], holes=holes)
        self.assertEqual(decision, valid)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("GPT 空洞判断必须覆盖全部 hole_id", chat.call_args.args[1])
        self.assertIn('"hole_id": "#5"', chat.call_args.args[1])

    def test_omitted_hole_is_preserved_and_routed_to_human_review(self):
        holes = [
            {"id": "1:hole:1:1:2:2", "shape_id": 1},
            {"id": "1:hole:3:3:4:4", "shape_id": 1},
        ]
        decision = workflow._validate_review_result({
            "status": "done",
            "reason": "Only one region was clear",
            "hole_assessment": [{"hole_id": holes[0]["id"],
                                 "semantic": "non_ground", "reason": "Visible obstacle"}],
            "operations": [],
        }, holes, [shape()], ["a"], 40, 40)
        repaired = {item["hole_id"]: item for item in decision["hole_assessment"]}
        self.assertEqual(repaired[holes[1]["id"]]["semantic"], "non_ground")
        self.assertEqual(decision["protocol_repairs"], [{
            "reason_code": "vlm_hole_assessment_omitted",
            "hole_id": holes[1]["id"], "route": "human_review_required"}])

    def test_omitted_hole_cannot_be_filled_and_unknown_ids_still_fail(self):
        holes = [
            {"id": "1:hole:1:1:2:2", "shape_id": 1},
            {"id": "1:hole:3:3:4:4", "shape_id": 1},
        ]
        fill = {"operation": "fill_gap", "shape_ids": [1],
                "hole_id": holes[1]["id"],
                "polygon": [[3, 3], [4, 3], [4, 4]], "reason": "Fill floor"}
        partial = {"status": "continue", "reason": "Partial answer",
                   "hole_assessment": [{"hole_id": holes[0]["id"],
                                        "semantic": "non_ground", "reason": "Obstacle"}],
                   "operations": [fill]}
        with self.assertRaisesRegex(ValueError, "只能修补判定为 ground"):
            workflow._validate_review_result(partial, holes, [shape()], ["a"], 40, 40)
        unknown = copy.deepcopy(partial)
        unknown["operations"] = []
        unknown["hole_assessment"].append(
            {"hole_id": "unknown", "semantic": "non_ground", "reason": "Unknown"})
        with self.assertRaisesRegex(ValueError, "无效 ID"):
            workflow._validate_review_result(unknown, holes, [shape()], ["a"], 40, 40)

    def test_omitted_hole_repair_finishes_as_review_required_without_discarding_candidates(self):
        decision = {"status": "done", "reason": "One region was omitted",
                    "hole_assessment": [], "operations": [],
                    "protocol_repairs": [{
                        "reason_code": "vlm_hole_assessment_omitted",
                        "hole_id": "1:hole:3:3:4:4",
                        "route": "human_review_required"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "review", return_value=decision):
            result = workflow.run({}, image_bytes(), {"description": "class a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "review_required")
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(len(result["shapes"]), 1)
        self.assertEqual(result["review_queue"][0]["reason_code"],
                         "vlm_hole_assessment_omitted")

    def test_empty_continue_is_normalized_to_done_but_unfilled_ground_is_not(self):
        empty = workflow._validate_review_result(
            {"status": "continue", "reason": "No edit remains", "operations": []},
            [], [shape()], ["a"], 40, 40)
        self.assertEqual(empty["status"], "done")
        self.assertEqual(empty["protocol_normalizations"][0]["reason_code"],
                         "empty_continue_normalized_to_done")
        hole = {"id": "1:hole:1:1:2:2", "shape_id": 1}
        pending = workflow._validate_review_result({
            "status": "continue", "reason": "Ground remains",
            "hole_assessment": [{"hole_id": hole["id"], "semantic": "ground",
                                 "reason": "Visible ground"}],
            "operations": [],
        }, [hole], [shape()], ["a"], 40, 40)
        self.assertEqual(pending["status"], "continue")
        self.assertNotIn("protocol_normalizations", pending)

    def test_review_retries_unknown_operation_and_missing_tool_fields(self):
        operation = {"operation": "add_region", "shape_ids": [1],
                     "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair floor"}
        corrected = {"status": "continue", "operations": [operation]}
        cases = [
            ({"status": "continue", "operations": [{**operation, "operation": "fill_hole"}]}, "操作名无效"),
            ({"status": "continue", "operations": [{key: value for key, value in operation.items()
                                                    if key != "shape_ids"}]}, "shape_ids"),
            ({"status": "continue", "operations": [{key: value for key, value in operation.items()
                                                    if key != "polygon"}]}, "polygon"),
            ({"status": "continue", "operations": [{**operation, "shape_ids": [99]}]}, "shape_ids"),
            ({"status": "done", "operations": None}, "operations"),
            ({"status": "done", "operations": {}}, "operations"),
        ]
        for invalid, error in cases:
            with self.subTest(error=error), \
                 patch.object(workflow, "sam_text", return_value=[sam_result()]), \
                 patch.object(workflow, "chat", side_effect=[
                     invalid, corrected, {"status": "done", "operations": []}]) as chat:
                result = workflow.run({}, image_bytes(), {"description": "floor",
                    "prompts": [{"label": "a", "text": "floor"}]}, ["a"], 2)
                self.assertTrue(result["agent"]["converged"])
                self.assertEqual(result["agent"]["rounds"], 2)
                self.assertEqual(len(result["applied"]), 1)
                self.assertIn(error, chat.call_args_list[1].args[1])
                self.assertIn(json.dumps(invalid, ensure_ascii=False), chat.call_args_list[1].args[1])
                self.assertEqual(chat.call_args_list[0].kwargs["extra_images"],
                                 chat.call_args_list[1].kwargs["extra_images"])
                mask = _rasterize(result["shapes"][0], 40, 40)
                self.assertEqual(mask[5, 23], 1)

    def test_review_retries_invalid_json_but_not_transport_errors(self):
        def response(content):
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch.object(workflow.urllib.request, "urlopen", side_effect=[
                response("invalid json"), response('{"status":"done","operations":[]}')]) as request:
            result = workflow.review({"llm_url": "http://example.test", "llm_model": "test"},
                image_bytes(), image_bytes(), {"description": "a", "prompts": []}, [], ["a"], [])
        self.assertEqual(result["status"], "done")
        self.assertEqual(request.call_count, 2)
        sent = json.loads(request.call_args.args[0].data)["messages"][0]["content"][0]["text"]
        self.assertIn("invalid json", sent)
        self.assertIn("未返回有效 JSON", sent)
        for error in (ValueError("大模型请求超时"), workflow.RunCancelled("用户已停止")):
            with self.subTest(error=error), patch.object(workflow, "chat", side_effect=error) as chat:
                with self.assertRaises(type(error)):
                    workflow.review({}, image_bytes(), image_bytes(),
                        {"description": "a", "prompts": []}, [], ["a"], [])
                self.assertEqual(chat.call_count, 1)
                self.assertIsNone(workflow._review_call.get())

    def test_exhausted_protocol_retries_preserve_candidates_and_trace_rejections(self):
        events = []
        invalid = {"status": "done", "operations": None}
        with workflow.trace_calls(events.append), \
             patch.object(workflow, "sam_text", return_value=[sam_result()]) as sam, \
             patch.object(workflow, "_chat", return_value=invalid) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(sam.call_count, 1)
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 0)
        self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
        self.assertIn("协议校验重试 2 次后仍失败", result["review_queue"][0]["error"])
        self.assertEqual(result["review_queue"][0]["reason_code"],
                         "agent_invalid_model_output")
        self.assertEqual(result["review_queue"][0]["route"], "human_review_required")
        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])
        rejected = [event for event in events if event["status"] == "rejected"]
        self.assertEqual([event["review_attempt"] for event in rejected], [1, 2, 3])
        self.assertEqual([event["retrying"] for event in rejected], [True, True, False])
        self.assertEqual(len({event["id"] for event in rejected}), 3)
        self.assertEqual(rejected[-1]["output"]["parsed"], invalid)
        self.assertIn("operations", rejected[-1]["error"])
        self.assertIsNone(workflow._review_call.get())

    def test_missing_sam_assessment_is_sent_back_for_correction(self):
        with patch.object(workflow, "chat", side_effect=[
                {"status": "done", "operations": []},
                {"status": "done", "operations": [],
                 "sam_assessment": {"improved": False, "reason": "lost correct region"}}]) as chat:
            result = workflow.review({}, image_bytes(), image_bytes(),
                {"description": "a", "prompts": []}, [shape()], ["a"], [],
                sam_comparison={"shape_id": 1, "before_image": image_bytes()})
        self.assertFalse(result["sam_assessment"]["improved"])
        self.assertIn("未评估 SAM 3 点提示结果", chat.call_args.args[1])

    def test_legacy_review_request_ends_without_human_review(self):
        original = shape()
        result = self.run_agent([{"status": "needs_review", "reason": "uncertain",
            "operations": [{"operation": "review", "shape_ids": [1]}]}])
        self.assertEqual(result["agent"]["stop_reason"], "done")
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["review_queue"], [])
        self.assertEqual(result["shapes"][0]["parts"], original["parts"])
        self.assertEqual(result["agent"]["audit"][0]["decision"]["operations"], [])

    def test_legacy_review_with_valid_edit_applies_edit(self):
        result = self.run_agent([{"status": "needs_review", "operations": [
            {"operation": "review", "shape_ids": [1]},
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "done", "operations": []}])
        self.assertEqual(result["agent"]["stop_reason"], "done")
        self.assertEqual(result["shapes"][0]["label"], "b")
        self.assertEqual(result["review_queue"], [])

    def test_multiple_opencv_rounds_change_only_requested_regions_without_sam_reentry(self):
        decisions = [
            {"status": "continue", "operations": [{"operation": "add_region", "shape_ids": [1],
                "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "extend right edge"}]},
            {"status": "continue", "operations": [{"operation": "erase_region", "shape_ids": [1],
                "polygon": [[2, 2], [5, 2], [5, 18], [2, 18]], "reason": "trim left edge"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_request", return_value={"results": [sam_result()]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a"], 3)
        sam.assert_called_once_with({}, "/segment_text", image_bytes(), prompt="square")
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 3)
        self.assertEqual(result["candidates_added"], 1)
        expected = _rasterize(shape(), 40, 40)
        expected[2:19, 16:26] = 1
        expected[2:19, 2:6] = 0
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())

    def test_multi_prompt_generation_reuses_one_image_request(self):
        response = {"prompts": {
            "a": {"results": [sam_result()]},
            "b": {"results": []},
        }, "diagnostics": {"image_encoded_once": True, "total_seconds": 0.2}}
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "classes a and b",
            "prompts": [{"label": "a", "text": "square"},
                        {"label": "b", "text": "circle"}],
        }, ["a", "b"], 1)
        with patch.object(workflow, "sam_request", return_value=response) as request:
            agent.generate()
        request.assert_called_once_with({}, "/segment_text_multi", image_bytes(),
                                        prompts={"a": "square", "b": "circle"})
        self.assertEqual([item["label"] for item in agent.shapes], ["a"])
        self.assertEqual(agent.sam_generation["mode"], "multi_prompt")
        self.assertTrue(agent.sam_generation["diagnostics"]["image_encoded_once"])

    def test_multi_prompt_generation_falls_back_to_sequential_requests(self):
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "classes a and b",
            "prompts": [{"label": "a", "text": "square"},
                        {"label": "b", "text": "circle"}],
        }, ["a", "b"], 1)
        with patch.object(workflow, "sam_request", side_effect=[
                ValueError("SAM 3 请求失败：Not Found"),
                {"results": [sam_result()]}, {"results": []}]) as request:
            agent.generate()
        self.assertEqual([call.args[1] for call in request.call_args_list],
                         ["/segment_text_multi", "/segment_text", "/segment_text"])
        self.assertEqual(agent.sam_generation["mode"], "sequential_fallback")
        self.assertIn("Not Found", agent.sam_generation["error"])
        self.assertEqual([item["label"] for item in agent.shapes], ["a"])

    def test_keyword_from_gpt_is_preserved_when_short_and_relevant(self):
        answer = {"description": "标注参考帧里的可见车辆实例。", "prompts": [
            {"label": "车辆", "text": "forklift"},
            {"label": "person", "text": "worker"}]}
        result = workflow.restrict_description(answer, ["车辆"])
        self.assertEqual(result["description"], answer["description"])
        self.assertEqual(result["prompts"], [{"label": "车辆", "text": "forklift"}])

    def test_pallet_prompt_cannot_drift_to_stacked_cargo(self):
        for text in ("", " stacked boxes ", "wooden pallet", "pallet"):
            with self.subTest(text=text):
                answer = {"description": "只标注可见托盘结构，不把堆叠箱体当作托盘。",
                          "prompts": [{"label": "pallet", "text": text}]}
                result = workflow.restrict_description(answer, ["pallet"])
                self.assertEqual(result["description"], answer["description"])
                self.assertEqual(result["prompts"], [{"label": "pallet", "text": "pallet"}])
                self.assertEqual(answer["prompts"][0]["text"], text)

    def test_known_workbench_labels_have_stable_sam_prompt_fallbacks(self):
        labels = ["车辆", "pallet", "person", "floor", "slope", "ego", "other_robot", "cargo_floor"]
        result = workflow.restrict_description(
            {"description": "Review every configured class.", "prompts": []}, labels)
        self.assertEqual(result["prompts"], [
            {"label": "车辆", "text": "vehicle"},
            {"label": "pallet", "text": "pallet"},
            {"label": "person", "text": "person"},
            {"label": "floor", "text": "floor"},
            {"label": "slope", "text": "traversable surface"},
            {"label": "ego", "text": "ego robot"},
            {"label": "other_robot", "text": "robot"},
            {"label": "cargo_floor", "text": "robot cargo platform"},
        ])

    def test_slope_prompt_uses_evidence_backed_traversable_query(self):
        result = workflow.restrict_description(
            {"description": "Review the ramp.",
             "prompts": [{"label": "slope", "text": "yellow ramp"}]}, ["slope"])
        self.assertEqual(result["prompts"], [
            {"label": "slope", "text": "traversable surface"}])

    def test_slope_geometry_filter_keeps_broad_plane_and_routes_tall_strip(self):
        def polygon(candidate_id, label, points):
            part = {"points": points, "holes": []}
            return {"id": candidate_id, "label": label, "points": points,
                    "holes": [], "parts": [part], "score": 0.8}
        broad = polygon(1, "slope", [[1, 2], [31, 2], [31, 10], [1, 10]])
        tall = polygon(2, "slope", [[2, 1], [10, 1], [10, 31], [2, 31]])
        floor = polygon(3, "floor", [[1, 1], [5, 1], [5, 30], [1, 30]])
        retained, suppressed = workflow.filter_slope_candidate_geometry(
            [broad, tall, floor], 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_not_broad_planar")
        self.assertEqual(suppressed[0]["route"], "human_review_required")

    def test_slope_geometry_filter_routes_thin_horizontal_strip(self):
        points = [[1, 2], [35, 2], [35, 3], [1, 3]]
        part = {"points": points, "holes": []}
        candidate = {"id": 1, "label": "slope", "points": points,
                     "holes": [], "parts": [part], "score": 0.8}
        retained, suppressed = workflow.filter_slope_candidate_geometry(
            [candidate], 40)
        self.assertEqual(retained, [])
        self.assertEqual([item["id"] for item in suppressed], [1])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_below_minimum_height")
        self.assertEqual(suppressed[0]["minimum_height"], 4)

    def test_slope_generation_uses_overlap_crops_only_after_full_frame_rejection(self):
        def result(points, score=0.8):
            return {"parts": [{"points": points, "holes": []}], "score": score}
        tall = result([[2, 1], [10, 1], [10, 31], [2, 31]])
        crop_wide = result([[1, 1], [20, 1], [20, 8], [1, 8]], 0.7)
        crop_tall = result([[1, 1], [8, 1], [8, 20], [1, 20]])
        agent = workflow.AnnotationAgent({}, image_bytes(), {
            "description": "slope",
            "prompts": [{"label": "slope", "text": "traversable surface"}],
        }, ["slope"], 1)
        with patch.object(workflow, "sam_text", side_effect=[
                [tall], [crop_wide], [crop_tall], [crop_tall], [crop_tall]]) as segment:
            agent.generate()
        self.assertEqual(segment.call_count, 5)
        self.assertEqual(len(agent.shapes), 1)
        self.assertEqual(agent.shapes[0]["source"], "sam_text_overlap_crop_fallback")
        self.assertEqual(agent.shapes[0]["crop_index"], 0)
        self.assertEqual(len(agent.candidate_filters), 4)

    def test_slope_pallet_conflict_filter_preserves_pallet_and_disjoint_slope(self):
        pallet = shape(1, "pallet", 2, 18)
        overlapping = shape(2, "slope", 3, 17)
        disjoint = shape(3, "slope", 22, 38)
        retained, suppressed = workflow.filter_slope_pallet_conflicts(
            [pallet, overlapping, disjoint], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["pallet_id"], 1)
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_pallet_conflict")

    def test_slope_pallet_conflict_uses_occlusion_envelope_when_masks_are_disjoint(self):
        pallet_points = [[0, 0], [39, 0], [39, 5], [5, 5], [5, 39], [0, 39]]
        pallet_part = {"points": pallet_points, "holes": []}
        pallet = {"id": 1, "label": "pallet", "points": pallet_points,
                  "holes": [], "parts": [pallet_part], "score": 0.9}
        slope_points = [[20, 20], [30, 20], [30, 30], [20, 30]]
        slope_part = {"points": slope_points, "holes": []}
        slope = {"id": 2, "label": "slope", "points": slope_points,
                 "holes": [], "parts": [slope_part], "score": 0.8}
        retained, suppressed = workflow.filter_slope_pallet_conflicts(
            [pallet, slope], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["slope_covered_by_pallet"], 0.0)
        self.assertEqual(suppressed[0]["slope_envelope_covered_by_pallet"], 1.0)

    def test_slope_floor_conflict_routes_only_nearly_covered_slope(self):
        floor = shape(1, "floor", 1, 30)
        covered = shape(2, "slope", 3, 20)
        disjoint = shape(3, "slope", 31, 38)
        retained, suppressed = workflow.filter_slope_floor_conflicts(
            [floor, covered, disjoint], 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["floor_id"], 1)
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_floor_conflict")
        self.assertEqual(suppressed[0]["route"], "human_review_required")
        self.assertGreaterEqual(suppressed[0]["slope_covered_by_floor"], 0.9)

    def test_slope_reference_ego_guard_requires_same_resolution_reviewed_ego(self):
        ego = shape(1, "ego", 2, 24)
        covered = shape(2, "slope", 3, 20)
        disjoint = shape(3, "slope", 27, 38)
        reference = {"image": image_bytes(), "annotations": [ego]}
        retained, suppressed = workflow.filter_slope_reference_ego_conflicts(
            [ego, covered, disjoint], reference, 40, 40)
        self.assertEqual([item["id"] for item in retained], [1, 3])
        self.assertEqual([item["id"] for item in suppressed], [2])
        self.assertEqual(suppressed[0]["reason_code"],
                         "slope_candidate_reference_ego_conflict")
        self.assertEqual(suppressed[0]["route"], "human_review_required")
        self.assertGreaterEqual(suppressed[0]["slope_covered_by_reference_ego"], 0.8)
        smaller = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(smaller, "JPEG")
        unchanged, warnings = workflow.filter_slope_reference_ego_conflicts(
            [covered], {"image": smaller.getvalue(), "annotations": [ego]}, 40, 40)
        self.assertEqual(unchanged, [covered])
        self.assertEqual(warnings, [])

    def test_robot_and_person_prompts_use_exact_canonical_queries(self):
        labels = ["ego", "other_robot", "person"]
        expected = [
            {"label": "ego", "text": "ego robot"},
            {"label": "other_robot", "text": "robot"},
            {"label": "person", "text": "person"}]
        for prompts in ([], [
                {"label": "ego", "text": "camera viewpoint"},
                {"label": "other_robot", "text": "human operator"},
                {"label": "person", "text": "operator"}]):
            with self.subTest(prompts=prompts):
                result = workflow.restrict_description(
                    {"description": "区分相机载体、其他机器人与人员。", "prompts": prompts}, labels)
                self.assertEqual(result["prompts"], expected)

    def test_cargo_floor_prompt_is_overridden_even_when_gpt_returns_ambiguous_text(self):
        result = workflow.restrict_description(
            {"description": "cargo area", "prompts": [{"label": "cargo_floor", "text": "cargo floor"}]},
            ["cargo_floor"])
        self.assertEqual(result["prompts"], [
            {"label": "cargo_floor", "text": "robot cargo platform"}])

    def test_person_prompt_uses_evidence_backed_canonical_query(self):
        result = workflow.restrict_description(
            {"description": "people", "prompts": [
                {"label": "person", "text": "worker operator worker worker"}]},
            ["person"])
        self.assertEqual(result["prompts"], [{"label": "person", "text": "person"}])

    def test_candidate_filter_removes_same_label_and_floor_conflicts(self):
        floor = shape(label="floor", shape_id=1)
        floor_copy = shape(label="floor", shape_id=2)
        cargo_copy = shape(label="cargo_floor", shape_id=3)
        kept, filtered = workflow.suppress_near_duplicate_candidates(
            [floor, floor_copy, cargo_copy], 40, 40)
        self.assertEqual([(item["id"], item["label"]) for item in kept], [(1, "floor")])
        self.assertEqual({item["reason"] for item in filtered},
                         {"near_duplicate_same_label", "overlaps_warehouse_floor"})

    def test_unreferenced_draft_preserves_floor_deck_ambiguity_and_same_class_dedup(self):
        originals = [shape(1, "floor"), shape(2, "floor"), shape(3, "cargo_floor")]
        baseline = copy.deepcopy(originals)
        kept, filtered = workflow.suppress_near_duplicate_candidates(
            originals, 40, 40, preserve_class_conflicts=True)
        self.assertEqual({item["id"] for item in kept}, {1, 3})
        self.assertEqual([item["reason"] for item in filtered], ["near_duplicate_same_label"])
        self.assertEqual(originals, baseline)

    def test_class_overlap_flags_only_conflicting_high_iou_pairs(self):
        cases = [
            ([shape(1, "ego"), shape(2, "other_robot")], 1),
            ([shape(1, "车辆"), shape(2, "ego")], 1),
            ([shape(1, "floor"), shape(2, "cargo_floor")], 1),
            ([shape(1, "person"), shape(2, "ego")], 0),
            ([shape(1, "ego"), shape(2, "ego")], 0),
            ([shape(1, "ego"), shape(2, "other_robot", 23, 35)], 0),
        ]
        for shapes, expected in cases:
            with self.subTest(labels=[item["label"] for item in shapes], expected=expected):
                baseline = copy.deepcopy(shapes)
                conflicts = workflow.candidate_class_overlap_conflicts(shapes, 40, 40)
                self.assertEqual(len(conflicts), expected)
                self.assertEqual(shapes, baseline)
                if expected:
                    self.assertEqual(conflicts[0]["shape_ids"], [1, 2])
                    self.assertEqual(conflicts[0]["route"], "human_review_required")
                    self.assertEqual(conflicts[0]["iou"], 1.0)

    def test_overlap_conflict_ids_follow_merged_candidate_ids(self):
        generated = [shape(1, "ego"), shape(2, "other_robot")]
        combined, id_map = server.merge_inference_candidates(
            [shape(1, "pallet")], generated, ["ego", "other_robot"], return_id_map=True)
        result = {"agent": {}, "applied": [], "review_queue":
                  workflow.candidate_class_overlap_conflicts(generated, 40, 40)}
        server.bind_merged_candidate_references(result, id_map)
        by_id = {item["id"]: item["label"] for item in combined}
        conflict = result["review_queue"][0]
        self.assertEqual([by_id[item] for item in conflict["shape_ids"]],
                         ["ego", "other_robot"])
        self.assertEqual(conflict["original_shape_ids"], [1, 2])

    def test_unreferenced_pipeline_retains_conflict_despite_model_done(self):
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", return_value={"status": "done", "operations": []}):
            result = workflow.run({}, image_bytes(), {"description": "draft", "prompts": [
                {"label": "floor", "text": "floor"},
                {"label": "cargo_floor", "text": "robot cargo platform"}]},
                ["floor", "cargo_floor"])
        self.assertEqual({item["label"] for item in result["shapes"]}, {"floor", "cargo_floor"})
        self.assertEqual(len(result["review_queue"]), 1)
        self.assertEqual(result["review_queue"][0]["reason_code"], "candidate_cross_class_overlap")
        self.assertEqual(result["review_queue"][0]["shape_ids"], [1, 2])

    def test_unknown_non_ascii_label_still_requires_an_explicit_mapping(self):
        with self.assertRaisesRegex(ValueError, "缺少简短英文"):
            workflow.restrict_description(
                {"description": "unknown class", "prompts": []}, ["未知类别"])

    def test_gpt_reviews_initial_sam_candidates_against_detailed_rule(self):
        prompts = []
        def inspect(config, instruction, image, extra_images=()):
            prompts.append((instruction, len(extra_images)))
            return {"status": "done", "reason": "边界符合参考帧规范", "operations": []}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=inspect):
            result = workflow.run({}, image_bytes(), {"description": "只标注可见目标边界，不补全遮挡部分。",
                "prompts": [{"label": "a", "text": "target"}]}, ["a"], 2)
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(len(prompts), 1)
        self.assertIn("只标注可见目标边界", prompts[0][0])
        self.assertIn('"id": 1', prompts[0][0])
        self.assertEqual(prompts[0][1], 2)

    def test_unreferenced_batch_skips_reference_dependent_floor_gap_analysis(self):
        decision = {"status": "done", "reason": "No supported edit.",
                    "hole_assessment": [], "operations": []}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "candidate_holes") as holes, \
             patch.object(workflow, "chat", return_value=decision):
            result = workflow.run({}, image_bytes(),
                {"description": "generic floor", "prompts": [{"label": "floor", "text": "floor"}]},
                ["floor"], 2, reference=None)
        holes.assert_not_called()
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["review_queue"], [])

    def test_model_trace_records_empty_sam_response_and_bad_gpt_json(self):
        events = []
        with workflow.trace_calls(events.append), \
             patch.object(workflow.urllib.request, "urlopen", side_effect=[
                 io.BytesIO(b'{"results": [], "diagnostics": {"model_mask_count": 0}}'),
                 io.BytesIO(b'{"choices":[{"message":{"content":"invalid json"}}]}')]):
            self.assertEqual(workflow.sam_text(image_bytes(), "person", {}), [])
            with self.assertRaises(ValueError):
                workflow.chat({"llm_url": "http://example.test", "llm_model": "test",
                               "llm_key": "secret-token"}, "inspect", image_bytes())
        self.assertEqual([event["status"] for event in events],
                         ["running", "succeeded", "running", "failed"])
        self.assertEqual(events[1]["output"]["returned_count"], 0)
        self.assertEqual(events[1]["input"]["image"]["width"], 40)
        self.assertEqual(events[3]["output"]["raw_text"], "invalid json")
        self.assertNotIn("secret-token", json.dumps(events))
        self.assertNotIn("base64", json.dumps(events))

    def test_compact_review_uses_bounded_generation_budget(self):
        captured = {}
        def respond(request, timeout):
            captured.update(json.loads(request.data))
            return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        token = workflow._review_call.set({"attempt": 1, "feedback": None})
        try:
            with patch.object(workflow.urllib.request, "urlopen", side_effect=respond):
                workflow._chat({"llm_url": "http://example.test", "llm_model": "test",
                                "llm_prompt_profile": "compact"}, "inspect", image_bytes())
        finally:
            workflow._review_call.reset(token)
        self.assertEqual(captured["max_tokens"], 768)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact"}, 0), 768)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact"}, 4), 1536)
        self.assertEqual(workflow.compact_review_token_budget(
            {"llm_prompt_profile": "compact", "llm_max_tokens": 1024}, 4), 1024)

        from source.agent.protocol_schema import review_schema
        schema = review_schema(["floor"], [shape(label="floor")], [{
            "id": "1:hole:1:1:2:2", "area": 4, "bbox": [1, 1, 2, 2]}])
        properties = schema["properties"]
        self.assertEqual(properties["reason"]["maxLength"], 120)
        self.assertEqual(properties["hole_assessment"]["items"]["properties"]
                         ["reason"]["maxLength"], 100)
        operation_variants = properties["operations"]["items"]["anyOf"]
        self.assertTrue(all(item["properties"]["reason"]["maxLength"] == 120
                            for item in operation_variants))

    def test_review_gap_selection_excludes_open_image_border_artifacts(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        holes = [
            {"id": "edge-right", "kind": "open_gap", "area": 1000,
             "bbox": [25, 4, 29, 14], "mask": mask},
            {"id": "edge-bottom", "kind": "open_gap", "area": 900,
             "bbox": [4, 18, 18, 19], "mask": mask},
            {"id": "internal-gap", "kind": "open_gap", "area": 400,
             "bbox": [5, 5, 12, 12], "mask": mask},
            {"id": "enclosed-hole", "kind": "hole", "area": 300,
             "bbox": [0, 0, 3, 3], "mask": mask},
        ]
        selected = workflow.select_review_gaps(holes, maximum=4)
        self.assertEqual({item["id"] for item in selected},
                         {"internal-gap", "enclosed-hole"})

    def test_review_gap_selection_suppresses_resolution_scaled_tail_noise(self):
        mask = np.zeros((1536, 1920), dtype=np.uint8)
        holes = [
            {"id": "tiny", "kind": "hole", "area": 50,
             "bbox": [10, 10, 20, 20], "mask": mask},
            {"id": "meaningful", "kind": "hole", "area": 70,
             "bbox": [30, 30, 40, 40], "mask": mask},
        ]
        selected = workflow.select_review_gaps(holes)
        self.assertEqual([item["id"] for item in selected], ["meaningful"])
        self.assertEqual([item["id"] for item in workflow.select_review_gaps(
            holes, minimum_area=40)], ["tiny", "meaningful"])

    def test_truncated_compact_review_doubles_budget_once_for_protocol_repair(self):
        budgets, events = [], []
        def fake_chat(config, instruction, image, extra_images=(), raw=None,
                      response_schema=None):
            budgets.append(int(config.get("llm_max_tokens") or 768))
            if len(budgets) == 1:
                raw.update(text='{"status":"done","operations":[{"reason":"unfinished',
                           finish_reason="length")
                raise workflow.ModelOutputError(
                    "大模型未返回有效 JSON：Unterminated string", raw["text"])
            raw.update(text='{"status":"done","operations":[]}', finish_reason="stop")
            return {"status": "done", "operations": []}
        with workflow.trace_calls(events.append), patch.object(workflow, "_chat", side_effect=fake_chat):
            result = workflow.review({"llm_prompt_profile": "compact"}, image_bytes(),
                image_bytes(), {"description": "a", "prompts": []}, [], ["a"], [])
        self.assertEqual(result["status"], "done")
        self.assertEqual(budgets, [768, 1536])
        gpt_events = [event for event in events if event["service"] == "GPT"
                      and event["status"] in ("rejected", "succeeded")]
        self.assertEqual(gpt_events[0]["output"]["finish_reason"], "length")
        self.assertEqual(gpt_events[-1]["input"]["max_tokens"], 1536)

    def test_local_vlm_busy_waits_and_retries_without_repeating_sam(self):
        busy = workflow.urllib.error.HTTPError(
            "http://example.test", 503, "Service Unavailable", None,
            io.BytesIO(b'{"error":{"message":"Local VLM is busy"}}'))
        success = io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"{\\"status\\":\\"done\\"}"}}]}')
        raw, events = {}, []
        token = workflow._review_call.set({"record": {
            "id": "busy-call", "service": "GPT", "status": "running"}})
        try:
            with workflow.trace_calls(events.append), \
                 patch.object(workflow.urllib.request, "urlopen", side_effect=[busy, success]) as request, \
                 patch.object(workflow, "_wait_cancellable") as wait:
                result = workflow._chat({"llm_url": "http://example.test",
                    "llm_model": "test"}, "inspect", image_bytes(), raw=raw)
        finally:
            workflow._review_call.reset(token)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(request.call_count, 2)
        wait.assert_called_once_with(5)
        self.assertEqual(raw["busy_retry_count"], 1)
        self.assertEqual(events[0]["status"], "waiting")
        self.assertEqual(events[0]["retry_in_seconds"], 5)

    def test_local_vlm_request_stats_are_preserved_without_unbounded_fields(self):
        response = {
            "local_vlm": {
                "backend": "transformers_local_cuda",
                "image_count": 1,
                "effective_image_token_budget": 256,
                "actual_image_token_counts": [252],
                "image_grid_thw": [[1, 28, 36]],
                "prompt_tokens": 900,
                "completion_tokens": 120,
                "generation_seconds": 12.5,
                "seconds": 12.7,
                "inference_stats": {"prompt_tokens": 900,
                    "completion_tokens": 120, "generation_seconds": 12.5,
                    "peak_allocated_bytes": 123456789},
                "cache_release": {"large": "not copied"},
            },
            "usage": {"prompt_tokens": 900, "completion_tokens": 120,
                      "total_tokens": 1020},
            "choices": [{"finish_reason": "stop", "message": {
                "content": "{\"status\":\"done\"}"}}],
        }
        raw = {}
        with patch.object(workflow.urllib.request, "urlopen",
                          return_value=io.BytesIO(json.dumps(response).encode())):
            result = workflow._chat({"llm_url": "http://example.test",
                "llm_model": "test"}, "inspect", image_bytes(), raw=raw)
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(raw["local_vlm"]["actual_image_token_counts"], [252])
        self.assertEqual(raw["local_vlm"]["generation_seconds"], 12.5)
        self.assertEqual(raw["local_vlm"]["seconds"], 12.7)
        self.assertEqual(raw["local_vlm"]["inference_stats"], {
            "prompt_tokens": 900, "completion_tokens": 120,
            "generation_seconds": 12.5})
        self.assertEqual(raw["local_vlm"]["usage"]["total_tokens"], 1020)
        self.assertNotIn("cache_release", raw["local_vlm"])

    def test_trace_records_sam_transport_failure_and_restores_context(self):
        events = []
        with workflow.trace_calls(events.append), \
             patch.object(workflow.urllib.request, "urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(ValueError, "超时"):
                workflow.sam_text(image_bytes(), "person", {})
        self.assertEqual(events[-1]["status"], "failed")
        self.assertIn("超时", events[-1]["error"])
        self.assertIsNone(workflow._trace_callback.get())

    def run_agent(self, decisions, max_rounds=3):
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=decisions):
            return workflow.run({}, image_bytes(),
                {"description": "annotate only class a", "prompts": [{"label": "a", "text": "square"}]},
                ["a", "b"], max_rounds=max_rounds)

    def test_edit_requires_new_observation_before_done(self):
        result = self.run_agent([
            {"status": "done", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "b", "reason": "wrong class"}]},
            {"status": "done", "operations": []}])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 2)
        self.assertEqual(result["shapes"][0]["label"], "b")

    def test_cycle_and_round_limit_require_review(self):
        cycle = self.run_agent([
            {"status": "continue", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "b", "reason": "reference class"}]},
            {"status": "continue", "operations": [{"operation": "reclassify",
                "shape_ids": [1], "label": "a", "reason": "reference class"}]}])
        self.assertEqual(cycle["agent"]["stop_reason"], "cycle")
        self.assertTrue(cycle["review_queue"])
        limit = self.run_agent([{"status": "continue", "operations": [
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]}], 1)
        self.assertEqual(limit["agent"]["stop_reason"], "max_rounds")
        self.assertFalse(limit["agent"]["converged"])

    def test_invalid_sam_repair_requests_preserve_candidates(self):
        for kind in ("sam_points", "sam_text"):
            for initial in ([sam_result()], []):
                with self.subTest(kind=kind, empty=not initial):
                    decision = {"status": "done", "operations": [{"operation": kind,
                        "shape_ids": [1] if initial else [], "label": "a", "prompt": "square",
                        "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]}
                    with patch.object(workflow, "sam_request", return_value={"results": initial}) as sam, \
                         patch.object(workflow, "chat", return_value=decision) as chat:
                        result = workflow.run({}, image_bytes(),
                            {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a"], 2)
                    self.assertEqual(sam.call_args_list[0].args[1], "/segment_text")
                    self.assertEqual(len(result["shapes"]), len(initial))
                    if initial:
                        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])
                        self.assertEqual(result["shapes"][0]["id"], 1)
                    self.assertFalse(result["agent"]["converged"])
                    self.assertEqual(result["agent"]["stop_reason"], "invalid_model_output")
                    self.assertEqual(result["agent"]["audit"], [])
                    self.assertEqual(chat.call_count, 3)
                    self.assertEqual(sam.call_count, 1)
                    self.assertIn("协议校验重试 2 次后仍失败", result["review_queue"][0]["error"])
                    self.assertEqual(result["review_queue"][0]["reason_code"],
                                     "agent_invalid_model_output")
                    self.assertEqual(result["review_queue"][0]["route"],
                                     "human_review_required")

    def test_sam_points_refine_existing_mask_locally_and_require_next_gpt_assessment(self):
        edit = {"operation": "sam_points", "shape_ids": [1],
                "polygon": [[15, 0], [25, 0], [25, 20], [15, 20]],
                "points": [{"x": 20, "y": 5, "label": 1}, {"x": 28, "y": 5, "label": 0}],
                "reason": "repair right boundary"}
        decisions = [{"status": "continue", "operations": [edit]},
                     {"status": "done", "sam_assessment": {"improved": True,
                        "reason": "missing region filled; correct boundaries preserved"}, "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(left=0, right=30)["parts"]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions) as chat:
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(result["agent"]["rounds"], 2)
        self.assertEqual(sam.call_args.args[1], "/segment_points")
        self.assertEqual(sam.call_args.kwargs["points"], edit["points"])
        self.assertEqual(sam.call_args.kwargs["polygon"], edit["polygon"])
        import numpy as np
        prior = np.asarray(Image.open(io.BytesIO(base64.b64decode(sam.call_args.kwargs["mask"])))) > 127
        expected = _rasterize(shape(), 40, 40)
        self.assertEqual(prior.tolist(), expected.astype(bool).tolist())
        expected[2:19, 15:26] = 1
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())
        self.assertEqual(result["shapes"][0]["id"], 1)
        self.assertEqual(len(chat.call_args_list[1].kwargs["extra_images"]), 3)
        self.assertIn("图像 4 是修改前候选", chat.call_args_list[1].args[1])

    def test_duplicate_only_sam_points_and_zero_area_polygon_are_rejected_before_sam(self):
        invalid = {"status": "continue", "operations": [{"operation": "sam_points",
            "shape_ids": [1], "polygon": [[10, 10], [10, 10], [10, 10]],
            "points": [{"x": 10, "y": 10, "label": 1}] * 3,
            "reason": "repair boundary"}]}
        with self.assertRaisesRegex(ValueError, "有效 polygon|重复的同一个点"):
            workflow._validate_review_result(invalid, [], [shape()], ["a"], 40, 40)

    def test_conflicting_and_out_of_bounds_sam_points_are_rejected(self):
        base = {"status": "continue", "operations": [{"operation": "sam_points",
            "shape_ids": [1], "polygon": [[0, 0], [20, 0], [20, 20]],
            "points": [{"x": 5, "y": 5, "label": 1}, {"x": 5, "y": 5, "label": 0}],
            "reason": "repair boundary"}]}
        with self.assertRaisesRegex(ValueError, "冲突"):
            workflow._validate_review_result(base, [], [shape()], ["a"], 40, 40)
        base["operations"][0]["points"] = [{"x": 40, "y": 5, "label": 1}]
        with self.assertRaisesRegex(ValueError, "有效 points"):
            workflow._validate_review_result(base, [], [shape()], ["a"], 40, 40)

    def test_sam_improvement_claim_without_pending_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "禁止声明 improved"):
            workflow._validate_review_result({"status": "done", "operations": [],
                "sam_assessment": {"improved": True, "reason": "looks better"}},
                [], [shape()], ["a"], 40, 40)

    def test_sam_regression_rolls_back_before_opencv_fallback(self):
        edit = {"operation": "sam_points", "shape_ids": [1],
                "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
                "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}
        decisions = [{"status": "continue", "operations": [edit]},
            {"status": "continue", "sam_assessment": {"improved": False, "reason": "lost correct area"},
             "operations": [{"operation": "add_region", "shape_ids": [1],
                             "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair boundary"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(right=10)["parts"]}) as sam, \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertEqual(sam.call_count, 1)
        expected = _rasterize(shape(), 40, 40)
        expected[2:19, 16:26] = 1
        self.assertEqual(_rasterize(result["shapes"][0], 40, 40).tolist(), expected.tolist())
        self.assertEqual(result["agent"]["audit"][1]["sam_rollback"]["shape_id"], 1)

    def test_sam_no_change_allows_next_round_to_use_opencv(self):
        decisions = [{"status": "continue", "operations": [{"operation": "sam_points", "shape_ids": [1],
            "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]},
            {"status": "continue", "operations": [{"operation": "add_region", "shape_ids": [1],
             "polygon": [[16, 2], [25, 2], [25, 18], [16, 18]], "reason": "repair boundary"}]},
            {"status": "done", "operations": []}]
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value=sam_result()), \
             patch.object(workflow, "chat", side_effect=decisions):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"])
        self.assertTrue(result["agent"]["converged"])
        self.assertIn("OpenCV", result["agent"]["audit"][0]["tool_results"][0]["fallback"])

    def test_unassessed_sam_result_is_rolled_back_at_round_limit(self):
        decision = {"status": "continue", "operations": [{"operation": "sam_points", "shape_ids": [1],
            "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "sam_request", return_value={"parts": shape(right=10)["parts"]}), \
             patch.object(workflow, "chat", return_value=decision):
            result = workflow.run({}, image_bytes(), {"description": "a only",
                "prompts": [{"label": "a", "text": "square"}]}, ["a"], 1)
        self.assertFalse(result["agent"]["converged"])
        self.assertEqual(result["shapes"][0]["parts"], shape()["parts"])

    def test_stop_during_model_call_returns_without_waiting_for_response(self):
        stop, entered, release, finished = (threading.Event() for _ in range(4))
        errors = []
        def model():
            entered.set()
            release.wait(5)
            return "late response"
        def caller():
            try:
                with workflow.cancellation(stop):
                    workflow._cancellable_call(model)
            except workflow.RunCancelled:
                errors.append("cancelled")
            finally:
                finished.set()
        worker = threading.Thread(target=caller)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            stop.set()
            self.assertTrue(finished.wait(2))
            self.assertEqual(errors, ["cancelled"])
        finally:
            release.set()
            worker.join(2)

    def test_repeated_failed_operation_is_skipped_and_audited(self):
        failed = {"operation": "sam_points", "shape_ids": [1], "label": "a",
                  "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
                  "points": [{"x": 5, "y": 5, "label": 1}], "reason": "repair boundary"}
        decision = {"status": "continue", "operations": [failed]}
        first = {"status": "continue", "operations": [failed,
            {"operation": "reclassify", "shape_ids": [1], "label": "b", "reason": "reference class"}]}
        with patch.object(workflow, "sam_text", return_value=[sam_result()]), \
             patch.object(workflow, "chat", side_effect=[first, decision]), \
             patch.object(workflow, "sam_request") as sam:
            result = workflow.run({}, image_bytes(),
                {"description": "class a", "prompts": [{"label": "a", "text": "square"}]}, ["a", "b"], 2)
        self.assertEqual([item["status"] for item in result["agent"]["audit"][1]["tool_results"]],
                         ["skipped"])
        self.assertTrue(any("停止重复重试" in item["reason"]
                            for item in result["review_queue"]))


class FakeVideo:
    path = Path("/tmp/test.mcap")
    topics = {"camera0": {"count": 3}, "camera1": {"count": 3}}
    catalog = topics
    duration = 1

    def frame(self, topic, index):
        return image_bytes()

    def timeline(self, topic, directory):
        return {"status": "ready", "times": [0.0, 0.5, 1.0], "fps": 2,
                "duration": 1.0, "count": 3, "timestamp_source": "capture",
                "progress": 1, "playback_ready": False, "url": None}


class ServerTests(unittest.TestCase):
    def test_model_readiness_probes_sam_and_vlm_concurrently(self):
        old_state = server.state
        barrier = threading.Barrier(2, timeout=1)
        calls = []

        def probe(url, timeout=3):
            calls.append(url)
            barrier.wait()
            if url.endswith("/health"):
                return {"status": "ok", "model_loaded": True}
            return {"data": [{"id": "vlm-test"}]}

        server.state = {"config": {
            "sam3_url": "http://sam",
            "llm_url": "http://vlm/v1",
            "llm_model": "vlm-test",
        }}
        try:
            with patch.object(server, "_fetch_readiness_json", side_effect=probe):
                result = server.app.test_client().get("/api/model_readiness").json
            self.assertTrue(result["ready"])
            self.assertEqual(set(calls), {"http://sam/health", "http://vlm/v1/models"})
            self.assertGreaterEqual(result["elapsed_ms"], 0)
        finally:
            server.state = old_state

    def test_run_accepts_exact_retry_frames_without_reprocessing_successful_range(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            calls = []
            def successful(*args, **kwargs):
                calls.append(len(calls) + 1)
                return {"shapes": [shape(calls[-1])], "candidates_added": 1,
                        "applied": [], "review_queue": [],
                        "agent": {"stop_reason": "done", "rounds": 1}}
            try:
                client = server.app.test_client()
                with patch.object(server, "run", side_effect=successful):
                    response = client.post("/api/run", json={
                        "frames": [2, 1, 2, 0], "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual([item["frame"] for item in response.json["frames"]], [2, 1])
                self.assertEqual(response.json["requested_frames"], [2, 1, 0])
                self.assertEqual(len(calls), 2)
                self.assertEqual(set(server.state["candidates"]), {"1", "2"})
                invalid_type = client.post("/api/run", json={
                    "frames": [True], "description": description})
                self.assertEqual(invalid_type.status_code, 400, invalid_type.json)
                invalid_range = client.post("/api/run", json={
                    "frames": [3], "description": description})
                self.assertEqual(invalid_range.status_code, 400, invalid_range.json)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_summary_is_bounded_scoped_and_recoverable_after_idle_restart(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            server.run_status = {"status": "idle"}
            output = {
                "task_id": "task-123", "status": "failed", "file": str(FakeVideo.path),
                "topic": "camera0", "reference": 0, "total": 2, "targets": [1, 2],
                "frames": [
                    {"frame": 1, "remaining": 3, "candidates_added": 2,
                     "failed": False, "agent": {"large": "x" * 10000}},
                    {"frame": 2, "remaining": 1, "candidates_added": 0,
                     "failed": True, "existing_candidates_preserved": True,
                     "review_queue": [{"reason": "large" * 1000}]},
                ],
                "failed_frames": [2], "message": "1/2 帧失败",
                "description": {"large": "y" * 10000}, "trace": [{"large": "z" * 10000}],
                "steps": [{"image": "huge" * 10000}], "current": None,
            }
            try:
                server.save_run(output)
                summary_path = server.WORK / "last_run_summary.json"
                self.assertTrue(summary_path.is_file())
                self.assertLess(summary_path.stat().st_size, 2048)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["task_id"], "task-123")
                self.assertEqual(summary["failed_frames"], [2])
                self.assertEqual(summary["pending_frames"], [])
                self.assertNotIn("trace", summary)
                self.assertNotIn("agent", summary["frames"][0])
                summary_path.unlink()
                migrated = server.load_run_summary()
                self.assertEqual(migrated["task_id"], "task-123")
                self.assertTrue(summary_path.is_file())
                client = server.app.test_client()
                recovered = client.get("/api/run/result?task_id=task-123")
                self.assertEqual(recovered.status_code, 200, recovered.json)
                status = client.get("/api/run/status").json
                self.assertTrue(status["result_available"])
                self.assertEqual(status["result_task_id"], "task-123")
                self.assertEqual(status["failed_frames"], [2])
                stale = client.get("/api/run/result?task_id=other-task")
                self.assertEqual(stale.status_code, 409, stale.json)
                server.state["topic"] = "camera1"
                wrong_scope = client.get("/api/run/result?task_id=task-123")
                self.assertEqual(wrong_scope.status_code, 409, wrong_scope.json)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_interrupted_run_summary_exposes_only_unfinished_frames_for_resume(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            server.run_status = {"status": "idle"}
            output = {
                "task_id": "interrupted-123", "status": "running",
                "file": str(FakeVideo.path), "topic": "camera0", "reference": 0,
                "total": 2, "targets": [1, 2],
                "frames": [{"frame": 1, "remaining": 2, "failed": False}],
                "current": {"frame": 2, "phase": "observing", "round": 1},
            }
            try:
                server.save_run(output)
                summary = server.load_run_summary()
                self.assertEqual(summary["pending_frames"], [2])
                self.assertEqual(summary["completed"], 1)
                client = server.app.test_client()
                status = client.get("/api/run/status").json
                self.assertTrue(status["result_available"])
                self.assertTrue(status["result_interrupted"])
                self.assertEqual(status["pending_frames"], [2])
                recovered = client.get(
                    "/api/run/result?task_id=interrupted-123").json
                self.assertEqual(recovered["pending_frames"], [2])
                self.assertEqual([item["frame"] for item in recovered["frames"]], [1])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_and_debug_records_use_compact_atomic_json_without_semantic_loss(self):
        old_work = server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            output = {"status": "running", "file": "中文.mcap", "topic": "相机",
                      "targets": [1, 2], "frames": [{"frame": 1, "remaining": 3}],
                      "trace": [{"message": "斜坡", "values": list(range(100))}]}
            trace = [{"service": "GPT", "message": "人工复核", "round": 2}]
            try:
                server.save_run(output)
                server.save_debug(trace)
                run_bytes = (server.WORK / "last_run.json").read_bytes()
                debug_bytes = (server.WORK / "model_debug.json").read_bytes()
                self.assertNotIn(b"\n  ", run_bytes)
                self.assertNotIn(b"\n  ", debug_bytes)
                self.assertEqual(json.loads(run_bytes.decode("utf-8")), output)
                self.assertEqual(json.loads(debug_bytes.decode("utf-8")), {"trace": trace})
                self.assertFalse((server.WORK / "last_run.tmp").exists())
                self.assertFalse((server.WORK / "model_debug.tmp").exists())
            finally:
                server.WORK = old_work

    def test_timeline_endpoint_returns_exact_times_without_starting_playback(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0"}
            try:
                response = server.app.test_client().get("/api/timeline?topic=camera1")
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["times"], [0.0, 0.5, 1.0])
                self.assertEqual(response.json["topic"], "camera1")
                self.assertEqual(response.json["offset"], 0)
                self.assertFalse(response.json["playback_ready"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_fast_and_streaming_json_writes_are_equivalent(self):
        value = {"中文": [shape(), {"text": "斜坡", "value": 1.25}], "flag": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fast, streaming = root / "fast.json", root / "streaming.json"
            server._write_json(fast, value, fast=True)
            server._write_json(streaming, value, fast=False)
            self.assertEqual(fast.read_bytes(), streaming.read_bytes())
            self.assertEqual(json.loads(fast.read_text(encoding="utf-8")), value)
            self.assertTrue(server._fast_json_write_allowed(fast))
            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(server.FAST_JSON_WRITE_LIMIT + 1)
            self.assertFalse(server._fast_json_write_allowed(oversized))
            self.assertFalse(server._fast_json_write_allowed(root / "missing.json"))

    def test_mcap_catalog_index_reuses_metadata_and_invalidates_on_file_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "sample.mcap"
            recording.write_bytes(b"first-version")
            metadata = ({"camera": {
                "type": "image", "schema": "test.Image", "encoding": "protobuf",
                "count": 3, "offset": 0.0, "log_offset": 0.0,
            }}, 123, 2.5)
            with patch.object(mcap_video, "catalog", return_value=metadata) as build:
                first = mcap_video.Video(recording, root / "index")
                second = mcap_video.Video(recording, root / "index")
                self.assertEqual(build.call_count, 1)
                self.assertEqual(first.catalog, second.catalog)
                self.assertEqual(second.origin_ns, 123)
                self.assertEqual(second.duration, 2.5)
                recording.write_bytes(b"second-version-is-different")
                third = mcap_video.Video(recording, root / "index")
                self.assertEqual(build.call_count, 2)
                self.assertEqual(third.catalog, metadata[0])

    def test_run_claim_rejects_concurrent_run_and_reference_draft(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            def blocked_run(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                responses["run"] = server.app.test_client().post("/api/run", json={
                    "frame": 1, "count": 1, "description": description})
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    responses["draft"] = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 2, "labels": ["a"]})
                    self.assertEqual(responses["draft"].status_code, 503)
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["run"].status_code, 200, responses["run"].json)
                self.assertEqual(server.run_status["status"], "completed")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_claim_rejects_concurrent_reference_draft_and_run(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            def blocked_describe(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"description": "a", "prompts": []}
            result = {"shapes": [shape()], "candidates_added": 1, "applied": [],
                      "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_draft():
                responses["draft"] = server.app.test_client().post("/api/reference_draft", json={
                    "frame": 1, "labels": ["a"]})
            worker = threading.Thread(target=start_draft)
            try:
                with patch.object(server, "describe", side_effect=blocked_describe), \
                     patch.object(server, "run", return_value=result), \
                     patch.object(server, "audit_high_risk_candidates", return_value=([], [])):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    responses["run"] = server.app.test_client().post("/api/run", json={"frame": 1})
                    self.assertEqual(responses["run"].status_code, 503)
                    for endpoint in ("/api/describe", "/api/segment_points",
                                     "/api/reference_candidates/audit"):
                        with self.subTest(endpoint=endpoint):
                            rejected = server.app.test_client().post(endpoint, json={"frame": 0})
                            self.assertEqual(rejected.status_code, 503, rejected.json)
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["draft"].status_code, 200, responses["draft"].json)
                self.assertEqual(server.run_status["status"], "completed")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_claim_keeps_ownership_through_terminal_response_and_error_cleanup(self):
        old_status, old_task, old_stop = server.run_status, server.run_task_id, server.run_stop_event
        try:
            for terminal in ("completed", "failed", "stopped"):
                with self.subTest(terminal=terminal):
                    server.run_status = {"status": "idle"}
                    entered, release = threading.Event(), threading.Event()
                    errors, task_ids = [], []
                    @server.run_claimed
                    def old_task():
                        task_ids.append(server.run_task_id)
                        server.update_run_status(status=terminal, phase=terminal)
                        entered.set()
                        if not release.wait(3):
                            raise AssertionError("未释放旧任务收尾")
                        if terminal == "failed":
                            raise RuntimeError("旧任务异常")
                        server.update_run_status(error="旧任务收尾")
                    @server.run_claimed
                    def new_task():
                        task_ids.append(server.run_task_id)
                        self.assertFalse(server.run_stop_event.is_set())
                        server.update_run_status(status="completed", phase="completed", error="新任务")
                    def start_old():
                        try:
                            old_task()
                        except Exception as error:
                            errors.append(str(error))
                    worker = threading.Thread(target=start_old)
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(2))
                        with self.assertRaisesRegex(RuntimeError, "已有推理任务"):
                            new_task()
                        self.assertEqual(len(task_ids), 1)
                    finally:
                        release.set()
                        worker.join(4)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(errors, ["旧任务异常"] if terminal == "failed" else [])
                    new_task()
                    self.assertNotEqual(task_ids[0], task_ids[1])
                    self.assertEqual(server.run_status["error"], "新任务")
                    self.assertEqual(server.run_status["status"], "completed")
        finally:
            server.run_status, server.run_task_id, server.run_stop_event = old_status, old_task, old_stop

    def test_run_claim_reserves_before_preparation_and_preserves_stop(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_task, old_stop = server.run_task_id, server.run_stop_event
        entered, release = threading.Event(), threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            original = server.restrict_description
            def prepare(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return original(*args, **kwargs)
            def start():
                responses.append(server.app.test_client().post("/api/run", json={
                    "frame": 1, "count": 1, "description": description}))
            worker = threading.Thread(target=start)
            try:
                with patch.object(server, "restrict_description", side_effect=prepare), \
                     patch.object(server, "run") as inference:
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    task_event = server.run_stop_event
                    self.assertTrue(server.app.test_client().post("/api/run/stop").json["stopped"])
                    conflict = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 2, "labels": ["a"]})
                    self.assertEqual(conflict.status_code, 503)
                    self.assertIs(server.run_stop_event, task_event)
                    self.assertTrue(task_event.is_set())
                    release.set()
                    worker.join(4)
                    inference.assert_not_called()
                self.assertFalse(worker.is_alive())
                self.assertTrue(responses[0].json["stopped"])
                self.assertEqual(server.run_status["status"], "stopped")
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work
                server.run_task_id, server.run_stop_event = old_task, old_stop

    def test_run_model_wait_does_not_block_state_and_cancel_prevents_commit(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        entered, release = threading.Event(), threading.Event()
        response_box = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            def blocked_run(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                with server.app.test_client() as client:
                    response_box.append(client.post("/api/run", json={"frame": 1, "count": 1,
                                                                        "description": description}))
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(server.app.test_client().get("/api/state").status_code, 200)
                    self.assertTrue(server.app.test_client().post("/api/run/stop").json["stopped"])
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertTrue(response_box[0].json["stopped"])
                self.assertNotIn("1", server.state["candidates"])
                self.assertNotIn("1", server.state["agent_runs"])
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_stale_target_and_preserves_completed_frame(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        entered, release = threading.Event(), threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            calls = []
            def blocked_run(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    entered.set()
                    self.assertTrue(release.wait(3))
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
            def start_run():
                with server.app.test_client() as client:
                    responses.append(client.post("/api/run", json={"frame": 1, "count": 3,
                                                                      "description": description}))
            worker = threading.Thread(target=start_run)
            try:
                with patch.object(server, "run", side_effect=blocked_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    with server.lock:
                        server.state["candidates"]["2"] = [shape(99)]
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses[0].status_code, 409, responses[0].json)
                self.assertIn("1", server.state["candidates"])
                self.assertEqual(server.state["candidates"]["2"], [shape(99)])
            finally:
                release.set()
                if worker.ident is not None:
                    worker.join(4)
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_rejects_concurrent_scope_config_truth_reference_labels_and_run_changes(self):
        mutations = {
            "scope": lambda: server.state.update(topic="camera1"),
            "video": lambda: server.state.update(video=FakeVideo()),
            "config": lambda: server.state["config"].update(llm_key="new-key"),
            "labels": lambda: server.state["labels"].append("b"),
            "auto_labels": lambda: server.state.update(auto_labels=["a"]),
            "truth": lambda: server.state["truth"].update({"1": [shape(77)]}),
            "reference_truth": lambda: server.state["truth"]["0"].append(shape(2)),
            "references": lambda: server.state["reference_frames"].update(camera0=[0, 2]),
            "candidate": lambda: server.state["candidates"].update({"1": [shape(88)]}),
            "run": lambda: server.state["agent_runs"].update({"1": {"manual": True}}),
        }
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        for name, mutate in mutations.items():
            with self.subTest(change=name), tempfile.TemporaryDirectory() as directory:
                entered, release = threading.Event(), threading.Event()
                responses = []
                server.WORK = Path(directory)
                server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                    "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                    "config": {"llm_key": "captured-key"}, "candidates": {}, "agent_runs": {}}
                server.run_status = {"status": "idle"}
                description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                               "reference_signature": server.reference_signature(0)}
                def blocked_run(config, *args, **kwargs):
                    entered.set()
                    self.assertTrue(release.wait(4))
                    callback = workflow._trace_callback.get()
                    callback({"id": "snapshot", "message": "captured-key"})
                    self.assertEqual(config["llm_key"], "captured-key")
                    return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                        "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1}}
                def start_run():
                    with server.app.test_client() as client:
                        responses.append(client.post("/api/run", json={"frame": 1, "count": 1,
                                                                          "description": description}))
                worker = threading.Thread(target=start_run)
                try:
                    with patch.object(server, "run", side_effect=blocked_run):
                        worker.start()
                        self.assertTrue(entered.wait(2))
                        with server.lock:
                            mutate()
                            expected_candidates = copy.deepcopy(server.state["candidates"])
                            expected_runs = copy.deepcopy(server.state["agent_runs"])
                            expected_truth = copy.deepcopy(server.state["truth"])
                        release.set()
                        worker.join(5)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(responses[0].status_code, 409, responses[0].json)
                    self.assertEqual(server.state["candidates"], expected_candidates)
                    self.assertEqual(server.state["agent_runs"], expected_runs)
                    self.assertEqual(server.state["truth"], expected_truth)
                    trace = json.loads((server.WORK / "model_debug.json").read_text())["trace"][0]
                    self.assertEqual(trace["topic"], "camera0")
                    self.assertEqual(trace["message"], "[REDACTED]")
                finally:
                    release.set()
                    if worker.ident is not None:
                        worker.join(5)
                    server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_state_read_waits_for_atomic_mutation_even_without_pending_reviews(self):
        old_state = server.state
        server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                        "reference_frames": {}, "config": {}, "candidates": {},
                        "agent_runs": {}, "human_review_transactions": {"done": {"status": "committed"}}}
        acquired, release, finished = threading.Event(), threading.Event(), threading.Event()
        responses = []
        def hold_lock():
            with server.lock:
                acquired.set()
                release.wait(5)
        def read_state():
            try:
                with server.app.test_request_context("/api/state"):
                    responses.append(server.get_state().status_code)
            finally:
                finished.set()
        holder = threading.Thread(target=hold_lock)
        reader = threading.Thread(target=read_state)
        holder.start()
        try:
            self.assertTrue(acquired.wait(2))
            reader.start()
            self.assertFalse(finished.wait(0.1), "state read escaped the mutation lock")
            release.set()
            self.assertTrue(finished.wait(2), "state read did not resume after mutation")
            self.assertEqual(responses, [200])
        finally:
            release.set()
            holder.join(5)
            if reader.ident is not None:
                reader.join(5)
            server.state = old_state

    def test_state_summary_is_small_and_preserves_guide_decisions(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["person", "slope"],
                "truth": {"2": [shape()], "10": [shape()]},
                "reference_frames": {"camera0": [2]}, "config": {"llm_key": "secret"},
                "candidates": {"3": [shape(), shape()], "8": []}, "agent_runs": {},
                "human_review_transactions": {},
            }
            try:
                with patch.object(server, "current_truth_review_status", return_value={
                        "2": {"reusable_as_reference": True},
                        "10": {"reusable_as_reference": False}}):
                    response = server.app.test_client().get("/api/state/summary")
                self.assertEqual(response.status_code, 200)
                payload = response.json
                self.assertEqual(payload["schema_version"], "labelagent.state-summary.v1")
                self.assertEqual(payload["truth_frames"], ["2", "10"])
                self.assertEqual(payload["verified_truth_frames"], ["2"])
                self.assertEqual(payload["candidate_counts"], {"3": 2, "8": 0})
                self.assertEqual(payload["reference_frames"], {"camera0": [2]})
                self.assertNotIn("truth", payload)
                self.assertNotIn("candidates", payload)
                self.assertNotIn("agent_runs", payload)
                self.assertNotIn("config", payload)
                self.assertLess(len(response.data), 1024)
            finally:
                server.state, server.WORK = old_state, old_work

    def test_workbench_bootstrap_preserves_metadata_without_all_frame_geometry(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            dense = [shape(index + 1) for index in range(240)]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a", "slope"],
                "truth": {"0": [shape()], "2": [shape(2)]},
                "reference_frames": {}, "config": {"llm_model": "test", "llm_key": "secret"},
                "candidates": {"1": dense, "2": [shape(3)]},
                "agent_runs": {"1": {"rounds": 2}},
                "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                full = client.get("/api/state")
                bootstrap = client.get("/api/bootstrap")
                self.assertEqual(bootstrap.status_code, 200, bootstrap.json)
                payload = bootstrap.json
                self.assertEqual(payload["schema_version"], "labelagent.workbench-bootstrap.v1")
                self.assertTrue(payload["compact_geometry"])
                self.assertTrue(payload["loaded"])
                self.assertEqual(payload["path"], str(FakeVideo.path))
                self.assertEqual(payload["topics"], FakeVideo.topics)
                self.assertEqual(payload["truth_frames"], ["0", "2"])
                self.assertEqual(payload["candidate_counts"], {"1": 240, "2": 1})
                self.assertEqual(payload["truth"], {})
                self.assertEqual(payload["candidates"], {})
                self.assertEqual(payload["agent_runs"], {})
                self.assertNotIn("llm_key", payload["config"])
                self.assertLess(len(bootstrap.data), len(full.data) * 0.05)
            finally:
                server.state, server.WORK = old_state, old_work

    def test_reference_draft_inference_allows_state_reads_and_guards_commit(self):
        old_state, old_status, old_work, old_stop = (server.state, server.run_status,
                                                     server.WORK, server.run_stop_event)
        cases = [(stage, "success") for stage in ("discover", "describe", "run", "audit")]
        cases += [("run", change) for change in ("video", "topic", "config", "labels",
                  "candidates", "agent_runs", "truth", "reference_frames", "stop")]
        try:
            for stage, change in cases:
                with self.subTest(stage=stage, change=change), tempfile.TemporaryDirectory() as directory:
                    server.WORK = Path(directory)
                    server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                        "truth": {}, "reference_frames": {}, "config": {"llm_model": "test"},
                        "candidates": {}, "agent_runs": {}}
                    server.run_status = {"status": "idle"}
                    server.run_stop_event = threading.Event()
                    entered, release, read_finished = threading.Event(), threading.Event(), threading.Event()
                    responses, reads = [], []
                    discovery = {"visible_labels": ["person"], "uncertain_labels": [],
                                 "unknown_candidates": []}
                    description = {"description": "person", "prompts": []}
                    result = {"shapes": [shape(label="person")], "candidates_added": 1,
                              "applied": [], "review_queue": [],
                              "agent": {"stop_reason": "done", "rounds": 1, "audit": []}}
                    def block(name, value):
                        def invoke(*args, **kwargs):
                            if name == stage:
                                entered.set()
                                if not release.wait(5):
                                    raise RuntimeError("测试推理等待超时")
                            return value
                        return invoke
                    def start():
                        body = ({"frame": 1, "labels": ["person"]}
                                if stage == "describe"
                                else {"frame": 1, "auto_detect_visible": True})
                        responses.append(server.app.test_client().post(
                            "/api/reference_draft", json=body))
                    def read_state():
                        try:
                            reads.append(server.app.test_client().get("/api/state"))
                        finally:
                            read_finished.set()
                    worker, reader = threading.Thread(target=start), threading.Thread(target=read_state)
                    try:
                        with patch.object(server, "discover_reference_labels", side_effect=block("discover", (discovery, 0))), \
                             patch.object(server, "describe", side_effect=block("describe", description)), \
                             patch.object(server, "run", side_effect=block("run", result)), \
                             patch.object(server, "audit_high_risk_candidates", side_effect=block("audit", ([], []))):
                            worker.start()
                            self.assertTrue(entered.wait(3), "未进入模拟推理")
                            reader.start()
                            self.assertTrue(read_finished.wait(3), "推理期间状态读取被阻塞")
                            self.assertEqual(reads[0].status_code, 200)
                            with server.lock:
                                if change == "video":
                                    server.state["video"] = FakeVideo()
                                elif change == "topic":
                                    server.state["topic"] = "camera1"
                                elif change == "config":
                                    server.state["config"]["llm_model"] = "changed"
                                elif change == "labels":
                                    server.state["labels"].append("other")
                                elif change in ("candidates", "agent_runs", "truth"):
                                    server.state[change]["1"] = {"changed": True} if change == "agent_runs" else [{"changed": True}]
                                elif change == "reference_frames":
                                    server.state[change]["camera0"] = [1]
                            if change == "stop":
                                self.assertTrue(server.app.test_client().post("/api/run/stop").json["stopped"])
                            before_candidates = copy.deepcopy(server.state["candidates"])
                            before_runs = copy.deepcopy(server.state["agent_runs"])
                            release.set()
                            worker.join(5)
                            self.assertFalse(worker.is_alive())
                        self.assertEqual(len(responses), 1)
                        if change == "success":
                            self.assertEqual(responses[0].status_code, 200, responses[0].json)
                            self.assertIn("1", server.state["candidates"])
                        else:
                            self.assertEqual(server.state["candidates"], before_candidates)
                            self.assertEqual(server.state["agent_runs"], before_runs)
                            if change == "stop":
                                self.assertTrue(responses[0].json["stopped"])
                                self.assertEqual(server.run_status["status"], "stopped")
                            else:
                                self.assertEqual(responses[0].status_code, 409, responses[0].json)
                                self.assertEqual(server.run_status["status"], "failed")
                    finally:
                        release.set()
                        if worker.ident is not None:
                            worker.join(5)
                        if reader.ident is not None:
                            reader.join(5)
        finally:
            server.state, server.run_status, server.WORK, server.run_stop_event = (
                old_state, old_status, old_work, old_stop)

    def test_state_read_recovers_pending_review_under_lock(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                            "reference_frames": {}, "config": {}, "candidates": {},
                            "agent_runs": {}, "human_review_transactions": {}}
            try:
                transaction = {"status": "state_saved_pending_audit"}
                server.state["human_review_transactions"]["pending"] = transaction
                audit_folder = server.WORK / "human_reviews"
                audit_folder.mkdir()
                (audit_folder / "record-pending.json").write_text("{}", encoding="utf-8")
                original_save = server.save
                def check_save():
                    self.assertTrue(server.lock._is_owned())
                    original_save()
                with patch.object(server, "save", side_effect=check_save) as saved:
                    response = server.app.test_client().get("/api/state")
                self.assertEqual(response.status_code, 200)
                saved.assert_called_once()
                self.assertEqual(transaction["status"], "committed")
                self.assertEqual(transaction["audit_file"], "record-pending.json")
                persisted = json.loads((server.WORK / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(persisted["human_review_transactions"]["pending"], transaction)
                (audit_folder / "record-pending.json").unlink()
                (server.WORK / "state.json").unlink()
            finally:
                server.state, server.WORK = old_state, old_work

    def test_visibility_queue_preserves_conflict_even_with_candidate(self):
        flag = {"label": "other_robot", "reason_code": "vlm_cross_label_entity_conflict",
                "route": "human_review_required", "conflicting_labels": ["车辆", "other_robot"],
                "shared_anchor": [700, 450]}
        discovery = {"uncertain_labels": ["other_robot"], "consistency_flags": [flag]}
        queue = server.reference_visibility_review_queue(discovery, [shape(label="other_robot")])
        conflicts = [item for item in queue if item["reason_code"] == flag["reason_code"]]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(queue), 1)
        self.assertTrue(conflicts[0]["candidate_available"])
        self.assertEqual(conflicts[0]["evidence"], flag)
        conflicts[0]["evidence"]["shared_anchor"][0] = 0
        self.assertEqual(flag["shared_anchor"], [700, 450])

    def test_point_segment_has_bounded_timeout_and_elapsed_receipt(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
        payload = {"points": [[1, 1], [2, 1], [2, 2]], "holes": [],
                   "parts": [{"points": [[1, 1], [2, 1], [2, 2]], "holes": []}]}
        try:
            with patch.object(server.urllib.request, "urlopen",
                              return_value=io.BytesIO(json.dumps(payload).encode())) as opened:
                response = server.app.test_client().post("/api/segment_points", json={
                    "frame": 1, "points": [{"x": 1, "y": 1, "label": 1}]})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertGreaterEqual(response.json["inference_elapsed_seconds"], 0)
            self.assertEqual(opened.call_args.kwargs["timeout"], 180)
        finally:
            server.state = old_state

    def test_point_segment_rejects_duplicate_concurrent_request(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
            "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
        self.assertTrue(server.point_segment_lock.acquire(blocking=False))
        try:
            response = server.app.test_client().post("/api/segment_points", json={
                "frame": 1, "points": [{"x": 1, "y": 1, "label": 1}]})
            self.assertEqual(response.status_code, 503, response.json)
            self.assertIn("请勿重复点击", response.json["error"])
        finally:
            server.point_segment_lock.release()
            server.state = old_state

    def test_reference_draft_releases_lock_and_rejects_stale_commits(self):
        mutations = {
            "unchanged": None,
            "candidate": lambda: server.state["candidates"].update({"1": [shape(label="person")]}),
            "agent_run": lambda: server.state["agent_runs"].update({"1": {"manual": True}}),
            "topic": lambda: server.state.update(topic="camera1"),
            "video": lambda: server.state.update(video=FakeVideo()),
            "config": lambda: server.state["config"].update(llm_model="changed"),
            "credential": lambda: server.state["config"].update(llm_key="changed-key"),
            "labels": lambda: server.state["labels"].append("new-label"),
            "truth": lambda: server.state["truth"].update({"1": [shape(label="person")]}),
            "reference": lambda: server.state["reference_frames"].update(camera0=[1]),
            "audit_version": lambda: setattr(server, "_CANDIDATE_AUDIT_LOADED_CODE_SHA256", "changed"),
            "cancel": lambda: server.app.test_client().post("/api/run/stop"),
        }
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        old_stop = server.run_stop_event
        old_protocol = server._CANDIDATE_AUDIT_LOADED_CODE_SHA256
        try:
            for stage in ("describe", "audit"):
                for name, mutate in mutations.items():
                    with self.subTest(stage=stage, mutation=name), tempfile.TemporaryDirectory() as directory:
                        server.WORK = Path(directory)
                        server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                            "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                            "candidates": {}, "agent_runs": {}}
                        server.run_status = {"status": "idle"}
                        server.run_stop_event = threading.Event()
                        server._CANDIDATE_AUDIT_LOADED_CODE_SHA256 = old_protocol
                        entered, release, finished = (threading.Event() for _ in range(3))
                        responses, errors = [], []
                        result = {"shapes": [shape(label="person")], "candidates_added": 1,
                            "applied": [], "review_queue": [], "agent": {"stop_reason": "done"}}
                        def block_model(*args, **kwargs):
                            entered.set()
                            if not release.wait(5):
                                raise RuntimeError("测试未释放模型")
                            return ({"description": "people", "prompts": []}
                                    if stage == "describe" else ([], []))
                        def draft():
                            try:
                                responses.append(server.app.test_client().post("/api/reference_draft",
                                    json={"frame": 1, "labels": ["person"]}))
                            except Exception as error:
                                errors.append(error)
                            finally:
                                finished.set()
                        worker = threading.Thread(target=draft)
                        with patch.object(server, "describe", side_effect=(block_model if stage == "describe"
                                else lambda *args, **kwargs: {"description": "people", "prompts": []})), \
                             patch.object(server, "run", return_value=result), \
                             patch.object(server, "audit_high_risk_candidates", side_effect=(block_model
                                if stage == "audit" else lambda *args, **kwargs: ([], []))):
                            worker.start()
                            try:
                                self.assertTrue(entered.wait(2), "模型未进入阻塞点")
                                state_done, state_responses = threading.Event(), []
                                def read_state():
                                    try:
                                        state_responses.append(server.app.test_client().get("/api/state"))
                                    finally:
                                        state_done.set()
                                reader = threading.Thread(target=read_state)
                                reader.start()
                                self.assertTrue(state_done.wait(2), "模型运行时状态接口被阻塞")
                                reader.join(2)
                                self.assertEqual(state_responses[0].status_code, 200)
                                self.assertEqual(state_responses[0].json["candidates"], {})
                                self.assertEqual(state_responses[0].json["agent_runs"], {})
                                duplicate = server.app.test_client().post("/api/reference_draft",
                                    json={"frame": 1, "labels": ["person"]})
                                self.assertEqual(duplicate.status_code, 503)
                                if mutate:
                                    with server.lock:
                                        mutate()
                                expected_candidates = copy.deepcopy(server.state["candidates"])
                                expected_runs = copy.deepcopy(server.state["agent_runs"])
                            finally:
                                release.set()
                                worker.join(5)
                            self.assertTrue(finished.is_set(), "草稿线程未退出")
                            self.assertEqual(errors, [])
                            response = responses[0]
                            if name == "unchanged":
                                self.assertEqual(response.status_code, 200, response.json)
                                self.assertEqual(server.run_status["status"], "completed")
                            elif name == "cancel":
                                self.assertEqual(response.status_code, 200, response.json)
                                self.assertTrue(response.json["stopped"])
                                self.assertEqual(server.run_status["status"], "stopped")
                            else:
                                self.assertEqual(response.status_code, 409, response.json)
                                self.assertTrue(response.json["stale_review"])
                            if name != "unchanged":
                                self.assertEqual(server.state["candidates"], expected_candidates)
                                self.assertEqual(server.state["agent_runs"], expected_runs)
                                self.assertFalse((server.WORK / "state.json").exists())
                                self.assertEqual(server.state.get("candidate_revisions", {}), {})
        finally:
            server.state, server.run_status, server.WORK = old_state, old_status, old_work
            server.run_stop_event = old_stop
            server._CANDIDATE_AUDIT_LOADED_CODE_SHA256 = old_protocol

    def test_reference_draft_records_candidate_only_provenance(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            generated = shape(label="person")
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": False, "rounds": 1,
                "max_rounds": 100, "stop_reason": "no_progress", "audit": []}}
            try:
                with patch.object(server, "describe", return_value={"description": "people",
                        "prompts": [{"label": "person", "text": "person"}]}), \
                     patch.object(server, "run", return_value=result):
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"]})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["candidate_only"])
                self.assertFalse(response.json["human_truth"])
                self.assertFalse(response.json["approved_for_training"])
                self.assertEqual(len(response.json["input_sha256"]), 64)
                self.assertEqual(len(response.json["candidate_event_id"]), 64)
                self.assertEqual(len(response.json["candidate_request_id"]), 64)
                self.assertEqual(len(response.json["candidate_output_sha256"]), 64)
                self.assertEqual(len(response.json["candidate_artifact_sha256"]), 64)
                candidate = server.state["candidates"]["1"][0]
                self.assertTrue(candidate["candidate_only"])
                self.assertFalse(candidate["human_truth"])
                self.assertFalse(candidate["approved_for_training"])
                self.assertEqual(candidate["provenance"]["pipeline"], "sam_vlm_agent")
                self.assertEqual(candidate["provenance"]["candidate_request_id"],
                                 response.json["candidate_request_id"])
                self.assertEqual(candidate["provenance"]["candidate_output_sha256"],
                                 response.json["candidate_output_sha256"])
                self.assertEqual(candidate["provenance"]["sam_prompts"], [
                    {"label": "person", "text": "person"}])
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["reference_frames"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_rerun_archives_prior_candidates_and_audit_append_only(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            prior_candidate = shape(label="person", left=2, right=12)
            prior_candidates = [prior_candidate]
            prior_sha = server._sha256_json(prior_candidates)
            prior_run = {"candidate_event_id": "a" * 64,
                         "candidate_artifact_sha256": prior_sha,
                         "stop_reason": "done", "rounds": 1, "audit": [{"round": 1}]}
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {"llm_model": "test-vlm"},
                "candidates": {"1": copy.deepcopy(prior_candidates)},
                "agent_runs": {"1": copy.deepcopy(prior_run)}, "candidate_revisions": {}}
            server.run_status = {"status": "idle"}
            generated = shape(label="person", left=4, right=18)
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": False, "rounds": 1,
                "max_rounds": 100, "stop_reason": "no_progress", "audit": []}}
            try:
                with patch.object(server, "describe", return_value={"description": "people",
                        "prompts": [{"label": "person", "text": "person"}]}), \
                     patch.object(server, "run", return_value=result):
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "labels": ["person"]})
                self.assertEqual(response.status_code, 200, response.json)
                revisions = server.state["candidate_revisions"]["1"]
                self.assertEqual(len(revisions), 1)
                revision = revisions[0]
                self.assertEqual(response.json["prior_candidate_revision_id"], revision["revision_id"])
                self.assertEqual(response.json["candidate_revision_count"], 1)
                self.assertEqual(revision["prior_candidates"], prior_candidates)
                self.assertEqual(revision["prior_agent_run"], prior_run)
                self.assertEqual(revision["prior_candidate_artifact_sha256"], prior_sha)
                self.assertTrue(revision["prior_artifact_integrity_match"])
                self.assertTrue(revision["candidate_only"])
                self.assertFalse(revision["human_truth"])
                self.assertFalse(revision["approved_for_training"])
                self.assertNotEqual(response.json["candidate_event_id"],
                                    prior_run["candidate_event_id"])
                self.assertEqual(server.state["truth"], {})
                state_response = server.app.test_client().get("/api/state")
                summary = state_response.json["candidate_revision_summary"]["1"][0]
                self.assertEqual(summary["revision_id"], revision["revision_id"])
                self.assertNotIn("prior_candidates", summary)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_candidate_event_id_changes_with_output_not_stable_request(self):
        request_id = "a" * 64
        first = server.reference_candidate_event_id(request_id, "b" * 64)
        replay = server.reference_candidate_event_id(request_id, "b" * 64)
        changed = server.reference_candidate_event_id(request_id, "c" * 64)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_reference_draft_auto_discovers_visible_labels_before_sam(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "labels": ["person", "pallet", "slope"], "truth": {}, "reference_frames": {},
                "config": {"llm_model": "test-vlm"}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [
                    {"label": "person", "evidence": "present", "confidence": .95, "reason": "visible"},
                    {"label": "pallet", "evidence": "absent", "confidence": .91, "reason": "absent"},
                    {"label": "slope", "evidence": "uncertain", "confidence": .51, "reason": "occluded"}],
                "visible_labels": ["person"], "uncertain_labels": ["slope"],
                "unknown_candidates": [{"candidate_name": "rack", "evidence": "visible rack",
                    "confidence": .7, "route": "human_review_required",
                    "ontology_mutation_allowed": False}], "registry_mutated": False}
            generated = shape(label="person")
            result = {"shapes": [generated], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"stop_reason": "done", "rounds": 1, "audit": []}}
            try:
                with patch.object(server, "discover_visible_labels", return_value=discovery) as discover_mock, \
                     patch.object(server, "describe", return_value={"description": "person",
                        "prompts": [{"label": "person", "text": "person"}]}) as describe_mock, \
                     patch.object(server, "run", return_value=result) as run_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "auto_detect_visible": True})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["auto_detect_visible"])
                self.assertEqual(response.json["labels"], ["person", "slope"])
                self.assertTrue(response.json["human_review_required"])
                self.assertEqual(response.json["visible_label_discovery"]["uncertain_labels"], ["slope"])
                self.assertFalse(response.json["visible_label_discovery"]["unknown_candidates"][0]
                                 ["ontology_mutation_allowed"])
                discover_mock.assert_called_once()
                self.assertEqual(discover_mock.call_args.args[2], ["person", "pallet", "slope"])
                describe_mock.assert_not_called()
                self.assertEqual(run_mock.call_args.args[3], ["person", "slope"])
                self.assertEqual(run_mock.call_args.args[2]["prompt_source"],
                                 "deterministic_canonical_fallback")
                self.assertEqual(response.json["description_source"],
                                 "deterministic_after_visible_label_discovery")
                queue = server.state["agent_runs"]["1"]["review_queue"]
                self.assertIn({
                    "label": "slope", "reason_code": "vlm_visibility_uncertain",
                    "route": "human_review_required", "source": "visible_label_discovery",
                    "candidate_available": False}, queue)
                self.assertTrue(any(item["reason_code"] == "vlm_unknown_candidate" for item in queue))
                class_review = next(item for item in queue
                                    if item.get("source") == "candidate_class_audit")
                self.assertEqual(class_review["shape_ids"], [generated["id"]])
                self.assertEqual(class_review["candidate_artifact_sha256"],
                                 response.json["candidate_artifact_sha256"])
                self.assertEqual(server.state["truth"], {})
                self.assertEqual(server.state["reference_frames"], {})
                self.assertTrue(server.state["candidates"]["1"][0]["candidate_only"])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_draft_auto_detection_without_visible_known_label_fails_closed(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["person"],
                "truth": {}, "reference_frames": {}, "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            discovery = {"assessments": [{"label": "person", "evidence": "absent",
                "confidence": .9, "reason": "absent"}], "visible_labels": [], "uncertain_labels": [],
                "unknown_candidates": [], "registry_mutated": False}
            try:
                with patch.object(server, "discover_visible_labels", return_value=discovery), \
                     patch.object(server, "describe") as describe_mock, patch.object(server, "run") as run_mock:
                    response = server.app.test_client().post("/api/reference_draft", json={
                        "frame": 1, "auto_detect_visible": True})
                self.assertEqual(response.status_code, 400, response.json)
                self.assertIn("未发现可生成候选的已知类别", response.json["error"])
                describe_mock.assert_not_called(); run_mock.assert_not_called()
                self.assertEqual(server.state["candidates"], {})
                self.assertEqual(server.state["truth"], {})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_stop_endpoint_terminates_active_run_without_saving_partial_frame(self):
        old_state, old_status, old_work, old_stop = server.state, server.run_status, server.WORK, server.run_stop_event
        entered = threading.Event()
        responses = []
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            server.run_stop_event = threading.Event()
            description = {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                           "reference_signature": server.reference_signature(0)}
            def fake_run(*args, **kwargs):
                self.assertEqual(kwargs["max_rounds"], 5)
                entered.set()
                server.run_stop_event.wait(3)
                workflow.check_cancelled()
                raise AssertionError("run was not cancelled")
            def start():
                with server.app.test_client() as client:
                    responses.append(client.post("/api/run", json={"frame": 1, "count": 1,
                        "max_rounds": 5, "description": description}))
            worker = threading.Thread(target=start)
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    client = server.app.test_client()
                    stopped = client.post("/api/run/stop", json={})
                    self.assertEqual(stopped.json, {"status": "stopping", "stopped": True})
                    worker.join(3)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(responses[0].status_code, 200)
                self.assertTrue(responses[0].json["stopped"])
                self.assertEqual(client.get("/api/run/status").json["status"], "stopped")
                self.assertFalse(server.state["candidates"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
                self.assertFalse(client.post("/api/run/stop", json={}).json["stopped"])
            finally:
                server.run_stop_event.set()
                worker.join(3)
                server.state, server.run_status, server.WORK, server.run_stop_event = old_state, old_status, old_work, old_stop

    def test_old_description_requires_explicit_reference_preparation_then_is_reused(self):
        previous_state, previous_status, previous_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                with patch.object(server, "DESCRIPTION_INPUT_VERSION", 2):
                    old_signature = server.reference_signature(0)
                stale = {"description": "old rule", "prompts": [{"label": "a", "text": "target"}],
                         "reference_signature": old_signature}
                (server.WORK / "description.json").write_text(json.dumps(stale))
                client = server.app.test_client()
                self.assertIsNone(client.get("/api/state").json["description"])
                result = {"shapes": [], "candidates_added": 0, "applied": [], "review_queue": [],
                          "agent": {"converged": True, "rounds": 1, "stop_reason": "done"}}
                with patch.object(server, "describe", return_value={"description": "new rule",
                        "prompts": [{"label": "a", "text": "target"}]}) as describe, \
                     patch.object(server, "run", return_value=result) as run:
                    response = client.post("/api/run", json={"frame": 1, "count": 1, "description": stale})
                    self.assertEqual(response.status_code, 400, response.json)
                    describe.assert_not_called()
                    run.assert_not_called()
                    prepared = client.post("/api/describe", json={"frame": 0})
                    self.assertEqual(prepared.status_code, 200, prepared.json)
                    current = prepared.json
                    self.assertNotEqual(current["reference_signature"], old_signature)
                    self.assertEqual(current["description"], "new rule")
                    self.assertEqual(client.get("/api/state").json["description"], current)
                    self.assertTrue(server.description_cache_path(
                        current["reference_signature"]).exists())
                    for _ in range(2):
                        reused = client.post("/api/run", json={"frame": 1, "count": 1})
                        self.assertEqual(reused.status_code, 200, reused.json)
                    self.assertEqual(describe.call_count, 1)
                    self.assertEqual(run.call_count, 2)
                    for call in run.call_args_list:
                        self.assertEqual(call.kwargs["reference"]["annotations"], [shape()])
            finally:
                server.state, server.run_status, server.WORK = previous_state, previous_status, previous_work

    def test_descriptions_are_cached_per_topic_and_restored_on_camera_switch(self):
        previous_state, previous_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]},
                "truths_by_topic": {"camera1": {"0": [shape(left=3)]}},
                "reference_frames": {"camera0": [0], "camera1": [0]},
                "config": {}, "candidates": {}, "candidates_by_topic": {},
                "agent_runs": {}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
            }
            try:
                camera0 = {"description": "camera zero", "prompts": [
                    {"label": "a", "text": "zero"}],
                    "reference_signature": server.reference_signature(0)}
                server.save_reference_description(camera0)
                client = server.app.test_client()

                switched = client.post("/api/topic", json={"topic": "camera1"})
                self.assertEqual(switched.status_code, 200, switched.json)
                self.assertFalse(switched.json["description_cache_hit"])
                camera1 = {"description": "camera one", "prompts": [
                    {"label": "a", "text": "one"}],
                    "reference_signature": server.reference_signature(0)}
                server.save_reference_description(camera1)
                self.assertTrue(client.get("/api/run/readiness").json["ready"])

                restored = client.post("/api/topic", json={"topic": "camera0"})
                self.assertEqual(restored.status_code, 200, restored.json)
                self.assertTrue(restored.json["description_cache_hit"])
                self.assertEqual(restored.json["description"], camera0)
                self.assertEqual(client.get("/api/state").json["description"], camera0)
                self.assertTrue(client.get("/api/run/readiness").json["ready"])
                readiness = client.get("/api/state/summary").json["topic_readiness"]
                self.assertTrue(readiness["camera0"]["ready"])
                self.assertTrue(readiness["camera1"]["ready"])
                self.assertEqual(readiness["camera0"]["reference_frame"], 0)
                self.assertFalse(readiness["camera0"]["human_truth_verified"])

                cache_files = list((server.WORK / "descriptions").glob("*.json"))
                self.assertEqual(len(cache_files), 2)
                self.assertEqual({json.loads(path.read_text())["description"]
                                  for path in cache_files}, {"camera zero", "camera one"})
            finally:
                server.state, server.WORK = previous_state, previous_work

    def test_compact_topic_switch_loads_only_selected_frame_and_guards_lazy_reads(self):
        previous_state, previous_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            dense = [shape(index + 1) for index in range(240)]
            server.state = {
                "video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()], "2": [shape(2)]},
                "truths_by_topic": {"camera1": {}}, "reference_frames": {},
                "config": {}, "candidates": {"1": dense, "2": [shape(3)]},
                "candidates_by_topic": {}, "agent_runs": {"1": {"rounds": 2}},
                "agent_runs_by_topic": {}, "candidate_revisions": {},
                "candidate_revisions_by_topic": {}, "human_review_transactions": {},
            }
            try:
                client = server.app.test_client()
                full = client.post("/api/topic", json={"topic": "camera0"})
                compact = client.post("/api/topic", json={
                    "topic": "camera0", "frame": 0, "compact": True})
                self.assertEqual(compact.status_code, 200, compact.json)
                self.assertTrue(compact.json["compact_geometry"])
                self.assertEqual(compact.json["frame"], 0)
                self.assertEqual(compact.json["truth"], {"0": [shape()]})
                self.assertEqual(compact.json["candidates"], {})
                self.assertEqual(compact.json["truth_frames"], ["0", "2"])
                self.assertEqual(compact.json["candidate_counts"], {"1": 240, "2": 1})
                self.assertLess(len(compact.data), len(full.data) * 0.05)

                lazy = client.get("/api/frame_state/1", query_string={
                    "topic": "camera0", "path": str(FakeVideo.path)})
                self.assertEqual(lazy.status_code, 200, lazy.json)
                self.assertEqual(len(lazy.json["candidates"]["1"]), 240)
                self.assertEqual(lazy.json["agent_runs"], {"1": {"rounds": 2}})
                binding = lazy.json["candidate_review_bindings"]["1"]
                self.assertEqual(len(binding["candidates"]), 240)

                stale = client.get("/api/frame_state/1", query_string={
                    "topic": "camera1", "path": str(FakeVideo.path)})
                self.assertEqual(stale.status_code, 409)
                wrong_file = client.get("/api/frame_state/1", query_string={
                    "topic": "camera0", "path": "/tmp/other.mcap"})
                self.assertEqual(wrong_file.status_code, 409)
            finally:
                server.state, server.WORK = previous_state, previous_work

    def test_describe_releases_state_lock_and_rejects_stale_completion(self):
        old_state, old_status, old_work, old_stop = (
            server.state, server.run_status, server.WORK, server.run_stop_event)
        entered, release = threading.Event(), threading.Event()
        responses = {}
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            server.run_stop_event = threading.Event()
            def blocked_describe(*args, **kwargs):
                entered.set()
                self.assertTrue(release.wait(3))
                return {"description": "a only",
                        "prompts": [{"label": "a", "text": "a"}]}
            def request_description():
                responses["describe"] = server.app.test_client().post(
                    "/api/describe", json={"frame": 0})
            def read_state():
                responses["state"] = server.app.test_client().get("/api/state")
            worker = threading.Thread(target=request_description)
            reader = threading.Thread(target=read_state)
            try:
                with patch.object(server, "describe", side_effect=blocked_describe):
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    reader.start()
                    reader.join(1)
                    self.assertFalse(reader.is_alive(), "描述模型调用不应阻塞状态读取")
                    self.assertEqual(responses["state"].status_code, 200)
                    with server.lock:
                        server.state["config"]["llm_prompt_profile"] = "compact"
                    release.set()
                    worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(responses["describe"].status_code, 409)
                self.assertFalse((server.WORK / "description.json").exists())
                self.assertEqual(server.run_status["status"], "failed")
            finally:
                release.set()
                worker.join(3)
                reader.join(3)
                server.state, server.run_status, server.WORK, server.run_stop_event = (
                    old_state, old_status, old_work, old_stop)

    def test_reference_outline_does_not_tint_raw_floor_or_reuse_old_description(self):
        old_state = server.state
        server.state = {"video": FakeVideo(), "topic": "camera0",
                        "truth": {"0": [shape(label="地面")]}}
        try:
            original = Image.open(io.BytesIO(server.frame_image(0))).convert("RGB")
            outline = Image.open(io.BytesIO(server.overlay(0))).convert("RGB")
            center = outline.getpixel((25, 25))
            self.assertTrue(all(abs(a-b) < 45 for a, b in zip(center, original.getpixel((25, 25)))))
            self.assertGreater(outline.getpixel((2, 10))[0], outline.getpixel((2, 10))[1])
            old_signature = server.reference_signature(0)
            original_truth = server.state["truth"]["0"]
            server.state["truth"]["0"] = [shape(label="地面", left=3)]
            self.assertNotEqual(old_signature, server.reference_signature(0))
            self.assertEqual(original_truth[0]["label"], "地面")
        finally:
            server.state = old_state

    def test_clear_auto_result_removes_only_selected_frame_and_preserves_truth(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            original_truth = {"0": [shape()], "2": [shape(2)]}
            server.state = {"video": FakeVideo(), "topic": "camera0",
                "truth": original_truth.copy(), "candidates": {"1": [shape()], "2": [shape(2)]},
                "agent_runs": {"1": {"rounds": 1}, "2": {"rounds": 1}},
                "reference_frames": {"camera0": [0]}, "config": {}}
            server.run_status = {"status": "completed", "topic": "camera0",
                "file": str(FakeVideo.path), "completed": 2, "total": 2,
                "events": [{"frame": 1}, {"frame": 2}]}
            def call(frame, topic="camera0", purpose="自动标注"):
                return {"id": str(frame) + topic + purpose, "frame": frame, "topic": topic,
                    "file": str(FakeVideo.path), "purpose": purpose}
            kept_image = server.save_inspection_image(image_bytes())
            marked = io.BytesIO()
            Image.new("RGB", (40, 40), "black").save(marked, "JPEG")
            removed_image = server.save_inspection_image(marked.getvalue())
            shared_description = {"description": "floor", "prompts": [{"label": "a", "text": "floor"}],
                                  "reference_signature": server.reference_signature(0)}
            server.save_description_trace([call(0, purpose="生成真值描述")], shared_description)
            server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                "description": shared_description,
                "trace": [call(1) | {"input": {"image": {"url": kept_image}}},
                          call(2) | {"input": {"image": {"url": removed_image}}},
                          call(2, purpose="生成真值描述"), call(2, topic="camera1")],
                "steps": [{"frame": 1, "image": kept_image}, {"frame": 2, "image": removed_image}],
                "frames": [{"frame": 1}, {"frame": 2}]})
            server.save_debug([call(1), call(2), call(2, topic="camera1")])
            try:
                client = server.app.test_client()
                self.assertEqual(client.delete("/api/auto_result/1", json={"topic": "camera1"}).status_code, 400)
                server.run_status["status"] = "running"
                self.assertEqual(client.delete("/api/auto_result/1", json={"topic": "camera0"}).status_code, 503)
                server.run_status["status"] = "completed"
                response = client.delete("/api/auto_result/2", json={"topic": "camera0"})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["truth_preserved"])
                self.assertEqual(server.state["truth"], original_truth)
                self.assertEqual(list(server.state["candidates"]), ["1"])
                self.assertEqual(list(server.state["agent_runs"]), ["1"])
                self.assertEqual(list(json.loads(server.state_path().read_text())["candidates"]), ["1"])
                output = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual([item["frame"] for item in output["steps"]], [1])
                self.assertEqual([item["frame"] for item in output["frames"]], [1])
                self.assertEqual(len(output["trace"]), 3)
                self.assertEqual(len(json.loads((server.WORK / "model_debug.json").read_text())["trace"]), 2)
                inspection = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertFalse(inspection["steps"])
                self.assertFalse(inspection["trace"])
                self.assertIsNone(inspection["description"])
                kept = client.get("/api/run/inspection?topic=camera0&frame=1").json
                self.assertTrue(kept["trace"])
                self.assertFalse(any(item.get("purpose") == "生成真值描述" for item in kept["trace"]))
                self.assertEqual(kept["description"], shared_description)
                reference = client.get("/api/run/inspection?topic=camera0&frame=0").json
                self.assertEqual(reference["description"], shared_description)
                self.assertTrue(any(item["purpose"] == "生成真值描述"
                                    for item in reference["trace"]))
                self.assertEqual(server.run_status["events"], [{"frame": 1}])
                self.assertEqual(server.run_status["completed"], 1)
                self.assertTrue((server.WORK / "inspection" / kept_image.rsplit("/", 1)[-1]).exists())
                self.assertFalse((server.WORK / "inspection" / removed_image.rsplit("/", 1)[-1]).exists())
                server.save_run({"trace": [call(1), call(1, topic="camera1")],
                    "frames": [{"frame": 1}], "steps": [{"frame": 1}]})
                self.assertEqual(client.delete("/api/auto_result/1", json={"topic": "camera0"}).status_code, 200)
                self.assertEqual(server.state["candidates"], {})
                self.assertEqual(server.state["truth"], original_truth)
                legacy = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual(legacy["frames"], [])
                self.assertEqual(legacy["steps"], [])
                self.assertEqual([item["topic"] for item in legacy["trace"]], ["camera1"])
                self.assertEqual(server.run_status["status"], "idle")
                self.assertEqual(server.run_status["events"], [])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_inspection_preserves_segmentation_protocol_retry_and_edited_result(self):
        previous_state, previous_status, previous_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            decisions = [{"status": "continue", "operations": [
                {"operation": "erase_region", "shape_ids": [1], "reason": "edge"}]},
                {"status": "continue", "reason": "trim edge", "operations": [
                {"operation": "erase_region", "shape_ids": [1], "reason": "edge",
                 "polygon": [[12, 0], [39, 0], [39, 39], [12, 39]]}]},
                {"status": "done", "reason": "matches reference", "operations": []}]
            try:
                with patch.object(workflow, "_sam_request", return_value={"results": [sam_result()]}), \
                     patch.object(workflow, "_chat", side_effect=decisions):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 2, "count": 1,
                        "description": {"description": "square only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                record = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual([step["phase"] for step in record["steps"]], ["generated", "edited", "finished"])
                self.assertNotEqual(record["steps"][0]["image"], record["steps"][-1]["image"])
                self.assertEqual([call["service"] for call in record["trace"]], ["SAM 3", "GPT", "GPT", "GPT"])
                reviews = record["trace"][1:]
                self.assertEqual([call["round"] for call in reviews], [1, 1, 2])
                self.assertEqual([call["review_attempt"] for call in reviews], [1, 2, 1])
                self.assertEqual([call["status"] for call in reviews], ["rejected", "succeeded", "succeeded"])
                self.assertIn("polygon", reviews[0]["error"])
                self.assertIn("polygon", reviews[1]["retry_feedback"])
                self.assertEqual(reviews[0]["output"]["parsed"], decisions[0])
                self.assertEqual(reviews[0]["input"]["images"], reviews[1]["input"]["images"])
                self.assertEqual(server.state["agent_runs"]["2"]["rounds"], 2)
                for call in reviews:
                    images = call["input"]["images"]
                    self.assertEqual(len(images), 5)
                    self.assertEqual(images[3]["url"], server.save_inspection_image(server.frame_image(0)))
                    self.assertEqual(images[4]["url"], server.save_inspection_image(server.overlay(0)))
                    self.assertIn(json.dumps([shape()], ensure_ascii=False), call["input"]["instruction"])
                self.assertNotEqual(reviews[1]["input"]["images"][1]["url"], reviews[2]["input"]["images"][1]["url"])
                self.assertEqual(record["steps"][-1]["image"], reviews[2]["input"]["images"][1]["url"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
            finally:
                server.state, server.run_status, server.WORK = previous_state, previous_status, previous_work

    def test_inspection_exposes_model_images_and_filters_frame_and_topic(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "config": {}}
            image = image_bytes()
            try:
                with patch.object(workflow, "_sam_request", return_value={"results": [sam_result()] }):
                    events = []
                    with workflow.trace_calls(events.append, server.save_inspection_image):
                        workflow.sam_text(image, "floor", {})
                self.assertEqual(events[-1]["status"], "succeeded")
                url = events[-1]["output"]["segmentation"]["url"]
                with server.app.test_client().get(url) as image_response:
                    self.assertEqual(image_response.status_code, 200)
                self.assertIn("/api/inspection/image/", events[-1]["input"]["image"]["url"])
                event = events[-1] | {"frame": 2, "topic": "camera0", "file": str(FakeVideo.path), "purpose": "自动标注"}
                other = event | {"frame": 1, "id": "another"}
                server.save_run({"file": str(FakeVideo.path), "topic": "camera0",
                    "reference": 0, "description": {"description": "floor"},
                    "trace": [event, other], "steps": [{"frame": 2, "phase": "finished", "image": url}]})
                client = server.app.test_client()
                result = client.get("/api/run/inspection?topic=camera0&frame=2").json
                self.assertEqual(result["frames"], [1, 2])
                self.assertEqual([item["frame"] for item in result["trace"]], [2])
                self.assertEqual(result["steps"][0]["phase"], "finished")
                self.assertFalse(client.get("/api/run/inspection?topic=camera1&frame=2").json["trace"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_missing_or_stale_description_rejected_without_model_calls_or_record_changes(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            record = server.WORK / "last_run.json"
            record.write_text('{"previous":true}')
            try:
                with patch.object(server, "describe") as describe, patch.object(server, "run") as run:
                    client = server.app.test_client()
                    missing = client.post("/api/run", json={"frame": 1, "count": 1})
                    self.assertEqual(missing.status_code, 400)
                    stale = {"description": "old", "prompts": [{"label": "a", "text": "square"}],
                             "reference_signature": "stale"}
                    (server.WORK / "description.json").write_text(json.dumps(stale))
                    response = client.post("/api/run", json={"frame": 1, "count": 1})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("生成真值描述", response.json["error"])
                    describe.assert_not_called()
                    run.assert_not_called()
                    self.assertEqual(record.read_text(), '{"previous":true}')
                    self.assertEqual(server.run_status, {"status": "idle"})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_debug_persists_calls_without_overwriting_failed_candidate_audit(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {"llm_url": "http://example.test", "llm_model": "test",
                           "llm_key": "secret-token"}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            try:
                responses = [io.BytesIO(b'{"results":[]}'), io.BytesIO(json.dumps({
                    "choices": [{"message": {"content": json.dumps({"status": "needs_review",
                        "reason": "secret-token no masks", "operations": []})}}]}).encode())]
                with patch.object(workflow.urllib.request, "urlopen", side_effect=responses):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 1, "count": 1,
                        "description": {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                debug = client.get("/api/run/debug").json
                self.assertEqual(len(debug["trace"]), 2)
                self.assertEqual(debug["trace"][0]["frame"], 1)
                self.assertEqual(debug["trace"][0]["output"]["returned_count"], 0)
                self.assertNotIn("secret-token", json.dumps(debug))
                self.assertIn("[REDACTED]", json.dumps(debug))
                self.assertEqual(json.loads((server.WORK / "model_debug.json").read_text()), debug)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_reference_only_range_rejected_without_model_calls_or_overwriting_audit(self):
        previous = server.state
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(server, "WORK", Path(directory)), \
             patch.object(server, "describe") as describe, \
             patch.object(server, "run") as run:
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]}}
            path = Path(directory) / "last_run.json"
            path.write_text('{"previous": true}')
            try:
                response = server.app.test_client().post("/api/run", json={"frame": 0, "count": 1})
                self.assertEqual(response.status_code, 400)
                self.assertIn("没有待处理帧", response.json["error"])
                describe.assert_not_called()
                run.assert_not_called()
                self.assertEqual(json.loads(path.read_text()), {"previous": True})
            finally:
                server.state = previous

    def test_auto_labels_can_run_unreferenced_classes_as_a_separate_batch(self):
        previous = server.state
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "WORK", Path(directory)):
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a", "b"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            try:
                before = server.reference_signature(0)
                client = server.app.test_client()
                response = client.post("/api/auto_labels", json={"labels": ["a", "b"]})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["labels"], ["a", "b"])
                self.assertNotEqual(before, server.reference_signature(0))
                state = client.get("/api/state").json
                self.assertEqual(state["auto_labels"], ["a", "b"])
                self.assertEqual(state["reference_only_labels"], ["a"])
                separate = client.post("/api/auto_labels", json={"labels": ["b"]})
                self.assertEqual(separate.status_code, 200, separate.json)
                self.assertEqual(separate.json["labels"], ["b"])
            finally:
                server.state = previous

    def test_separate_class_batch_preserves_other_class_candidates(self):
        existing = [{"label": "pallet", "points": [[0, 0], [2, 0], [2, 2]]},
                    {"label": "floor", "points": [[0, 0], [3, 0], [3, 3]]}]
        generated = [{"label": "floor", "points": [[1, 1], [4, 1], [4, 4]]},
                     {"label": "ego", "points": [[2, 2], [5, 2], [5, 5]]}]
        merged = server.merge_inference_candidates(existing, generated, ["floor", "ego"])
        self.assertEqual([shape["label"] for shape in merged], ["pallet", "floor", "ego"])
        self.assertEqual(merged[1:], generated)

    def test_describe_can_generate_prompts_for_unreferenced_selected_classes(self):
        response = {"description": "Reference class a; class b has no reference contour.",
                    "prompts": [{"label": "a", "text": "class a"},
                                {"label": "b", "text": "class b"}]}
        with patch.object(workflow, "chat", return_value=response) as chat:
            result = workflow.describe({}, b"image", [shape()], ["a", "b"],
                                       include_unreferenced=True)
        self.assertEqual([item["label"] for item in result["prompts"]], ["a", "b"])
        self.assertIn('"b"', chat.call_args.args[1])
        self.assertIn("没有参考轮廓", chat.call_args.args[1])

    def test_describe_can_use_only_unreferenced_selected_classes(self):
        response = {"description": "Class b uses a generic object rule.",
                    "prompts": [{"label": "b", "text": "class b"}]}
        with patch.object(workflow, "chat", return_value=response) as chat:
            result = workflow.describe({}, b"image", [], ["b"], include_unreferenced=True)
        self.assertEqual(result["prompts"], [{"label": "b", "text": "class b"}])
        self.assertIn('"b"', chat.call_args.args[1])

    def test_one_frame_starts_at_current_frame_and_uses_reference(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = []
            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                self.assertEqual(max_rounds, 2)
                observed.append(description)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": max_rounds, "stop_reason": "done", "audit": []}}
            (server.WORK / "description.json").write_text(json.dumps({"description": "a only",
                "prompts": [{"label": "a", "text": "a"}],
                "reference_signature": server.reference_signature(0)}))
            try:
                with patch.object(server, "run", side_effect=fake_run), \
                     patch.object(server, "describe", return_value={"description": "a only",
                         "prompts": [{"label": "a", "text": "a"}]}) as describe:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "max_rounds": 2})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual([item["frame"] for item in response.json["frames"]], [1])
                self.assertEqual(observed, [{"description": "a only",
                    "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": server.reference_signature(0)}])
                describe.assert_not_called()
                self.assertIn("1", server.state["candidates"])
                self.assertEqual(server.state["truth"], {"0": [shape()]})
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_defaults_to_six_review_rounds_and_rejects_out_of_range(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = []
            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                observed.append(max_rounds)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": max_rounds, "stop_reason": "done", "audit": []}}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 1, "count": 1,
                        "description": description})
                    invalid = client.post("/api/run", json={"frame": 2, "count": 1,
                        "max_rounds": 101, "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(observed, [6])
                self.assertEqual(response.json["frames"][0]["agent"]["max_rounds"], 6)
                self.assertEqual(invalid.status_code, 400, invalid.json)
                self.assertIn("复查轮数须为", invalid.json["error"])
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_exposes_shared_vlm_wait_as_a_live_phase(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            observed = {}
            def fake_run(*args, **kwargs):
                workflow._emit_trace({"id": "busy", "service": "GPT", "status": "waiting",
                    "busy_retry_count": 2, "retry_in_seconds": 10,
                    "input": {}, "started_at": 1})
                observed.update(server.app.test_client().get("/api/run/status").json)
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                    "review_queue": [], "agent": {"converged": True, "rounds": 1,
                    "max_rounds": 6, "stop_reason": "done", "audit": []}}
            description = {"description": "a only", "prompts": [{"label": "a", "text": "a"}],
                           "reference_signature": server.reference_signature(0)}
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1, "description": description})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(observed["phase"], "vlm_waiting")
                self.assertEqual(observed["busy_retry_count"], 2)
                self.assertEqual(observed["retry_in_seconds"], 10)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_retries_one_transient_failure_then_saves_success(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {}, "agent_runs": {}}
            server.run_status = {"status": "idle"}
            attempts = []
            failed = {"shapes": [], "candidates_added": 0, "applied": [],
                "review_queue": [{"reason": "Agent 调用失败",
                                  "error": "HTTP Error 503: Service Unavailable"}],
                "agent": {"converged": False, "rounds": 0, "max_rounds": 100,
                          "stop_reason": "error", "audit": []}}
            succeeded = {"shapes": [shape()], "candidates_added": 1, "applied": [],
                "review_queue": [], "agent": {"converged": True, "rounds": 1,
                "max_rounds": 100, "stop_reason": "done", "audit": []}}
            def fake_run(*args, **kwargs):
                attempts.append(True)
                return copy.deepcopy(failed if len(attempts) == 1 else succeeded)
            try:
                with patch.object(server, "run", side_effect=fake_run):
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1,
                        "description": {"description": "a only",
                            "prompts": [{"label": "a", "text": "a"}],
                            "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(len(attempts), 2)
                self.assertFalse(response.json["frames"][0]["failed"])
                self.assertEqual(response.json["frames"][0]["agent"]["transient_retry_count"], 1)
                self.assertIn("1", server.state["candidates"])
                self.assertTrue(any(event["phase"] == "retrying"
                                    for event in server.run_status["events"]))
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_does_not_retry_deterministic_failure_and_preserves_candidates(self):
        old_state, old_status, old_work = server.state, server.run_status, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            existing = [shape()]
            server.state = {"video": FakeVideo(), "topic": "camera0", "labels": ["a"],
                "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                "config": {}, "candidates": {"1": copy.deepcopy(existing)},
                "agent_runs": {"1": {"stop_reason": "done"}}}
            server.run_status = {"status": "idle"}
            failed = {"shapes": [], "candidates_added": 0, "applied": [],
                "review_queue": [{"reason": "Agent 调用失败",
                                  "error": "GPT 未评估 SAM 3 点提示结果"}],
                "agent": {"converged": False, "rounds": 1, "max_rounds": 100,
                          "stop_reason": "error", "audit": []}}
            try:
                with patch.object(server, "run", return_value=copy.deepcopy(failed)) as run_mock:
                    response = server.app.test_client().post("/api/run", json={
                        "frame": 1, "count": 1,
                        "description": {"description": "a only",
                            "prompts": [{"label": "a", "text": "a"}],
                            "reference_signature": server.reference_signature(0)}})
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(run_mock.call_count, 1)
                self.assertTrue(response.json["frames"][0]["failed"])
                self.assertEqual(response.json["frames"][0]["agent"]["transient_retry_count"], 0)
                self.assertEqual(server.state["candidates"]["1"], existing)
            finally:
                server.state, server.run_status, server.WORK = old_state, old_status, old_work

    def test_run_persists_audit_and_topic_switch(self):
        old_state = server.state
        old_status = server.run_status
        old_work = server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "video_path": str(FakeVideo.path),
                "topic": "camera0", "labels": ["a"], "truth": {"0": [shape()]},
                "reference_frames": {"camera0": [0]}, "config": {},
                "candidates": {}, "candidates_by_topic": {}, "agent_runs": {},
                "agent_runs_by_topic": {}}
            server.run_status = {"status": "idle"}
            audit = [{"round": 1, "decision": {"status": "done", "operations": []},
                      "tool_results": []}]

            def fake_run(config, image, description, labels, max_rounds, on_progress, reference):
                on_progress({"phase": "observing", "round": 1, "audit": audit,
                             "review_queue": [], "shapes": [shape()]})
                stored = json.loads((server.WORK / "last_run.json").read_text())
                self.assertEqual(stored["current"]["audit"], audit)
                status = server.app.test_client().get("/api/run/status").json
                self.assertEqual(status["status"], "running")
                self.assertEqual(status["frame"], 1)
                self.assertEqual(status["shape_count"], 1)
                self.assertEqual(status["events"][-1]["phase"], "observing")
                return {"shapes": [shape()], "candidates_added": 1, "applied": [],
                        "review_queue": [], "agent": {"converged": True,
                        "rounds": 1, "max_rounds": max_rounds,
                        "stop_reason": "done", "audit": audit}}

            try:
                with patch.object(server, "run", side_effect=fake_run):
                    client = server.app.test_client()
                    response = client.post("/api/run", json={"frame": 0,
                        "count": 2, "max_rounds": 2,
                        "description": {"description": "a only", "prompts": [{"label": "a", "text": "square"}],
                                        "reference_signature": server.reference_signature(0)}})
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertEqual(client.get("/api/run/status").json["status"], "completed")
                    self.assertEqual(response.json["frames"][0]["frame"], 1)
                    self.assertEqual(json.loads(server.state_path().read_text())
                        ["agent_runs"]["1"]["audit"], audit)
                    client.post("/api/topic", json={"topic": "camera1"})
                    self.assertEqual(client.get("/api/state").json["agent_runs"], {})
                    client.post("/api/topic", json={"topic": "camera0"})
                    self.assertEqual(client.get("/api/state").json["agent_runs"]["1"]["audit"], audit)
            finally:
                server.state = old_state
                server.run_status = old_status
                server.WORK = old_work

    def test_run_status_exposes_live_timing_and_sequence_contract(self):
        old_status = server.run_status
        try:
            server.run_status = {"status": "idle"}
            with patch.object(server.time, "time", side_effect=[100.0, 103.0]):
                server.update_run_status(status="running", phase="discovering",
                                         started_at=90.0, total=1, completed=0)
                status = server.app.test_client().get("/api/run/status").json
            self.assertEqual(status["progress_contract_version"], 2)
            self.assertEqual(status["status_sequence"], 1)
            self.assertEqual(status["elapsed_seconds"], 13.0)
            self.assertEqual(status["phase_elapsed_seconds"], 3.0)
            self.assertEqual(status["phase_started_at"], 100.0)
            with patch.object(server.time, "time", return_value=104.0):
                server.update_run_status(shape_count=2)
            self.assertEqual(server.run_status["status_sequence"], 2)
            self.assertEqual(server.run_status["phase_started_at"], 100.0)
            server.run_status.update(status="completed", updated_at=105.0)
            with patch.object(server.time, "time", return_value=999.0):
                completed = server.app.test_client().get("/api/run/status").json
            self.assertEqual(completed["elapsed_seconds"], 15.0)
            self.assertEqual(completed["phase_elapsed_seconds"], 5.0)
            self.assertEqual(completed["observed_at"], 999.0)
        finally:
            server.run_status = old_status

    def test_state_json_is_streamed_and_compact(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": None, "labels": ["地面"],
                            "truth": {"1": [shape()]}, "config": {}}
            try:
                server.save()
                raw = server.state_path().read_text(encoding="utf-8")
                self.assertNotIn("\n", raw)
                self.assertEqual(server._read_json(server.state_path())["labels"], ["地面"])
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_file_sessions_migrate_once_and_are_not_rewritten_for_current_state(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            active = str((Path(directory) / "active.mcap").resolve())
            inactive = str((Path(directory) / "inactive.mcap").resolve())
            legacy = {
                "video_path": active, "topic": "camera0", "labels": ["a"],
                "truth": {"1": [shape()]}, "reference_frames": {}, "config": {},
                "candidates": {}, "agent_runs": {},
                "file_sessions": {
                    active: {"topic": "camera0", "truth": {"old": [shape(7)]}},
                    inactive: {"topic": "camera1", "truth": {"2": [shape(2)]}},
                },
            }
            server._write_json(server.state_path(), legacy)
            server.state = {"video": None, "topic": None, "labels": [], "truth": {},
                            "reference_frames": {}, "config": {}, "candidates": {},
                            "agent_runs": {}}
            try:
                server.load()
                self.assertNotIn(active, server.state["file_sessions"])
                self.assertIn(inactive, server.state["file_sessions"])
                self.assertTrue(server.file_sessions_dirty)
                server.save()
                current = server._read_json(server.state_path())
                sessions = server._read_json(server.file_sessions_path())
                self.assertNotIn("file_sessions", current)
                self.assertEqual(sessions[inactive]["truth"]["2"][0]["id"], 2)
                session_hash = server.file_sessions_path().read_bytes()
                server.state["labels"].append("b")
                server.save()
                self.assertEqual(server.file_sessions_path().read_bytes(), session_hash)
                self.assertEqual(server._read_json(server.state_path())["labels"], ["a", "b"])
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty

    def test_run_readiness_rejects_stale_description_then_accepts_current_reference(self):
        old_state, old_work = server.state, server.WORK
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            server.state = {"video": FakeVideo(), "video_path": str(FakeVideo.path),
                            "topic": "camera0", "labels": ["a"], "auto_labels": ["a"],
                            "truth": {"0": [shape()]}, "reference_frames": {"camera0": [0]},
                            "config": {}, "candidates": {}, "agent_runs": {}}
            try:
                client = server.app.test_client()
                missing = client.get("/api/run/readiness").json
                self.assertFalse(missing["ready"])
                self.assertEqual(missing["code"], "description_missing")
                (server.WORK / "description.json").write_text(json.dumps({
                    "description": "old", "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": "stale"}), encoding="utf-8")
                stale = client.get("/api/run/readiness").json
                self.assertFalse(stale["ready"])
                self.assertEqual(stale["code"], "description_stale")
                (server.WORK / "description.json").write_text(json.dumps({
                    "description": "current", "prompts": [{"label": "a", "text": "a"}],
                    "reference_signature": server.reference_signature(0)}), encoding="utf-8")
                ready = client.get("/api/run/readiness").json
                self.assertTrue(ready["ready"])
                self.assertEqual(ready["reference_frame"], 0)
                self.assertEqual(ready["labels"], ["a"])
            finally:
                server.state, server.WORK = old_state, old_work

    def test_run_data_readiness_decodes_one_requested_frame_and_reports_failures(self):
        old_state = server.state
        video = FakeVideo()
        calls = []
        original_frame = video.frame

        def tracked_frame(topic, index):
            calls.append((topic, index))
            return original_frame(topic, index)

        video.frame = tracked_frame
        server.state = {"video": video, "topic": "camera0"}
        try:
            client = server.app.test_client()
            ready = client.post("/api/run/data-readiness", json={
                "frames": [2, 1]}).json
            self.assertTrue(ready["ready"])
            self.assertEqual(ready["frame"], 2)
            self.assertEqual((ready["width"], ready["height"]), (40, 40))
            self.assertEqual(ready["scope"], "first_requested_frame")
            self.assertEqual(calls, [("camera0", 2)])

            video.frame = lambda topic, index: (_ for _ in ()).throw(
                ValueError("unsupported encoding"))
            failed = client.post("/api/run/data-readiness", json={
                "frame": 1, "count": 3}).json
            self.assertFalse(failed["ready"])
            self.assertEqual(failed["code"], "frame_decode_failed")
            self.assertIn("unsupported encoding", failed["message"])

            invalid = client.post("/api/run/data-readiness", json={
                "frame": 3, "count": 1}).json
            self.assertFalse(invalid["ready"])
            self.assertEqual(invalid["code"], "frame_out_of_range")
        finally:
            server.state = old_state

    def test_file_activation_probes_first_frame_before_mutating_current_dataset(self):
        old_state, old_work, old_dirty = server.state, server.WORK, server.file_sessions_dirty
        with tempfile.TemporaryDirectory() as directory:
            server.WORK = Path(directory)
            current = FakeVideo()
            current.path = Path(directory) / "current.mcap"
            server.state = {
                "video": current, "video_path": str(current.path), "topic": "camera0",
                "labels": ["a"], "truth": {"0": [shape()]}, "truths_by_topic": {},
                "reference_frames": {"camera0": [0]}, "config": {}, "candidates": {},
                "candidates_by_topic": {}, "agent_runs": {}, "agent_runs_by_topic": {},
                "candidate_revisions": {}, "candidate_revisions_by_topic": {},
                "file_sessions": {},
            }
            bad = FakeVideo()
            bad.path = Path(directory) / "bad.mcap"
            bad.frame = lambda topic, index: (_ for _ in ()).throw(
                ValueError("damaged payload"))
            try:
                with patch.object(server, "Video", return_value=bad):
                    with self.assertRaisesRegex(ValueError, "damaged payload"):
                        server.activate_file({"path": str(bad.path)})
                self.assertIs(server.state["video"], current)
                self.assertEqual(server.state["truth"], {"0": [shape()]})
                self.assertEqual(server.state["file_sessions"], {})
                self.assertFalse(server.state_path().exists())

                good = FakeVideo()
                good.path = Path(directory) / "good.mcap"
                with patch.object(server, "Video", return_value=good):
                    loaded = server.activate_file({"path": str(good.path)})
                self.assertTrue(loaded["data_compatibility"]["ready"])
                self.assertEqual(loaded["data_compatibility"]["frame"], 0)
                self.assertEqual(loaded["data_compatibility"]["width"], 40)
                self.assertIn("data_probe", loaded["load_timings_ms"])
                self.assertIs(server.state["video"], good)
            finally:
                server.state, server.WORK = old_state, old_work
                server.file_sessions_dirty = old_dirty


if __name__ == "__main__":
    unittest.main()
