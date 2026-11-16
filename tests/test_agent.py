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

from source.agent import server, workflow
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
        self.assertEqual(chat.call_args.kwargs["response_schema"]["properties"]["assessments"]["minItems"], 3)

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
            {"label": "slope", "text": "ramp"},
            {"label": "ego", "text": "ego robot"},
            {"label": "other_robot", "text": "robot"},
            {"label": "cargo_floor", "text": "robot cargo platform"},
        ])

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


class ServerTests(unittest.TestCase):
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
                self.assertEqual(describe_mock.call_args.args[3], ["person", "slope"])
                self.assertEqual(run_mock.call_args.args[3], ["person", "slope"])
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
                self.assertEqual(kwargs["max_rounds"], 100)
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
                    for _ in range(2):
                        reused = client.post("/api/run", json={"frame": 1, "count": 1})
                        self.assertEqual(reused.status_code, 200, reused.json)
                    self.assertEqual(describe.call_count, 1)
                    self.assertEqual(run.call_count, 2)
                    for call in run.call_args_list:
                        self.assertEqual(call.kwargs["reference"]["annotations"], [shape()])
            finally:
                server.state, server.run_status, server.WORK = previous_state, previous_status, previous_work

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
                self.assertEqual(max_rounds, 100)
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


if __name__ == "__main__":
    unittest.main()
