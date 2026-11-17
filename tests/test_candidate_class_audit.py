"""参考帧盲审与点选 SAM 配置的聚焦回归。"""
import copy
import hashlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from source.agent import server
from source.agent import candidate_class_audit as audit


class CandidateClassAuditTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"id": 7, "label": "ego", "score": .8,
                          "points": [[10, 10], [30, 10], [30, 30], [10, 30]]}
        output = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(output, "PNG")
        self.image = output.getvalue()
        self.labels = ["ego", "person", "other_robot"]

    def decision(self, label="ego", confidence=.96, classification="known_label"):
        return {"classification": classification, "observed_label": label,
                "candidate_supported": classification == "known_label", "confidence": confidence,
                "reason_code": "visible_known_label" if classification == "known_label" else "insufficient_visual_evidence",
                "visible_cues": ["visible structure"], "reason": "可见结构支持判断"}

    def test_blind_prompt_and_no_mutation(self):
        original = copy.deepcopy(self.candidate)
        def chat(config, prompt, image, **kwargs):
            self.assertNotIn("current_label", prompt)
            self.assertNotIn('"label": "ego"', prompt)
            self.assertEqual(len(kwargs["extra_images"]), 1)
            return self.decision("person")
        result = audit.audit_candidate_class_blind({}, self.image, self.candidate, self.labels, {}, chat)
        self.assertEqual(self.candidate, original)
        self.assertIsNone(result["operation_proposal"])
        self.assertEqual(result["route"], "human_review_required")
        self.assertFalse(result["crop"]["current_label_shown_to_model"])
        self.assertEqual(result["candidate_sha256"], audit.candidate_fingerprint(original))
        changed = copy.deepcopy(original)
        changed["points"][0][0] += 1
        self.assertNotEqual(result["candidate_sha256"], audit.candidate_fingerprint(changed))

    def test_review_routes_and_protocol(self):
        for decision, route in [(self.decision(), "candidate_preserved"),
                                (self.decision(confidence=.4), "human_review_required"),
                                (self.decision(None, .3, "uncertain"), "human_review_required"),
                                (self.decision(None, .3, "conflict"), "human_review_required")]:
            with self.subTest(decision=decision):
                result = audit.audit_candidate_class_blind({}, self.image, self.candidate, self.labels, {}, lambda *a, **k: decision)
                self.assertEqual(result["route"], route)
        with self.assertRaises(audit.CandidateClassProtocolError):
            audit.validate_blind_candidate_class_decision({}, self.labels)

    def test_high_risk_batch_reports_bounded_progress(self):
        progress = []
        audits, queue = audit.audit_high_risk_candidates(
            {}, self.image, [self.candidate, {**self.candidate, "id": 8, "label": "pallet"}],
            self.labels + ["pallet"], {}, lambda *args, **kwargs: self.decision(),
            on_progress=lambda completed, total, candidate, result:
                progress.append((completed, total, candidate["id"], result["route"])))
        self.assertEqual(len(audits), 1)
        self.assertEqual(queue, [])
        self.assertEqual(progress, [(1, 1, 7, "candidate_preserved")])

    def test_holes_and_invalid_coordinates(self):
        self.candidate["holes"] = [[[15, 15], [25, 15], [25, 25], [15, 25]]]
        evidence = audit.prepare_candidate_evidence(self.image, self.candidate)
        with Image.open(io.BytesIO(evidence["marked"])) as image:
            self.assertTrue(all(value > 230 for value in image.getpixel((20, 20))))
        self.candidate["points"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            audit.prepare_candidate_evidence(self.image, self.candidate)

    def test_reference_endpoint_routes_failures_and_binds_artifact(self):
        for outcome in [self.decision("person"), self.decision(None, .2, "uncertain"),
                        self.decision(None, .2, "conflict"), {}, RuntimeError("服务不可用")]:
            state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                     "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                     "reference_frames": {}, "candidates": {}, "agent_runs": {}}
            result = {"shapes": [copy.deepcopy(self.candidate)], "agent": {"stop_reason": "done"},
                      "review_queue": [], "applied": [], "candidates_added": 1}
            options = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
                 patch.object(server, "frame_image", return_value=self.image), \
                 patch.object(server, "describe", return_value={}), patch.object(server, "run", return_value=result), \
                 patch.object(server, "chat", **options), patch.object(server, "save"), \
                 patch.object(server, "archive_candidate_revision", return_value=None):
                response = server.app.test_client().post("/api/reference_draft", json={"frame": 0, "labels": ["ego"]})
                self.assertEqual(response.status_code, 200, response.json)
                payload = response.json
                self.assertEqual(state["truth"], {})
                self.assertEqual(payload["candidates"][0]["label"], "ego")
                entry = payload["agent"]["review_queue"][0]
                self.assertEqual(entry["shape_ids"], [7])
                self.assertEqual(entry["candidate_sha256"], audit.candidate_fingerprint(self.candidate))
                self.assertEqual(entry["candidate_artifact_sha256"], server._sha256_json(payload["candidates"]))
                self.assertEqual(payload["candidate_artifact_sha256"], entry["candidate_artifact_sha256"])

    def test_reference_endpoint_can_publish_candidate_before_deferred_class_audit(self):
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {}, "agent_runs": {}}
        result = {"shapes": [copy.deepcopy(self.candidate)], "agent": {"stop_reason": "done"},
                  "review_queue": [], "applied": [], "candidates_added": 1}
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "describe", return_value={"prompts": [{"label": "ego", "text": "ego robot"}]}), \
             patch.object(server, "run", return_value=result), patch.object(server, "chat") as chat, \
             patch.object(server, "save"), patch.object(server, "archive_candidate_revision", return_value=None):
            response = server.app.test_client().post("/api/reference_draft", json={
                "frame": 0, "labels": ["ego"], "defer_class_audit": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["candidate_class_audit_deferred"])
        self.assertEqual(response.json["candidate_class_audit_total"], 1)
        self.assertEqual(response.json["agent"]["candidate_class_audit_status"], "pending")
        self.assertEqual(response.json["agent"]["candidate_class_audits"], [])
        self.assertTrue(response.json["candidate_only"])
        self.assertFalse(response.json["human_truth"])
        self.assertFalse(response.json["approved_for_training"])
        self.assertEqual(state["candidates"]["0"], response.json["candidates"])
        self.assertEqual(state["agent_runs"]["0"]["candidate_class_audit_status"], "pending")
        chat.assert_not_called()

    def test_reference_endpoint_can_publish_guarded_sam_before_agent_review(self):
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {}, "agent_runs": {}}
        result = {"shapes": [copy.deepcopy(self.candidate)],
                  "agent": {"stop_reason": "review_deferred", "max_rounds": 1,
                            "candidate_filters": [], "sam_generation": {"mode": "multi_prompt"}},
                  "review_queue": [], "applied": [], "candidates_added": 1}
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "describe", return_value={"prompts": [{"label": "ego", "text": "ego robot"}]}), \
             patch.object(server, "generate_only", return_value=result) as generate, \
             patch.object(server, "run") as run, patch.object(server, "chat") as chat, \
             patch.object(server, "save"), patch.object(server, "archive_candidate_revision", return_value=None):
            response = server.app.test_client().post("/api/reference_draft", json={
                "frame": 0, "labels": ["ego"], "defer_agent_review": True,
                "defer_class_audit": True})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["agent_review_deferred"])
        self.assertEqual(response.json["agent_review_status"], "pending")
        self.assertEqual(response.json["agent"]["agent_review_status"], "pending")
        self.assertEqual(response.json["agent"]["candidate_class_audit_status"],
                         "blocked_on_agent_review")
        self.assertTrue(response.json["candidate_only"])
        self.assertFalse(response.json["human_truth"])
        self.assertFalse(response.json["approved_for_training"])
        self.assertEqual(state["truth"], {})
        generate.assert_called_once()
        run.assert_not_called()
        chat.assert_not_called()

    def test_deferred_agent_refinement_is_compare_and_swap_and_never_truth(self):
        candidate = copy.deepcopy(self.candidate)
        artifact = server._sha256_json([candidate])
        image_hash = hashlib.sha256(self.image).hexdigest()
        prior_run = {"agent_review_status": "pending", "agent_review_deferred": True,
                     "candidate_class_audit_status": "blocked_on_agent_review",
                     "candidate_artifact_sha256": artifact, "draft_labels": ["ego"],
                     "max_rounds": 1, "candidate_filters": [],
                     "sam_generation": {"mode": "multi_prompt"}, "review_queue": [],
                     "candidate_class_audits": [], "input_sha256": image_hash}
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {"0": [copy.deepcopy(candidate)]},
                 "agent_runs": {"0": copy.deepcopy(prior_run)},
                 "candidate_review_status": {}, "candidate_revisions": {}}
        refined = copy.deepcopy(candidate)
        refined["points"][0][0] = 9
        result = {"shapes": [refined], "agent": {"converged": True, "rounds": 1,
                  "max_rounds": 1, "stop_reason": "done", "audit": [{"round": 1}],
                  "candidate_filters": [], "sam_generation": {"mode": "multi_prompt"}},
                  "review_queue": [], "applied": [], "candidates_added": 1}
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "review_existing", return_value=result) as review, \
             patch.object(server, "save"), patch.object(server, "archive_candidate_revision", return_value=None):
            response = server.app.test_client().post("/api/reference_candidates/refine", json={
                "frame": 0, "topic": "camera",
                "expected_candidate_artifact_sha256": artifact,
                "expected_frame_sha256": image_hash})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["candidate_mutated"])
        self.assertTrue(response.json["candidate_only"])
        self.assertFalse(response.json["human_truth"])
        self.assertFalse(response.json["approved_for_training"])
        self.assertEqual(state["truth"], {})
        self.assertEqual(state["reference_frames"], {})
        self.assertEqual(state["candidates"]["0"][0]["points"], refined["points"])
        self.assertTrue(state["candidates"]["0"][0]["candidate_only"])
        self.assertFalse(state["candidates"]["0"][0]["human_truth"])
        self.assertFalse(state["candidates"]["0"][0]["approved_for_training"])
        self.assertEqual(state["candidates"]["0"][0]["provenance"]["agent_review_status"],
                         "complete")
        self.assertEqual(state["agent_runs"]["0"]["agent_review_status"], "complete")
        self.assertEqual(state["agent_runs"]["0"]["candidate_class_audit_status"], "pending")
        review.assert_called_once()

    def test_deferred_agent_refinement_never_overwrites_a_changed_candidate(self):
        candidate = copy.deepcopy(self.candidate)
        artifact = server._sha256_json([candidate])
        image_hash = hashlib.sha256(self.image).hexdigest()
        prior_run = {"agent_review_status": "pending", "agent_review_deferred": True,
                     "candidate_artifact_sha256": artifact, "draft_labels": ["ego"],
                     "max_rounds": 1, "candidate_filters": [],
                     "sam_generation": {"mode": "multi_prompt"}, "review_queue": []}
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {"0": [copy.deepcopy(candidate)]},
                 "agent_runs": {"0": copy.deepcopy(prior_run)},
                 "candidate_review_status": {}, "candidate_revisions": {}}
        result = {"shapes": [copy.deepcopy(candidate)],
                  "agent": {"converged": True, "rounds": 1, "max_rounds": 1,
                            "stop_reason": "done", "audit": [], "candidate_filters": [],
                            "sam_generation": {"mode": "multi_prompt"}},
                  "review_queue": [], "applied": [], "candidates_added": 1}
        def change_candidate(*_args, **_kwargs):
            state["candidates"]["0"][0]["score"] = .1
            return result
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "review_existing", side_effect=change_candidate), \
             patch.object(server, "save") as save, \
             patch.object(server, "archive_candidate_revision") as archive:
            response = server.app.test_client().post("/api/reference_candidates/refine", json={
                "frame": 0, "topic": "camera",
                "expected_candidate_artifact_sha256": artifact,
                "expected_frame_sha256": image_hash})
        self.assertEqual(response.status_code, 409, response.json)
        self.assertTrue(response.json["stale_review"])
        self.assertEqual(state["candidates"]["0"][0]["score"], .1)
        self.assertEqual(state["agent_runs"]["0"], prior_run)
        self.assertEqual(state["truth"], {})
        save.assert_not_called()
        archive.assert_not_called()

    def test_merge_keeps_existing_ids_and_renumbers_collisions_without_mutation(self):
        existing = [{"id": 3, "label": "floor"}]
        generated = [{"id": 3, "label": "ego"}, {"id": 4, "label": "person"}]
        original = copy.deepcopy(generated)
        merged, id_map = server.merge_inference_candidates(
            existing, generated, ["ego", "person"], return_id_map=True)
        self.assertEqual(merged[0], existing[0])
        self.assertEqual(len({shape["id"] for shape in merged}), 3)
        self.assertEqual(id_map, {3: [4], 4: [5]})
        self.assertEqual(generated, original)
        result = {"agent": {"audit": [{"candidate_id": 3}], "history": [{"shape_ids": [3]}]},
                  "review_queue": [{"candidate_id": 3, "shape_ids": [3]}],
                  "applied": [{"shape_ids": [3], "result_ids": [3]}]}
        server.bind_merged_candidate_references(result, id_map)
        self.assertEqual(result["review_queue"][0]["candidate_id"], 4)
        self.assertEqual(result["review_queue"][0]["original_candidate_id"], 3)
        self.assertEqual(result["agent"]["audit"][0]["candidate_id"], 4)
        self.assertEqual(result["agent"]["history"][0]["shape_ids"], [4])
        self.assertEqual(result["agent"]["history"][0]["original_shape_ids"], [3])
        self.assertEqual(result["agent"]["candidate_id_mapping"], [
            {"original_id": 3, "final_ids": [4]}, {"original_id": 4, "final_ids": [5]}])
        self.assertEqual(generated, original)

    def test_candidate_audit_endpoint_persists_only_bound_audit(self):
        candidate = copy.deepcopy(self.candidate)
        candidates = [candidate]
        artifact = server._sha256_json(candidates)
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {"0": candidates},
                 "agent_runs": {"0": {"stop_reason": "done", "review_queue": []}}}
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "chat", return_value=self.decision()), patch.object(server, "save") as save, \
             patch.object(server, "run") as run, patch.object(server, "describe") as describe:
            response = server.app.test_client().post("/api/reference_candidates/audit", json={
                "frame": 0, "topic": "camera",
                "expected_candidate_artifact_sha256": artifact,
                "expected_frame_sha256": hashlib.sha256(self.image).hexdigest()})
            self.assertEqual(response.status_code, 200, response.json)
            payload = response.json
            self.assertEqual(payload["candidates"], candidates)
            record = payload["agent"]["candidate_class_audits"][0]
            self.assertEqual(record["candidate_sha256"], audit.candidate_fingerprint(candidate))
            self.assertEqual(record["audit_version"], server.candidate_class_audit_version())
            self.assertEqual(payload["agent"]["candidate_class_audit_version"], record["audit_version"])
            self.assertEqual(payload["agent"]["candidate_class_audit_status"], "complete")
            self.assertEqual(payload["agent"]["review_queue"], [])
            self.assertTrue(payload["candidate_only"])
            self.assertFalse(payload["human_truth"])
            self.assertEqual(payload["binding"]["candidates"][0]["candidate_fingerprint_sha256"],
                             record["candidate_sha256"])
            self.assertEqual(payload["candidate_review_bindings"]["0"]["candidates"][0]["candidate_sha256"],
                             server._sha256_json(candidate))
            save.assert_called_once()
            run.assert_not_called()
            describe.assert_not_called()
        for blocked in ("truth", "reference"):
            state["truth"] = {"0": []} if blocked == "truth" else {}
            state["reference_frames"] = {"camera": [0]} if blocked == "reference" else {}
            with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
                 patch.object(server, "chat") as chat, patch.object(server, "save") as save:
                response = server.app.test_client().post("/api/reference_candidates/audit", json={
                    "frame": 0, "topic": "camera", "candidate_id": 7,
                    "expected_candidate_artifact_sha256": artifact})
                self.assertEqual(response.status_code, 400, response.json)
                chat.assert_not_called()
                save.assert_not_called()

    def test_candidate_audit_endpoint_reuses_only_exact_persisted_receipt(self):
        candidate = copy.deepcopy(self.candidate)
        candidates = [candidate]
        artifact = server._sha256_json(candidates)
        input_sha = hashlib.sha256(self.image).hexdigest()
        state = {
            "video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
            "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
            "reference_frames": {}, "candidates": {"0": candidates},
            "agent_runs": {}, "candidate_review_status": {},
        }
        with patch.object(server, "state", state):
            audit_version = server.candidate_class_audit_version()
        record = audit.audit_candidate_class_blind(
            {}, self.image, candidate, self.labels, {},
            lambda *_args, **_kwargs: self.decision(),
            candidate_artifact_sha256=artifact)
        record["audit_version"] = audit_version
        record["audit_sha256"] = server._sha256_json({
            key: value for key, value in record.items() if key != "audit_sha256"})
        fingerprint = record["candidate_sha256"]
        run = {
            "candidate_class_audit_status": "complete",
            "candidate_class_audit_version": audit_version,
            "candidate_artifact_sha256": artifact,
            "input_sha256": input_sha,
            "candidate_class_audits": [record],
            "review_queue": [],
        }
        review_status = {fingerprint: {
            "candidate_fingerprint": fingerprint,
            "candidate_artifact_sha256": artifact,
            "audit_sha256": record["audit_sha256"],
            "route": record["route"],
        }}
        state["agent_runs"] = {"0": run}
        state["candidate_review_status"] = {"0": review_status}
        with patch.object(server, "state", state), \
             patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "chat") as chat, patch.object(server, "save") as save:
            response = server.app.test_client().post(
                "/api/reference_candidates/audit", json={
                    "frame": 0, "topic": "camera",
                    "expected_candidate_artifact_sha256": artifact,
                    "expected_frame_sha256": input_sha})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["candidate_class_audit_cache_hit"])
        self.assertEqual(response.json["agent"], run)
        chat.assert_not_called()
        save.assert_not_called()

        state["candidate_review_status"]["0"] = {}
        with patch.object(server, "state", state), \
             patch.object(server, "run_status", {"status": "idle"}), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server, "chat") as chat, patch.object(server, "save") as save:
            response = server.app.test_client().post(
                "/api/reference_candidates/audit", json={
                    "frame": 0, "topic": "camera",
                    "expected_candidate_artifact_sha256": artifact,
                    "expected_frame_sha256": input_sha})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertTrue(response.json["candidate_class_audit_cache_hit"])
        chat.assert_not_called()
        save.assert_not_called()
        state["candidate_review_status"]["0"] = review_status

        for mutation in ("audit_hash", "binding", "version", "decision"):
            changed_run = copy.deepcopy(run)
            changed_review = copy.deepcopy(review_status)
            if mutation == "audit_hash":
                changed_run["candidate_class_audits"][0]["audit_sha256"] = "0" * 64
            elif mutation == "binding":
                changed_review[fingerprint]["route"] = "human_review_required"
            else:
                if mutation == "version":
                    changed_run["candidate_class_audit_version"] = {
                        **audit_version, "protocol_sha256": "0" * 64}
                else:
                    changed_run["candidate_class_audits"][0]["decision"] = {
                        "classification": "known_label"}
            state["agent_runs"]["0"] = changed_run
            state["candidate_review_status"]["0"] = changed_review
            with self.subTest(mutation=mutation), \
                 patch.object(server, "state", state), \
                 patch.object(server, "run_status", {"status": "idle"}), \
                 patch.object(server, "frame_image", return_value=self.image), \
                 patch.object(server, "chat", return_value=self.decision()) as chat, \
                 patch.object(server, "save") as save:
                response = server.app.test_client().post(
                    "/api/reference_candidates/audit", json={
                        "frame": 0, "topic": "camera",
                        "expected_candidate_artifact_sha256": artifact,
                        "expected_frame_sha256": input_sha})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertFalse(response.json["candidate_class_audit_cache_hit"])
            chat.assert_called_once()
            save.assert_called_once()

    def test_audit_endpoint_failures_and_cancellation_never_publish_partial_audits(self):
        candidates = [copy.deepcopy(self.candidate)]
        for outcome in (RuntimeError("盲审服务失败"), server.RunCancelled("已取消"), "save_failure"):
            state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                     "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                     "reference_frames": {}, "candidates": {"0": candidates},
                     "agent_runs": {"0": {"review_queue": [{"reason": "既有证据"}]}},
                     "candidate_review_status": {"0": {"old": {"route": "human_review_required"}}}}
            before = copy.deepcopy({key: state[key] for key in ("agent_runs", "candidate_review_status", "candidates")})
            options = {"return_value": self.decision()} if outcome == "save_failure" else {"side_effect": outcome}
            with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
                 patch.object(server, "frame_image", return_value=self.image), \
                 patch.object(server, "chat", **options), \
                 patch.object(server, "save", side_effect=RuntimeError("保存失败") if outcome == "save_failure" else None) as save:
                response = server.app.test_client().post("/api/reference_candidates/audit", json={
                    "frame": 0, "topic": "camera", "candidate_id": 7,
                    "expected_candidate_artifact_sha256": server._sha256_json(candidates)})
                self.assertEqual(response.status_code, 200 if isinstance(outcome, server.RunCancelled) else 503, response.json)
                if isinstance(outcome, server.RunCancelled):
                    self.assertTrue(response.json["stopped"])
                if outcome != "save_failure":
                    save.assert_not_called()
                for key, value in before.items():
                    self.assertEqual(state[key], value)

    def test_whole_frame_audit_replaces_stale_evidence_and_preserves_other_sources(self):
        second = copy.deepcopy(self.candidate) | {"id": 8, "label": "person"}
        low_risk = copy.deepcopy(self.candidate) | {"id": 9, "label": "floor"}
        candidates = [copy.deepcopy(self.candidate), second, low_risk]
        preserved_queue = {"source": "visible_label_discovery", "reason": "其他来源证据"}
        old_artifact = "0" * 64
        stale_audits = [{"candidate_id": identifier, "candidate_artifact_sha256": old_artifact}
                        for identifier in (7, 9, 99)]
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels + ["floor"], "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {"0": candidates},
                 "agent_runs": {"0": {"candidate_class_audits": stale_audits,
                     "review_queue": [preserved_queue] + [
                         {"source": "candidate_class_audit", "candidate_id": identifier}
                         for identifier in (7, 9, 99, 100)]}},
                 "candidate_review_status": {"0": {
                     audit.candidate_fingerprint(low_risk): {"candidate_artifact_sha256": old_artifact},
                     "removed_fingerprint": {"candidate_artifact_sha256": old_artifact}},
                     "other_frame": {"keep": True}}}
        original_candidates = copy.deepcopy(candidates)
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), patch.object(server, "frame_image", return_value=self.image), patch.object(server, "chat", side_effect=[self.decision(), self.decision(None, .2, "uncertain")]) as chat, patch.object(server, "save") as save:
            response = server.app.test_client().post("/api/reference_candidates/audit", json={
                "frame": 0, "topic": "camera",
                "expected_candidate_artifact_sha256": server._sha256_json(candidates)})
            self.assertEqual(response.status_code, 200, response.json)
            records = response.json["agent"]["candidate_class_audits"]
            self.assertEqual([item["candidate_id"] for item in records], [7, 8])
            self.assertTrue(all(item["candidate_artifact_sha256"] == server._sha256_json(candidates)
                                for item in records))
            queue = response.json["agent"]["review_queue"]
            self.assertEqual(queue[0], preserved_queue)
            self.assertEqual(len(queue), 2)
            self.assertEqual(queue[1]["candidate_id"], 8)
            self.assertEqual(queue[1]["source"], "candidate_class_audit")
            self.assertEqual(set(state["candidate_review_status"]["0"]),
                             {audit.candidate_fingerprint(item) for item in candidates[:2]})
            self.assertEqual(state["candidate_review_status"]["other_frame"], {"keep": True})
            self.assertEqual(state["candidates"]["0"], original_candidates)
            self.assertEqual(state["truth"], {})
            self.assertEqual(chat.call_count, 2)
            save.assert_called_once()

    def test_second_candidate_failure_keeps_entire_frame_evidence_unchanged(self):
        candidates = [copy.deepcopy(self.candidate),
                      copy.deepcopy(self.candidate) | {"id": 8, "label": "person"}]
        state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                 "topic": "camera", "labels": self.labels, "truth": {}, "config": {},
                 "reference_frames": {}, "candidates": {"0": candidates},
                 "agent_runs": {"0": {"candidate_class_audits": [{"candidate_id": 99}],
                                       "review_queue": [{"source": "candidate_class_audit", "candidate_id": 99}]}},
                 "candidate_review_status": {"0": {"old": {"route": "human_review_required"}}}}
        before = copy.deepcopy({key: value for key, value in state.items() if key != "video"})
        with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), patch.object(server, "frame_image", return_value=self.image), patch.object(server, "chat", side_effect=[self.decision(), RuntimeError("第二候选失败")]) as chat, patch.object(server, "save") as save:
            response = server.app.test_client().post("/api/reference_candidates/audit", json={
                "frame": 0, "topic": "camera",
                "expected_candidate_artifact_sha256": server._sha256_json(candidates)})
            self.assertEqual(response.status_code, 503, response.json)
            self.assertEqual(chat.call_count, 2)
            save.assert_not_called()
            self.assertEqual({key: value for key, value in state.items() if key != "video"}, before)

    def test_quality_gate_rejects_empty_duplicate_and_contradictory_evidence(self):
        for change in [{"visible_cues": []}, {"visible_cues": ["Human body", " human  BODY "]},
                       {"classification": "uncertain", "observed_label": None,
                        "reason_code": "insufficient_visual_evidence", "candidate_supported": True}]:
            with self.subTest(change=change), self.assertRaises(audit.CandidateClassProtocolError):
                audit.validate_blind_candidate_class_decision(self.decision() | change, self.labels)

    def test_config_change_during_audit_never_publishes_results(self):
        candidates = [copy.deepcopy(self.candidate)]
        for endpoint in ("/api/reference_candidates/audit", "/api/reference_draft"):
            state = {"video": SimpleNamespace(path=Path("test.png"), topics={"camera": {"count": 1}}),
                     "topic": "camera", "labels": self.labels, "truth": {}, "config": {"llm_model": "old"},
                     "reference_frames": {}, "candidates": {"0": copy.deepcopy(candidates)},
                     "agent_runs": {"0": {"review_queue": []}}}
            original_runs = copy.deepcopy(state["agent_runs"])
            result = {"shapes": copy.deepcopy(candidates), "agent": {"stop_reason": "done"},
                      "review_queue": [], "applied": [], "candidates_added": 1}
            def changing_chat(*args, **kwargs):
                state["config"]["llm_model"] = "new"
                return self.decision()
            with patch.object(server, "state", state), patch.object(server, "run_status", {"status": "idle"}), \
                 patch.object(server, "frame_image", return_value=self.image), \
                 patch.object(server, "chat", side_effect=changing_chat), patch.object(server, "save") as save, \
                 patch.object(server, "describe", return_value={}), patch.object(server, "run", return_value=result), \
                 patch.object(server, "archive_candidate_revision", return_value=None):
                response = server.app.test_client().post(endpoint, json={
                    "frame": 0, "topic": "camera", "labels": ["ego"],
                    "expected_candidate_artifact_sha256": server._sha256_json(candidates)})
                self.assertEqual(response.status_code, 409, response.json)
                self.assertEqual(state["agent_runs"], original_runs)
                save.assert_not_called()

    def test_audit_version_tracks_model_config_protocol_and_registry(self):
        state = {"config": {"llm_model": "model-a", "llm_key": "private",
                            "llm_fast_key": "fast-private"}, "labels": self.labels}
        with patch.object(server, "state", state):
            first = server.candidate_class_audit_version()
            state["config"]["llm_key"] = "rotated"
            self.assertEqual(first, server.candidate_class_audit_version())
            state["config"]["llm_fast_key"] = "fast-rotated"
            self.assertEqual(first, server.candidate_class_audit_version())
            for key in ("llm_model", "llm_prompt_profile", "llm_url",
                        "llm_fast_url", "llm_fast_model",
                        "llm_fast_max_pixels", "llm_fast_max_long_edge",
                        "llm_fast_max_extra_pixels", "llm_fast_max_extra_long_edge",
                        "llm_fast_cooldown_seconds"):
                original = copy.deepcopy(state["config"])
                state["config"][key] = "changed"
                self.assertNotEqual(first, server.candidate_class_audit_version())
                state["config"] = original
            state["labels"] = ["person"]
            self.assertNotEqual(first, server.candidate_class_audit_version())
            state["labels"] = self.labels
            with patch.object(server, "_CANDIDATE_AUDIT_LOADED_CODE_SHA256", "changed protocol"):
                self.assertNotEqual(first, server.candidate_class_audit_version())

    def test_sam3_config_save_and_point_request(self):
        state = {"video": object(), "topic": "camera", "config": {}, "reference_frames": {}}
        with patch.object(server, "state", state), patch.object(server, "save"), \
             patch.object(server, "frame_image", return_value=self.image), \
             patch.object(server.urllib.request, "urlopen") as urlopen:
            response_body = io.BytesIO(b'{"shapes": []}')
            urlopen.return_value.__enter__.return_value = response_body
            client = server.app.test_client()
            response = client.post("/api/config", json={"sam3_url": "http://127.0.0.1:9999/"})
            self.assertEqual(response.status_code, 200)
            response = client.post("/api/segment_points", json={"frame": 0, "points": []})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(urlopen.call_args.args[0].full_url, "http://127.0.0.1:9999/segment_points")


if __name__ == "__main__":
    unittest.main()
