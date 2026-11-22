from pathlib import Path


SCRIPT = Path("tools/run_sam3_assistant_crop_holdout.sh").read_text(encoding="utf-8")


def test_holdout_launcher_has_non_deployment_guards():
    assert '[[ "$SAM3_ASSISTANT_HOLDOUT_ROOT" != *test24* ]]' in SCRIPT
    assert '[[ "$SAM3_ASSISTANT_HOLDOUT_OUTPUT" != *test24* ]]' in SCRIPT
    assert '[[ ! -e "$SAM3_ASSISTANT_HOLDOUT_OUTPUT" ]]' in SCRIPT
    assert 'SAM3_ASSISTANT_HOLDOUT_SHA256' in SCRIPT
    assert 'SAM3_EVAL_CANDIDATE_SHA256' in SCRIPT


def test_holdout_launcher_always_runs_baseline_and_candidate():
    assert "run_eval baseline BASELINE BASELINE" in SCRIPT
    assert "candidate_checkpoint=$SAM3_EVAL_CANDIDATE_CHECKPOINT" in SCRIPT
    assert 'run_eval candidate "$candidate_checkpoint" "$candidate_checkpoint_sha"' in SCRIPT
    assert "compare_sam3_segmentation_predictions.py" in SCRIPT
    assert "--split assistant_holdout15" in SCRIPT
