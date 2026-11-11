"""Fail closed if obsolete SAM3 trainable-scope claims return."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_TARGETS = (
    "docs/SAM3_PEFT_TRAINING_GATE.md",
    "docs/CLOSED_LOOP_ARCHITECTURE.md",
    "configs/sam3_sfl_cpd15y_peft_plan.json",
    "configs/sam3_sfl_cpd15y_staged_unfreeze.json",
    "schemas/sam3-peft-plan.schema.json",
    "schemas/sam3-staged-unfreeze.schema.json",
    "tools/sam3_peft_launch_gate.py",
    "tools/sam3_trainable_inventory.py",
)
FORBIDDEN_PATTERNS = (
    r"full_segmentation_head",
    r"segmentation_head_plus_geometry_prompt_encoder",
    r"plus_upper_transformer_decoder_layers",
    r"future_candidate_prefixes",
    r"phase1_trainable_families",
    r"stage-[123]：",
    r"只开放经代码核验的 mask decoder",
    r"开放完整 `?segmentation_head",
)


def find_forbidden(text):
    return [pattern for pattern in FORBIDDEN_PATTERNS if re.search(pattern, text)]


def check_policy(root=ROOT):
    violations = []
    for relative in TEXT_TARGETS:
        text = (root / relative).read_text(encoding="utf-8")
        for pattern in find_forbidden(text):
            violations.append({"path": relative, "pattern": pattern})

    plan = json.loads(
        (root / "configs/sam3_sfl_cpd15y_peft_plan.json").read_text(encoding="utf-8")
    )
    staged = json.loads(
        (root / "configs/sam3_sfl_cpd15y_staged_unfreeze.json")
        .read_text(encoding="utf-8")
    )
    expected = ["segmentation_head.mask_predictor.mask_embed."]
    if plan["model"]["verified_trainable_scope"]["prefixes"] != expected:
        violations.append({"path": "peft_plan", "pattern": "verified scope drift"})
    if plan["model"]["conditional_expansion"]["enabled"]:
        violations.append({"path": "peft_plan", "pattern": "conditional expansion enabled"})
    if plan["model"]["conditional_expansion"]["predeclared_trainable_prefixes"]:
        violations.append({"path": "peft_plan", "pattern": "predeclared prefixes present"})
    if plan["model"]["full_model_training"]:
        violations.append({"path": "peft_plan", "pattern": "full model enabled"})
    if staged["current_verified_stage"]["trainable_prefixes"] != expected:
        violations.append({"path": "staged_plan", "pattern": "Stage-0 scope drift"})
    if staged["conditional_expansion"]["predeclared_stages"]:
        violations.append({"path": "staged_plan", "pattern": "predeclared stages present"})
    if staged["full_model_policy"]["enabled"]:
        violations.append({"path": "staged_plan", "pattern": "full model enabled"})
    return violations


def main():
    violations = check_policy()
    json.dump(
        {"status": "passed" if not violations else "failed",
         "violations": violations},
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
