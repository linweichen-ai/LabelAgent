# SAM3 Stage-0 execution and rollback contract

## Current claim boundary

This package is executable configuration plus a deterministic dry-run compiler. The checked-in package is deliberately fail-closed: it cannot create a training process. It does not prove that training ran, that a checkpoint exists, or that SAM3 quality improved.

The only verified Stage-0 trainable scope is the six tensors under `segmentation_head.mask_predictor.mask_embed.*`, totaling 197,376 parameters. Image and language backbones are frozen. Decoder, adapter, LoRA, higher-stage unfreezing, and full-model training are not claimed or authorized.

## Two-stage launch

1. `compile` validates source hashes, exact parameter scope, truth/version bindings, split isolation, fixed-holdout baseline, resource admission, paths, rollback target, and explicit authorization. It emits a command preview and a rejection/authorization decision but cannot create a process.
2. `launch` is a separate explicit action. It rejects any compiled plan with blockers before reaching `subprocess.Popen`. The checked-in package has `process_creation_currently_allowed=false`, so it is compile/dry-run only.

No idle polling, GPU reservation, automatic retry, or automatic launch is permitted.

## Data and reproducibility

- Training input: immutable CVAT `human-completed` pixel masks from train72 only.
- Validation input: a sequence-disjoint human-truth subset carved from train72.
- test24: evaluation only, never training or early stopping.
- Fixed 640 holdout: final comparison only, never training or early stopping.
- Machine preannotations and synthetic fixtures: prompts/tests only, never truth.
- Seed: 3407 for Python, NumPy, Torch, CUDA, and data order; deterministic cuDNN, TF32 disabled.
- Single-GPU profile: batch/GPU 1, gradient accumulation 2.
- Dual-GPU profile: synchronous NCCL DDP, batch/GPU 1, gradient accumulation 1.
- Both profiles have effective global batch 2. A future smoke must still verify the declared loss/parameter tolerance before formal training.

The optimizer policy is AdamW, learning rate `1e-4`, weight decay `0.01`, BF16, gradient clipping `0.1`, at most 20 epochs. The declared early-stopping and best-checkpoint policy uses only the train72 sequence-disjoint validation split (`mask_iou`, patience 5, min delta 0.001). The fixed holdout is excluded from tuning. The upstream trainer does not yet provide a verified native early-stopping hook for this package, so the first authorized machine-side launch must verify the runtime controller before formal training; the checked-in package remains blocked and makes no claim that early stopping has executed.

## Checkpoint, resume, and rollback

The upstream trainer writes rank-0 rolling and numbered epoch checkpoints under a new run-specific output root. It must never reuse another run's root. A future verified selector must choose the best train72-validation epoch, copy it to a new content-addressed candidate artifact, and record its SHA-256; until then no checkpoint is a deployment candidate. Resume requires an explicit checkpoint SHA and exact matches for recipe, parameter list, truth version, and truth manifest; otherwise it fails closed.

The incumbent is `/home/clw/models/labelagent/sam3.pt`, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. A failed or incomplete run cannot switch the deployment alias. Rollback means retaining or restoring this immutable incumbent; existing outputs are never overwritten.

## Deployment gate

The fixed holdout report must include mask IoU, boundary F1, object precision/recall, and false positives for overall, floor, slope, real, and simulation views. A deployment candidate is allowed only when both overall and slope do not regress on all four quality metrics and false positives do not increase. Real and simulation accounting remains separate. Any missing view, metric, hash, or failed guard retains the incumbent.
