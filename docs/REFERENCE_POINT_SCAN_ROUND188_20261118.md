# Reference-frame point scan, round 188

Date: 2026-11-18

## Outcome

The candidate-only reference-frame recovery path now has three independent stages:

1. one encoded image is scanned with 184 fixed, color-independent point prompts;
2. the top candidates must be classified as `other_robot` both without and with a
   sheet of 12 project-specific positive exemplars;
3. an isolated-mask check rejects candidates whose pixels are only wall, floor,
   ceiling, glare, or empty background.

No candidate is written to the workbench state, treated as human truth, or approved
for training.

## Read-only results

| Frame | Retained candidate IDs | Existing annotation recovered | Point scan service time |
| --- | --- | ---: | ---: |
| 0 | C05, C07 | 1 / 1 | 5.63 s |
| 55 | C01, C06 | 0 / 0 | 5.85 s |

For frame 0, C07 overlaps the existing assistant-reviewed candidate by
`IoU = 0.965947`. That existing annotation has `human_truth=false`,
`candidate_only=true`, and `approved_for_training=false`; the overlap is therefore
only a repeatability check, not a human-ground-truth score.

The false wall candidate seen in the preceding experiment did not pass round 188.
Visual inspection indicates frame-55 C01 is a mobile robot. C06 remains a review
candidate and must not enter training without confirmation.

## Runtime boundary

- Formal SAM service: healthy at `127.0.0.1:38766`; it was not modified by this trial.
- Candidate/fallback SAM service: healthy at `127.0.0.1:28766`; it hosts the new
  one-encode point-batch endpoint.
- Local VLM and workbench page both returned HTTP 200 after the experiments.
- The candidate service was deployed with a remote backup. The workbench continues to
  use the formal SAM service.

## Reproduction

Run from the repository root with the LabelAgent virtual environment:

```powershell
D:\Apps\LabelAgent\.venv\Scripts\python.exe -m tools.evaluate_reference_point_scan `
  --state D:\Apps\LabelAgent\work\state.json `
  --output artifacts\reference-point-scan `
  --frames 0,55 `
  --point-sam-url http://127.0.0.1:28766 `
  --top-k 12 `
  --exemplar-sheet artifacts\round181-cvat-other-robot-exemplars\other-robot-positive-sheet.jpg
```

## Next gate

Before integration into the visible workbench, evaluate the three-stage rule on a
frame-disjoint reviewed split from the 150-frame CVAT snapshot. Report recall,
candidate count per frame, and background false-candidate count. Keep all results
candidate-only until a human review action explicitly promotes them.
