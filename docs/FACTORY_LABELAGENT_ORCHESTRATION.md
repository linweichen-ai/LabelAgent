# Factory LabelAgent orchestration contract

## Claim boundary

This milestone is a CPU-only orchestration and replay contract. It validates immutable events and determines whether a stream may advance; it does not call SAM, VLM, Agent tools, CVAT, GPU, SSH, OSS, training, deployment, or rollback services. The four checked-in traces are synthetic contract fixtures, not production results, human truth, training runs, or deployment evidence.

## Business-state path

The main path is:

`ingested -> qc_passed -> candidate_planned -> vlm_reviewed -> agent_corrected -> vlm_rechecked -> human_review_required -> candidate_accepted -> truth_promoted -> training_eligible -> trained -> evaluated -> deployment_candidate -> deployed -> rolled_back`

Fail-closed branches are:

- `ingested -> quarantined` for explicit QC failures;
- `candidate_accepted -> evaluation_truth` for the test split, which can never become training eligible;
- `evaluated -> baseline_retained` when the overall or slope deployment guard fails.

An approved human gate is mandatory before `candidate_accepted`, `truth_promoted`, or `evaluation_truth`. Unknown candidates additionally require explicit ontology human approval. A truth version remains bound to its split and manifest SHA through training and evaluation. Evaluation and deployment decisions remain bound to the exact training run and checkpoint artifacts.

## Event and ledger invariants

Every event contains a stable batch key, stream ID, idempotency key, monotonically increasing state version, previous-event pointer, input/output/config/model/prompt hashes, retry state, queue route, human gate, real/simulation provenance, contract references, and an immutable event hash. Exact replay returns the prior record without appending; the same key with different immutable content is rejected.

Ledger records form a global SHA-256 chain. Recovery replays records from the last immutable snapshot and recomputes stream heads, queue counts, real/simulation accounting, and trace summaries; metadata that does not match replayed records is rejected. Snapshots use exclusive creation and never overwrite an existing path.

Operational failures retain the business state but increment the state version. Retryable failures enter `retry_wait` until the configured budget is exhausted, after which they enter `dead_letter`. Backpressure is applied only at or above the high watermark and is released only after capacity returns. QC failures enter the quarantine queue with a reason and cannot be silently skipped.

## Labels and deployment guard

Training and evaluation keep `floor` and `slope` as separate labels. The downstream navigation mapping is `traversable_area = union(floor, slope)`, with slope passability based on the current user-defined project semantics. Deployment evaluation reports mask IoU, boundary F1, object precision/recall, and false positives for overall, floor, slope, real, and simulation views. A candidate may advance only if both overall and slope guards pass; otherwise the incumbent baseline is retained.

## Static acceptance traces

The checked-in ledger contains four streams:

1. known-label candidate stopping at `human_review_required` because human pixel truth is absent;
2. unknown candidate stopping at human and ontology review;
3. QC failure entering quarantine with no silent skip;
4. hypothetical slope regression reaching `evaluated` in static replay and ending at `baseline_retained`.

All side-effecting states in the fourth trace are explicitly marked `simulated`, `synthetic_contract_fixture_not_production`, with zero external calls and zero production mutations.
