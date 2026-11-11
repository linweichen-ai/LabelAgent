# Offline CVAT review contract

This milestone is a static, CPU-only adapter. It does not connect to CVAT,
upload data, call model services, use a GPU, or mutate OSS.

## Export boundary

- `train72` is bound to `split=train`; `test24` is bound to `split=test`.
- Every candidate is exported as `machine_preannotation` with
  `human_completed=false` and `review_state=human_review_required`.
- Task, job and package versions, image/input/candidate/model/config hashes,
  preannotation hash, label-registry version, reason, priority and provenance
  are immutable bindings.
- `unknown` and `new_label_candidate` remain pending human review. Export never
  mutates the registered ontology.

## Return boundary

A return is accepted only when CVAT reports an authenticated human completion
and all task/job/version, split, source, candidate, annotation-export,
before/after-diff and approval hashes match. Registered labels are restricted
to the current ontology. `floor` and `slope` remain separate annotation labels;
only consumers map `traversable_area = floor UNION slope`.

- A valid `train72` return delegates to the existing immutable truth-promotion
  gate. The adapter does not implement a bypass.
- A valid `test24` return creates append-only evaluation truth with
  `training_return_allowed=false` and `truth_eligible_for_training=false`.
- Machine/automatic completion, missing reviewer evidence, version drift,
  split mixing, unknown labels, duplicate truth versions and hash tampering are
  explicit fail-closed rejection events.

The examples are synthetic contract fixtures, not completed CVAT work and not
production truth. The append-only review ledger uses exclusive-create records,
idempotency keys and a SHA-256 hash chain.
