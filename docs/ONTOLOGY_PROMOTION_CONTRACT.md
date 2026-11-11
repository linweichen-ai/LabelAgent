# VLM label discovery and ontology promotion contract

This milestone is local and CPU-only. It does not call a VLM, GPU, CVAT, OSS,
or any remote service. Static examples are synthetic contract fixtures, not
model findings or production ontology changes.

## Machine observation

Every observation binds multi-view image artifacts, candidate, model, prompt,
label-registry hashes and real/simulation provenance. The structured outcome is
one of `known_label`, `unknown_candidate`, or `uncertain_conflict`, with visual
evidence, confidence, aliases, nearest registered labels and multi-view
agreement. Unknown/new labels, low confidence and SAM/ontology conflict always
route to `human_review_required`. Model events cannot mutate ontology or truth.

## Deduplication and representative samples

Unknown observations can be grouped by a versioned clustering configuration.
The deterministic representative policy is maximum cited-view evidence, then
confidence, then event ID. A cluster remains `proposal_only`; source event IDs
are separately partitioned into real and simulation cohorts. Clustering cannot
name a production class or write a registry version.

## Human promotion gate

A new append-only registry version requires an authenticated human name,
definition, unique class ID, reviewer, timestamp and approval receipt, plus a
fixed regression manifest. Real and simulation metrics and truth versions are
recorded separately. The parent registry is retained unchanged. The resulting
historical replay plan is `planned_not_executed`, produces new outputs only and
cannot promote truth automatically.

`floor` and `slope` stay distinct training/annotation labels. The downstream
consumer mapping remains `traversable_area = floor UNION slope`.
