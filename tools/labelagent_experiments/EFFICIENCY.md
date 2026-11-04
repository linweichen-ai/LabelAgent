# Instance and gap-repair experiment

Opt-in research helpers. The application default is unchanged. Candidate outputs are not truth and are never automatically written back.

`run_efficiency_partial_comparison.py --config <private-config.json>` uses a frozen source snapshot, existing local model services, the original inference package and capture. Its config names `source_snapshot`, `output`, `dataset`, `package`, `dependency_dir`, `python`, `reference_annotations`, `reference_description`, `sam_url`, `vlm_url`, `vlm_model`, `max_rounds`, `cases`, `reference_case`, `additional_pal_shapes`, `instance_generation`, `optimization_plan` and `sam_gap_negative_policy`. Paths in historical generation/diagnostic helpers follow the archived workstation layout; adapt them for another machine. Private inputs, masks, pictures and caches are delivered separately.

The controller enables reviewed gap batches only with `llm_batch_reviewed_gaps`. `sam_accept_complete_gap_subsets` admits an individual approved gap only when every one of its pixels is contained by an actual SAM mask. Unmet regions remain unchanged and explicit in the pending comparison. Only the service's positive-coverage warning can take this path; its text is retained. Other warnings still reject. A real next-round comparison is required, and unreviewed pending output is rolled back. No old floor pixel may disappear or unapproved pixel be added. Limits remain three rounds, three operations per round and 32 cumulative SAM points.

Unknown kept regions are pixel-locked, not asserted as negative evidence. The helper's positive-only mode changes prompts, not edit bounds. SAM batches are used only when more than three reviewed edits remain; a small remainder uses individual bounded edits.

Instance children are image-derived hypotheses clipped to the original PAL union while original masks remain. This controls union area and cannot recover spatial misses outside that union. Cropped SAM calls and raw-photo model checks are not human validation. V2 projected reference scoring, support/purity thresholds and strict-improvement acceptance are unchanged. `audit_efficiency.py` and `diagnose_subset_limit.py` are retrospective tools and must never feed the proposal selector.

Original failed runs and all failed acceptance conditions remain in the private handoff. Two exposed same-capture images do not establish generalization or dense semantic accuracy.
