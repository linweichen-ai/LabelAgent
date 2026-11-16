# PAI Blackwell environment acceptance

Observed 2026-11-04 through the authorized LabelAgent web terminal. Dedicated torch2.8.0+cu128 / transformers4.57.6 / PEFT0.17.1 / accelerate1.10.1 environment installed. Both RTX PRO 5000 72GB Blackwell GPUs (capability12.0) passed BF16 matrix multiplication and native SDPA finite-output/nonzero-gradient checks. This is GPU primitive acceptance, not Qwen inference or training acceptance.

A single pinned Qwen3-VL-2B download was started; completion has not yet been verified. The bounded probe selects three existing tuning candidates and two blank controls, uses BF16+SDPA on GPU0 only, preserves input hashes, and writes responses/summary plus an offline HTML view. Five CPU boundary tests passed, including refusing to use reserved GPU1 if GPU0 is busy. Real input decoding and source-image digests were checked locally.

The natural candidates have no independent reviewed truth. This probe measures protocol/functional behavior only, never segmentation accuracy or full-Agent convergence. It does not train weights or write annotations. Cloud probe deployment and execution remain pending. Existing unrelated jobs and environments were preserved.
