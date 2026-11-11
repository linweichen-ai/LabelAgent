# Dual RTX 5090 acceptance, 2026-11-11

## Verified deployment

- GPU0 runs the SAM3 service on loopback port 18766. The checkpoint SHA-256 is `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- GPU1 runs Qwen3-VL-8B-Instruct with Transformers 4.57.6 and NF4 on loopback port 18878.
- Both isolated environments use Python 3.12.16 and PyTorch 2.10.0+cu128. `uv pip check` reported all packages compatible, and both GPUs passed a real BF16 matrix multiplication before the services were started.
- The copied VLM contains 44 files. Source and target per-file SHA-256 manifests both hash to `484cab50857adbf7f16e71fd2462af2a507c85ff92ef47f998174de7c776bfb1`.
- No password, model weight, dataset, or real deployment config is tracked by Git.

## Qwen direct-transfer and temporary-key cleanup receipt

The Qwen transfer is accepted as complete only after both the model-integrity check and the temporary SSH cleanup gate below passed.

### Model-integrity receipt

- Source: 4090 `/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-8B-Instruct`
- Target: 5090 `/home/clw/models/labelagent/Qwen3-VL-8B-Instruct`
- Source manifest: 44 files; manifest SHA-256 `484cab50857adbf7f16e71fd2462af2a507c85ff92ef47f998174de7c776bfb1`
- Target manifest: 44 files; manifest SHA-256 `484cab50857adbf7f16e71fd2462af2a507c85ff92ef47f998174de7c776bfb1`
- The per-file manifests were generated from sorted relative paths and each file's SHA-256, so equal manifest hashes mean the source and target file lists and file contents matched.
- A readback at `2026-11-11T11:52:40+08:00` found both model directories present with 44 files. `du -sb` reported 23,931,759,698 bytes for the source and 23,931,743,314 bytes for the target; the 16,384-byte directory-size difference is filesystem directory metadata and is not a per-file content mismatch.

### Temporary Ed25519 identity

- Comment: `labelagent-5090-transfer-20261111`
- Fingerprint: `SHA256:JNJ49G+Gc+bJwwKt1pZPp8QgY1ntMcs9DrX29KtlPJA`
- 5090 private-key path: `/home/clw/run/labelagent-5090/transfer_ed25519`
- 5090 public-key path: `/home/clw/run/labelagent-5090/transfer_ed25519.pub`
- 4090 authorization file: `/home/clw/.ssh/authorized_keys`

### Cleanup gate receipt

- Before removal, the exact comment matched one and only one line in the 4090 authorization file.
- Removal selected only a line ending in the exact comment `labelagent-5090-transfer-20261111`; it did not use a broad key-type or user match.
- Current 5090 readback: private key absent; public key absent.
- Current 4090 readback: fingerprint count `0`; exact-comment count `0`; authorization-file mode `600`; one unrelated key line remains.
- A batch login was then attempted from 5090 to 4090 with `IdentitiesOnly=yes`, agent/password/keyboard-interactive authentication disabled, and the deleted private-key path as the sole identity. SSH reported the identity file missing and exited `255` with `Permission denied`; failure was the expected result.
- Additional residue checks: 4090 `/tmp/transfer_5090_to_4090_20261111.pub` is absent, and local relay file `C:\Users\seer\Documents\ChatGPT\Seer2clw\task_workspaces\transfer_5090_to_4090_20261111.pub` is absent.

Cleanup gate: **PASS**. Qwen migration status: **COMPLETE**. This completion status was assigned only after the cleanup readback and expected-failure login test passed.

## Real request evidence

The request image is camera 0, frame 0 from the clean SFL-CPD15-Y batch A/000 MCAP. It is 1920x1536, 263785 bytes, and has SHA-256 `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`.

SAM3 used one positive point at `(1100, 620)` on the yellow ramp. This point was selected by Codex/script and is recorded as `assistant_selected_prompt` / `operator_seed`, not as human-selected or human truth. Cold model load plus inference took 7.767 seconds. The warm repeat took 0.083 seconds. Both returned HTTP 200, score 0.98046875, one connected part, 44 outer vertices, no holes, and no warnings. The retained service used approximately 5580 MiB of GPU memory after the request.

The VLM accepted the same image through the OpenAI-compatible endpoint and returned valid JSON. It used 594 prompt tokens and 24 completion tokens. The recorded preprocessing and generation times were 0.058 and 1.482 seconds. Peak allocated memory was 7508647424 bytes, and the retained process used approximately 8512 MiB after the request.

An isolated local workbench was configured through separate SSH tunnels and loaded the same clean MCAP without copying it. A workbench-to-SAM request returned the same score and 44-vertex contour in 0.670 seconds.

## Failures found and repaired

- Windows CRLF line endings initially broke the Linux launcher. `.gitattributes` now enforces LF for shell and environment-template files, with a regression test.
- The health wait function referenced a local variable too early under `set -u`; the declaration is now split and tested.
- SAM3 imports `einops`, `pycocotools`, and `psutil` at runtime but did not declare them as core dependencies. They are now explicit project dependencies.

## Acceptance boundary

This verifies installation, model integrity, both CUDA runtimes, both model services, one real SAM request, one real VLM request, and the workbench-to-SAM path. It does not verify segmentation accuracy across the dataset, SAM fine-tuning, or the full LabelAgent reference-description-review loop. The isolated workbench currently has zero human truth frames. A human-confirmed reference mask must be saved before the full workflow can be run without fabricating ground truth.
