# Dual RTX 5090 acceptance, 2026-11-11

## Verified deployment

- GPU0 runs the SAM3 service on loopback port 18766. The checkpoint SHA-256 is `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
- GPU1 runs Qwen3-VL-8B-Instruct with Transformers 4.57.6 and NF4 on loopback port 18878.
- Both isolated environments use Python 3.12.16 and PyTorch 2.10.0+cu128. `uv pip check` reported all packages compatible, and both GPUs passed a real BF16 matrix multiplication before the services were started.
- The copied VLM contains 44 files. Source and target per-file SHA-256 manifests both hash to `484cab50857adbf7f16e71fd2462af2a507c85ff92ef47f998174de7c776bfb1`.
- No password, model weight, dataset, or real deployment config is tracked by Git.

## Real request evidence

The request image is camera 0, frame 0 from the clean SFL-CPD15-Y batch A/000 MCAP. It is 1920x1536, 263785 bytes, and has SHA-256 `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`.

SAM3 used one positive point at `(1100, 620)` on the yellow ramp. Cold model load plus inference took 7.767 seconds. The warm repeat took 0.083 seconds. Both returned HTTP 200, score 0.98046875, one connected part, 44 outer vertices, no holes, and no warnings. The retained service used approximately 5580 MiB of GPU memory after the request.

The VLM accepted the same image through the OpenAI-compatible endpoint and returned valid JSON. It used 594 prompt tokens and 24 completion tokens. The recorded preprocessing and generation times were 0.058 and 1.482 seconds. Peak allocated memory was 7508647424 bytes, and the retained process used approximately 8512 MiB after the request.

An isolated local workbench was configured through separate SSH tunnels and loaded the same clean MCAP without copying it. A workbench-to-SAM request returned the same score and 44-vertex contour in 0.670 seconds.

## Failures found and repaired

- Windows CRLF line endings initially broke the Linux launcher. `.gitattributes` now enforces LF for shell and environment-template files, with a regression test.
- The health wait function referenced a local variable too early under `set -u`; the declaration is now split and tested.
- SAM3 imports `einops`, `pycocotools`, and `psutil` at runtime but did not declare them as core dependencies. They are now explicit project dependencies.

## Acceptance boundary

This verifies installation, model integrity, both CUDA runtimes, both model services, one real SAM request, one real VLM request, and the workbench-to-SAM path. It does not verify segmentation accuracy across the dataset, SAM fine-tuning, or the full LabelAgent reference-description-review loop. The isolated workbench currently has zero human truth frames. A human-confirmed reference mask must be saved before the full workflow can be run without fabricating ground truth.
