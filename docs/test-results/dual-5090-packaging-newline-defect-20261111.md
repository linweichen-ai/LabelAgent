# Dual 5090 packaging newline defect, 2026-11-11

## Classification

The reported `pipefail\r` symptom is classified as a **packaging newline defect**, not a SAM3 model, checkpoint, CUDA, or dependency failure. The bounded raw-byte scan was limited to the deployed copy at `/home/clw/projects/labelagent-5090/app`.

The deployed tree contained six shell scripts. Five contained CRLF; `tools/dual_5090_services.sh` itself did not. Before repair, the original service manager's `status sam` path exited 0 and reported the existing SAM process healthy.

## Bounded repair receipt

- Receipt directory: `/home/clw/run/labelagent-5090/packaging-newline-20261111-jlFtZav4`
- Before-manifest SHA-256: `107805aa474fe8da76a894ad46059faa6c7bc365049669f7be4f48ec490e5174`
- After-manifest SHA-256: `8e5f4a3ab11c09570dcfee6101aef5609ac38b7c0c85f9c8d3b7a20295e0d414`
- Original copies of all five changed files are retained under the receipt directory.
- Each repaired file was independently verified to equal its original bytes with only every `CRLF` pair replaced by `LF`.
- Post-repair CRLF hit count: `0`.

| CRLF pairs | Script | Before SHA-256 | After SHA-256 |
|---:|---|---|---|
| 21 | `tools/pai_dsw/install_cloud.sh` | `183a6f9c447ace183bd01cc49d747f26d46d6678b94976d30a861617d5bae56f` | `abe6622a2aed4c6afc793db99f0fbcc4af6c8df087243eccdd55cb9fcb3c82c8` |
| 36 | `tools/pai_dsw/start_vlm32b.sh` | `85aff76f6af02f8200440f7534ae2cace036767f95767007a423fd1105218fdf` | `68fd49bb8f62da4d1c35669eeb46315f4ab84772c48d38bdc420804cdf3e0113` |
| 47 | `tools/pai_exploration_20261104/setup_blackwell_sdpa.sh` | `3763601313a24422546f3f77640540bfab32b531fd8653d5fa98c06bd4c0acd2` | `f3b0009bed8a510fed160373021505a8a842cca4749e3b7982616f93fb23e800` |
| 21 | `tools/pai_exploration_20261104/setup_qwen.sh` | `e86cb6a72d88ddec09d58823d0df55344d8c4b4497e292c1c811dc86240e1b21` | `9f8c1c96533d25d1c7e244f7ba4cd8100cf23044d1022d8cad954d6f05f790a6` |
| 41 | `tools/pai_exploration_20261104/train_qwen.sh` | `7fda90e244c046d2900d2c33a312fe2ac868c160cc5fc1bafeb71ba94517d82b` | `13be3802a049a30251767e78d9546a7c6108b24ca0c897c353553542e40445db` |

All six deployed shell scripts passed `bash -n`. The five repaired scripts also passed actual execution of their header through the `set` statement. The original service manager's `status all` path then exited 0 and reported both SAM and VLM healthy.

## SAM restart and real-request acceptance

SAM was stopped and started only through `tools/dual_5090_services.sh`; no manual persistent service was created.

- PID: `1127250`
- Configured and observed GPU UUID: `GPU-8fc83bb8-cdd1-8da3-98b7-990f5de229ff`
- Observed GPU memory: 5,580 MiB after the request
- Loopback port: `18766`, listening
- Log: `/home/clw/logs/labelagent-5090/sam3-gpu0.log`
- Log SHA-256 after acceptance: `50fa98c275e193d79fd4c04dddb3aa43391ea12a4cc797937ee1e5989a3e1935`
- Health after request: `{"model_loaded":true,"status":"ok"}`

The real request used camera 0, frame 0 from the SFL-CPD15-Y factory MCAP. The temporary image was hash-checked and removed after the request; only JSON evidence remains outside Git.

- Image bytes: 263,785
- Image SHA-256: `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`
- Prompt: positive point `(1100, 620)`, selected by Codex/script and recorded as `assistant_selected_prompt` / `operator_seed` (not human-selected or human truth)
- HTTP status: 200
- Cold-load plus inference: 8.812 seconds
- Score: 0.98046875
- Parts / outer vertices / holes: 1 / 44 / 0
- Warnings: none

## Repository prevention gate

`tests/test_dual_5090_deployment.py` now scans every `tools/**/*.sh` file as raw bytes and rejects any carriage-return byte. The test failed against the stale Windows CRLF worktree and passed after normalization, demonstrating that it detects the packaging defect. `.gitattributes` continues to require `*.sh text eol=lf`, and the deployment guide now makes the test a mandatory pre-copy gate.
