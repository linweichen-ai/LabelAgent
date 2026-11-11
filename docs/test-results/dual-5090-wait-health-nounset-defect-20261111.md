# Dual 5090 wait_health nounset defect, 2026-11-11

## Root cause and repair

This failure was a Bash initialization-order defect, not a SAM3 model, checkpoint, CUDA, or dependency failure. With `set -u`, this declaration is unsafe because Bash expands the right-hand side for `url` before `port` has become a local variable:

```bash
local port="$1" limit="$2" pid="$3" url="http://127.0.0.1:$port/health"
```

The committed implementation assigns the arguments first and builds the URL on the next statement:

```bash
local port="$1" limit="$2" pid="$3"
local url="http://127.0.0.1:$port/health"
```

## Regression gate

`tests/test_dual_5090_deployment.py` now extracts the actual `wait_health` function from the repository script and invokes it under `set -u` with mocked `curl`, `kill`, and `sleep`. This is a runtime test of the real function body; the existing text-order check, raw-byte CR gate, and `bash -n` remain additional checks rather than substitutes.

- Deployment test module: 5/5 passed.
- Related deployment/VLM test selection: 22/22 passed.
- `bash -n tools/dual_5090_services.sh`: passed.
- Repair commit: `a6f3d313605af3b83666844bc379aa49599d14cf`.

## Same-commit deployment receipt

The deployed application is not a Git worktree, so the service script was packaged directly with `git archive` from commit `a6f3d31`. The Windows working tree was not copied.

- Git archive SHA-256: `35b9ef10f48712cf554e83d0edd66566bb24eb04d22338f77aca1526508a5904`
- Target: `/home/clw/projects/labelagent-5090/app/tools/dual_5090_services.sh`
- Previous target SHA-256: `89fd181d801619ff9d59e6de1a20b9c3ca7a7c9d9cd02c92f7f640547391dd12`
- Deployed target SHA-256: `47ab98176d8e6953aa161e1f1a5a366836bbd4eddcdd9e457d634543952421fc`
- CR byte count: 0
- Server receipt directory: `/home/clw/run/labelagent-5090/deploy-a6f3d31-CvpSBF91`

The previous script was retained in the receipt directory before the exact target was atomically replaced. SAM was then stopped and started only through the deployed `dual_5090_services.sh`. The successful `start sam` path executed `wait_health` without an unbound-variable failure.

## Post-restart acceptance

- SAM PID: `1131700`
- Configured and observed GPU UUID: `GPU-8fc83bb8-cdd1-8da3-98b7-990f5de229ff`
- GPU memory after real request: 5,580 MiB
- Loopback port: `18766`, listening
- Health: `{"model_loaded":true,"status":"ok"}`
- Log: `/home/clw/logs/labelagent-5090/sam3-gpu0.log`
- Log SHA-256: `4868bbaaae18589d755fda25c4d1cb968873ae50264c7f38c3b3384c38ba93f9`

The post-restart real request used camera 0, frame 0 from the SFL-CPD15-Y factory MCAP. Its SHA-256 was rechecked as `3775ee5f41374c9a302c458cfffa2799222e22ff17541045d6acfb8434bd682f`. The request returned HTTP 200 in 8.438 seconds with score 0.98046875, one part, 44 outer vertices, no holes, and no warnings. The temporary image was deleted after the JSON receipt was saved outside Git.

VLM remained healthy throughout; no manual persistent service or GPU hold process was created.
