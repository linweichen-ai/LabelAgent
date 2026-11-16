# PAI DSW 部署准备工具

完整参数及操作步骤见 [部署指南](../../docs/pai-dsw.md)。

- install_cloud.sh：仅在 PAI Linux 内新建项目环境，使用 uv 安装稳定版本并记录依赖；不创建实例、不下载权重。
- preflight_pai.py：只读检查分配的 GPU、共享内存、权重分片及可选摘要；不停止其他进程。
- start_vlm32b.sh：在分配的双 L20N 上启动 BF16 32B，仅绑定回环，保留 PAI 设置的设备可见范围。
- probe_vlm_api.py：通过 SSH 转发后的回环接口验证真实图片与严格 JSON；不修改工作台设置，不冒充 GPU 后端证明或质量验收。

本地 Python/Bash 语法及 CLI 检查通过。尚未在用户 PAI 实机安装、加载模型或执行云端推理。不要将本地检查当成部署成功。
