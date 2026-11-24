#!/usr/bin/env python3
"""One-shot localhost form for configuring LabelAgent expert API credentials."""

from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import secrets
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser
import winreg


PROVIDERS = {
    "openai": ("OpenAI / ChatGPT API", "LABELAGENT_OPENAI_API_KEY",
               "openai_chat"),
    "qwen": ("阿里云 Qwen API", "LABELAGENT_QWEN_API_KEY",
             "openai_chat"),
    "custom": ("自定义 OpenAI 兼容接口", "LABELAGENT_EXPERT_API_KEY",
               "openai_chat"),
}


def set_user_env(name: str, value: str) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def page(token: str, message: str = "", success: bool = False) -> bytes:
    options = "".join(
        f'<option value="{html.escape(key)}">{html.escape(label)}</option>'
        for key, (label, _, _) in PROVIDERS.items()
    )
    banner = (f'<div class="banner {"ok" if success else "error"}">'
              f'{html.escape(message)}</div>') if message else ""
    disabled = " disabled" if success else ""
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LabelAgent API 配置</title>
<style>body{{margin:0;background:#eef5f2;font:16px 'Microsoft YaHei',sans-serif;color:#153c34}}
.card{{max-width:680px;margin:7vh auto;background:white;border-radius:18px;padding:30px 36px;box-shadow:0 18px 50px #163d3020}}
h1{{margin:0 0 10px;font-size:27px}}p{{line-height:1.7;color:#526862}}label{{display:block;margin:18px 0 7px;font-weight:700}}
input,select{{box-sizing:border-box;width:100%;padding:12px;border:1px solid #b9cec7;border-radius:9px;font-size:16px}}
button{{margin-top:24px;width:100%;padding:13px;border:0;border-radius:9px;background:#14856b;color:white;font-size:17px;font-weight:700}}
.note{{background:#f4faf7;border-left:4px solid #14856b;padding:12px 15px;border-radius:7px}}.banner{{padding:12px;margin:15px 0;border-radius:8px}}
.ok{{background:#e4f7ee;color:#08734f}}.error{{background:#fff0ee;color:#a43c30}}</style></head><body><main class="card">
<h1>LabelAgent 外部视觉模型 API</h1><p class="note">此页面只运行在您的电脑 <b>127.0.0.1</b>。Key 不进入对话、项目、日志、GitHub 或 CNB。保存后先用合成图测试，不会自动上传工厂原图。</p>
{banner}<form method="post" action="/{token}">
<input type="hidden" name="csrf" value="{token}"><label>供应商</label><select name="provider"{disabled}>{options}</select>
<label>API Key</label><input type="password" name="api_key" autocomplete="off" required{disabled}>
<label>接口地址（可选）</label><input name="base_url" placeholder="留空则后续使用供应商官方地址"{disabled}>
<label>模型名（可选）</label><input name="model" placeholder="留空则连通后读取可用模型再选择"{disabled}>
<button type="submit"{disabled}>{'保存成功，可以关闭本页' if success else '安全保存'}</button></form></main></body></html>"""
    return body.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    token = ""

    def send_page(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path != f"/{self.token}":
            self.send_error(404)
            return
        self.send_page(page(self.token))

    def do_POST(self):
        if urlparse(self.path).path != f"/{self.token}":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 32_768:
                raise ValueError("请求大小无效")
            values = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
            provider = values.get("provider", [""])[0]
            key = values.get("api_key", [""])[0].strip()
            csrf = values.get("csrf", [""])[0]
            base_url = values.get("base_url", [""])[0].strip()
            model = values.get("model", [""])[0].strip()
            if csrf != self.token or provider not in PROVIDERS or not key:
                raise ValueError("配置内容不完整或页面已失效")
            if base_url and not base_url.startswith(("https://", "http://")):
                raise ValueError("接口地址必须以 https:// 或 http:// 开头")
            label, env_name, runtime_provider = PROVIDERS[provider]
            set_user_env(env_name, key)
            set_user_env("LABELAGENT_EXPERT_PROVIDER", runtime_provider)
            set_user_env("LABELAGENT_EXPERT_BASE_URL", base_url)
            set_user_env("LABELAGENT_EXPERT_MODEL", model)
            self.send_page(page(self.token, f"{label} 已安全保存。请返回 Codex 并回复：API 已配置。", True))
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        except Exception as exc:
            self.send_page(page(self.token, str(exc)), 400)

    def log_message(self, *_):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18774)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    Handler.token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/{Handler.token}"
    if args.open:
        webbrowser.open(url, new=2)
    print(url, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
