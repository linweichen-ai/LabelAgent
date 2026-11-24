"""Provider-specific HTTP envelopes for the opt-in external VLM expert.

This module is deliberately transport-only.  It neither selects a provider nor
decides whether a machine result is eligible to become training truth.
"""
import base64
import json
import urllib.request
from urllib.parse import quote


SUPPORTED_PROVIDERS = frozenset({
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
})


def validate_external_credential(credential):
    """Reject malformed secrets without ever echoing their value."""
    if not isinstance(credential, str) or not credential:
        raise ValueError("外部专家 VLM 凭据为空")
    if (credential != credential.strip()
            or any(ord(character) < 32 or ord(character) == 127
                   for character in credential)):
        raise ValueError("外部专家 VLM 凭据包含空白或控制字符，请重新安全配置")
    return credential


def _encoded_images(images):
    return [base64.b64encode(data).decode("ascii") for data in images]


def build_external_request(provider, base_url, model, credential, instruction,
                           images, response_schema=None, max_tokens=768):
    """Return a provider-native request without placing credentials in URLs."""
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"暂不支持外部专家 VLM 协议：{provider}")
    credential = validate_external_credential(credential)
    base_url = base_url.rstrip("/")
    encoded = _encoded_images(images)
    headers = {"Content-Type": "application/json"}

    if provider == "openai_chat":
        content = [{"type": "text", "text": instruction}]
        content.extend({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + data},
        } for data in encoded)
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "labelagent", "schema": response_schema,
                    "strict": True,
                },
            }
        headers["Authorization"] = "Bearer " + credential
        url = base_url + "/chat/completions"
    elif provider == "openai_responses":
        content = [{"type": "input_text", "text": instruction}]
        content.extend({
            "type": "input_image",
            "image_url": "data:image/jpeg;base64," + data,
            "detail": "auto",
        } for data in encoded)
        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
        }
        if response_schema is not None:
            payload["text"] = {"format": {
                "type": "json_schema", "name": "labelagent",
                "schema": response_schema, "strict": True,
            }}
        headers["Authorization"] = "Bearer " + credential
        url = base_url + "/responses"
    elif provider == "anthropic_messages":
        content = [{"type": "text", "text": instruction}]
        content.extend({
            "type": "image",
            "source": {
                "type": "base64", "media_type": "image/jpeg", "data": data,
            },
        } for data in encoded)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if response_schema is not None:
            payload["output_config"] = {"format": {
                "type": "json_schema", "schema": response_schema,
            }}
        headers.update({
            "x-api-key": credential,
            "anthropic-version": "2023-06-01",
        })
        url = base_url + "/messages"
    else:
        parts = [{"text": instruction}]
        parts.extend({
            "inline_data": {"mime_type": "image/jpeg", "data": data},
        } for data in encoded)
        generation_config = {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
        }
        if response_schema is not None:
            generation_config.update(
                responseMimeType="application/json",
                responseJsonSchema=response_schema,
            )
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        headers["x-goog-api-key"] = credential
        url = base_url + "/models/" + quote(model, safe="") + ":generateContent"

    return urllib.request.Request(
        url, json.dumps(payload).encode("utf-8"), headers=headers,
        method="POST")


def normalize_external_response(provider, result):
    """Normalize native provider responses to the existing choices contract."""
    if not isinstance(result, dict):
        raise ValueError("外部专家 VLM 返回格式无效")
    if provider == "openai_chat":
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("外部专家 VLM 未返回可解析结果")
        return result

    text = ""
    finish_reason = None
    usage = {}
    if provider == "openai_responses":
        if isinstance(result.get("output_text"), str):
            text = result["output_text"]
        else:
            chunks = []
            for item in result.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for part in item.get("content") or []:
                    if (isinstance(part, dict)
                            and part.get("type") in {"output_text", "text"}
                            and isinstance(part.get("text"), str)):
                        chunks.append(part["text"])
            text = "".join(chunks)
        finish_reason = result.get("status")
        source_usage = result.get("usage") or {}
        usage = {
            "prompt_tokens": source_usage.get("input_tokens"),
            "completion_tokens": source_usage.get("output_tokens"),
            "total_tokens": source_usage.get("total_tokens"),
        }
    elif provider == "anthropic_messages":
        text = "".join(
            part.get("text", "") for part in (result.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "text")
        finish_reason = result.get("stop_reason")
        source_usage = result.get("usage") or {}
        prompt_tokens = source_usage.get("input_tokens")
        completion_tokens = source_usage.get("output_tokens")
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens
                if type(prompt_tokens) is int and type(completion_tokens) is int
                else None),
        }
    elif provider == "gemini_generate_content":
        candidates = result.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        content = candidate.get("content") or {}
        text = "".join(
            part.get("text", "") for part in (content.get("parts") or [])
            if isinstance(part, dict) and isinstance(part.get("text"), str))
        finish_reason = candidate.get("finishReason")
        source_usage = result.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": source_usage.get("promptTokenCount"),
            "completion_tokens": source_usage.get("candidatesTokenCount"),
            "total_tokens": source_usage.get("totalTokenCount"),
        }
    else:
        raise ValueError(f"暂不支持外部专家 VLM 协议：{provider}")

    if not text:
        raise ValueError("外部专家 VLM 未返回文本结果")
    usage = {key: value for key, value in usage.items()
             if type(value) is int and value >= 0}
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": text},
        }],
        "usage": usage,
    }
