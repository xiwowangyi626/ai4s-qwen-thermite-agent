import os
from typing import Any

import httpx

from .safety import SAFETY_SYSTEM_PROMPT


class QwenClientError(RuntimeError):
    """Raised when the Qwen compatible API cannot return a usable answer."""


def _api_key() -> str | None:
    return os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")


def qwen_configured() -> bool:
    return bool(_api_key())


async def call_qwen(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    api_key = _api_key()
    if not api_key:
        raise QwenClientError(
            "Qwen API key is not configured. Set DASHSCOPE_API_KEY or BAILIAN_API_KEY."
        )

    base_url = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = os.getenv("QWEN_MODEL", "qwen-plus")
    url = f"{base_url}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "system", "content": SAFETY_SYSTEM_PROMPT}, *messages],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenClientError("Qwen response did not contain a chat message.") from exc

