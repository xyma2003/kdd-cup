from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, APITimeoutError, OpenAI


def _extract_final_answer_from_thinking(reasoning_content: str) -> str:
    """从 Qwen3 的 reasoning_content 里提取最终答案。

    SiliconFlow 的 Qwen3.5 实现把最终答案也埋在 reasoning_content 里，
    content 字段为空。最终答案通常在推理过程的末尾：
    - 最后一个代码块（```sql ... ``` 或 ```json ... ``` 或 ``` ... ```）
    - 或者 "Output exactly X" / "Final Decision: X" 后面的内容
    - 或者最后几行的非 bullet 文本
    """
    if not reasoning_content:
        return ""

    # 策略1：找最后一个代码块（SQL/JSON/裸代码）
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)\n?\s*```", reasoning_content, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # 策略2：找 "Output exactly `X`" 或 "output: X" 模式
    output_match = re.search(
        r"(?:output exactly|final output|final answer|output)[:\s]+`([^`]+)`",
        reasoning_content, re.IGNORECASE
    )
    if output_match:
        return output_match.group(1).strip()

    # 策略3：找最后一个非 bullet、非标题、有实质内容的段落
    paragraphs = [p.strip() for p in reasoning_content.strip().split("\n\n") if p.strip()]
    for para in reversed(paragraphs):
        # 跳过 bullet list 段落（大多数以 * 或数字开头）
        lines = para.splitlines()
        non_bullet = [l for l in lines if l.strip() and not re.match(r"^\s*[\*\-\d]", l)]
        if non_bullet:
            candidate = "\n".join(non_bullet).strip()
            if candidate:
                return candidate

    # 兜底：返回最后200字符
    return reasoning_content.strip()[-200:].strip()


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=180.0,  # 180s: thinking mode needs extra time for reasoning
        )

        model_lower = self.model.lower()
        is_gemini = "gemini" in model_lower
        # Qwen3/Qwen3.5：开启 thinking mode 让模型先推理再回答，提升复杂任务准确率。
        # SiliconFlow 的实现把最终答案也放在 reasoning_content 里（content 为空），
        # 需要用 _extract_final_answer_from_thinking() 从 reasoning_content 提取。
        is_qwen3 = "qwen3" in model_lower

        extra_body: dict[str, Any] = {}
        if is_gemini:
            extra_body = {
                "google": {
                    "thinking_config": {
                        "include_thoughts": False,
                        "thinking_budget": 0,
                    }
                }
            }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": 8192,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        # 429 TPM 指数退避重试（最多 5 次，等待 60/120/180/240/300s）
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                response = client.chat.completions.create(**kwargs)
                last_exc = None
                break
            except APITimeoutError as exc:
                raise RuntimeError(f"LLM call timed out after 180s: {exc}") from exc
            except APIError as exc:
                last_exc = exc
                if hasattr(exc, "status_code") and exc.status_code == 429:
                    wait = 60 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Model request failed: {exc}") from exc
        if last_exc is not None:
            raise RuntimeError(f"Model request failed after 5 retries: {last_exc}") from last_exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")

        message = choices[0].message
        content = message.content

        # Qwen3/Qwen3.5 on SiliconFlow: content is empty, answer is in reasoning_content
        if is_qwen3 and (not isinstance(content, str) or not content.strip()):
            reasoning = getattr(message, "reasoning_content", None) or ""
            if reasoning.strip():
                content = _extract_final_answer_from_thinking(reasoning)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model response missing text content.")
        return content.strip()

    def chat(self, messages: list[dict]) -> str:
        """Convenience wrapper accepting raw dicts instead of ModelMessage objects."""
        model_messages = [ModelMessage(role=m["role"], content=m["content"]) for m in messages]
        return self.complete(model_messages)


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
