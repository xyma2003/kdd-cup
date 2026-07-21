from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_format_correction_prompt,
    build_force_answer_prompt,
    build_observation_prompt,
    build_reflection_prompt,
    build_system_prompt,
    build_task_prompt,
    prepare_task_context,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 30
    max_retries_per_step: int = 2
    history_compress_after: int = 8   # compress history after this many steps
    soft_token_budget: int = 50_000   # chars; warn and compress
    hard_token_budget: int = 250_000  # chars; force answer
    force_answer_last_n: int = 2      # force answer on last N steps
    max_consecutive_errors: int = 5   # force answer after N errors in a row


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    # Try to find bare JSON object
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _estimate_chars(messages: list[ModelMessage]) -> int:
    return sum(len(m.content) for m in messages)


def _should_reflect(steps: list[StepRecord], max_steps: int) -> tuple[bool, str]:
    """Detect if agent is stuck and needs reflection nudge."""
    if len(steps) < 3:
        return False, ""

    # Same non-execute action repeated 3 times in a row
    last_actions = [s.action for s in steps[-3:]]
    if len(set(last_actions)) == 1 and last_actions[0] != "execute_python":
        return True, f"You have called '{last_actions[0]}' 3 times in a row with no progress. Try a completely different approach."

    # Used 70% of steps with no successful answer
    if len(steps) >= int(max_steps * 0.7):
        recent_ok = sum(1 for s in steps[-5:] if s.ok)
        if recent_ok == 0:
            return True, "You have used many steps without making progress. Reconsider your approach — try execute_python with a fresh strategy."

    return False, ""


def _compress_history(steps: list[StepRecord]) -> list[StepRecord]:
    """Keep only the last 3 steps to reduce context length."""
    if len(steps) <= 3:
        return steps
    # Summarize older steps as a single note
    return steps[-3:]


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        context_info: str,
        extra_user_message: str | None = None,
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task, context_info)))

        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )

        if extra_user_message:
            messages.append(ModelMessage(role="user", content=extra_user_message))

        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        context_info = prepare_task_context(task)
        consecutive_errors = 0

        for step_index in range(1, self.config.max_steps + 1):
            # Compress history if too long
            if len(state.steps) > self.config.history_compress_after:
                state.steps = _compress_history(state.steps)

            # Check if we should force answer
            messages = self._build_messages(task, state, context_info)
            total_chars = _estimate_chars(messages)
            is_last_steps = step_index >= self.config.max_steps - self.config.force_answer_last_n + 1
            is_over_budget = total_chars > self.config.hard_token_budget
            is_too_many_errors = consecutive_errors >= self.config.max_consecutive_errors

            extra_message: str | None = None
            if is_over_budget or is_last_steps or is_too_many_errors:
                extra_message = build_force_answer_prompt()
            else:
                should_reflect, reason = _should_reflect(state.steps, self.config.max_steps)
                if should_reflect:
                    extra_message = build_reflection_prompt(reason)

            if extra_message:
                messages = self._build_messages(task, state, context_info, extra_message)

            # Call model with retry on format error
            raw_response = self.model.complete(messages)
            parsed_ok = False

            for _retry in range(self.config.max_retries_per_step + 1):
                try:
                    model_step = parse_model_step(raw_response)
                    parsed_ok = True
                    break
                except Exception:
                    if _retry < self.config.max_retries_per_step:
                        # Inject format correction and retry
                        correction_messages = self._build_messages(task, state, context_info)
                        correction_messages.append(ModelMessage(role="assistant", content=raw_response))
                        correction_messages.append(ModelMessage(role="user", content=build_format_correction_prompt()))
                        raw_response = self.model.complete(correction_messages)

            if not parsed_ok:
                # Still failed after retries
                consecutive_errors += 1
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__parse_error__",
                        action_input={},
                        raw_response=raw_response,
                        observation={"ok": False, "error": "Failed to parse model response after retries."},
                        ok=False,
                    )
                )
                continue

            # Execute tool
            try:
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                consecutive_errors = 0 if tool_result.ok else consecutive_errors + 1

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break

            except Exception as exc:
                consecutive_errors += 1
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation={"ok": False, "error": str(exc)},
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
