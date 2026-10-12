from __future__ import annotations

from typing import Any

INTERMEDIATE_REPAIR_INSTRUCTION = """Your previous response did not follow the required format.

If you want to provide a final answer, use the following format:

Thought: <optional short reasoning>
Final Answer: <your answer>

If you want to call a tool, use the following format:

Thought: <optional short reasoning>
Action: <one tool name listed in the Tools section>
Action Input: <a valid JSON object matching that tool's Input format>

Output plain text only. Do not use Markdown formatting."""

FINAL_REPAIR_INSTRUCTION = """Your previous response did not follow the required format.

This is the final turn. If you want to provide a final answer, start your response with "Final Answer:".

Output plain text only. Do not use Markdown formatting.
"""

def build_protocol_repair_messages(
    *,
    original_prompt: str,
    invalid_output: str,
    force_final: bool,
) -> list[dict[str, Any]]:
    """Build a transient chat that asks the model to repair its own output.

    These messages are used for the paid repair call only. They are deliberately
    never appended to EpisodeState, so later solving prompts contain only the
    accepted ReAct decision.
    """

    instruction = (
        FINAL_REPAIR_INSTRUCTION if force_final else INTERMEDIATE_REPAIR_INSTRUCTION
    )
    return [
        {"role": "system", "content": original_prompt},
        {"role": "assistant", "content": invalid_output},
        {"role": "user", "content": instruction},
    ]
