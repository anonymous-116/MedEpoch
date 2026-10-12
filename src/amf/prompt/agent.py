SYSTEM_INSTRUCTION = """Answer the task using the provided context.
Follow the current Output Format and return exactly one permitted decision for
this turn."""

COT_SYSTEM_INSTRUCTION = """Answer the task."""

COT_OUTPUT_FORMAT = ""

FINAL_OUTPUT_FORMAT = """If you want to provide a final answer, use the following format:
Thought: <optional short reasoning>
Final Answer: <your answer>
"""

FORCED_FINAL_OUTPUT_FORMAT = """Final Answer: <your answer>"""

TOOL_OUTPUT_FORMAT = """If you want to call a tool, use the following format:
Thought: <optional short reasoning>
Action: <one tool name listed in the Tools section>
Action Input: <a valid JSON object matching that tool's Input format>
"""

STRICT_OUTPUT_FORMAT_INSTRUCTION = """You must strictly follow the format that applies to your response.
Output plain text only. Do not use Markdown formatting."""

FINAL_TURN_INSTRUCTION = """This is the final turn. Use the information already
available and answer now. Do not call any tool."""

MODE_INSTRUCTIONS = {
    "react": "",
    "epoch": (
        "Use the global Skill as general solving guidance and retrieved Memories "
        "as conditional reusable medical information. Memories may be provisional "
        "or disputed, so apply them only when their conditions match."
    ),
}
