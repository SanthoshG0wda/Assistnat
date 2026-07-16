import re

from livekit.agents import llm

from .registry import register


@llm.function_tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression (+, -, *, /, **, %)."""
    cleaned = re.sub(r"[^0-9+\-*/.%() ]", "", str(expression))
    try:
        result = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception:
        return "Error: Invalid expression"


register(calculate)
