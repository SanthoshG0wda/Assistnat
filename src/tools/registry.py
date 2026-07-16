from collections.abc import Callable
from typing import Any

from livekit.agents.llm.tool_context import FunctionTool

agent_tools: list[FunctionTool] = []


def register(func: FunctionTool) -> None:
    agent_tools.append(func)
