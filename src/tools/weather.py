from livekit.agents import llm

from .registry import register


@llm.function_tool
async def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is currently sunny and 22°C."


register(get_weather)
