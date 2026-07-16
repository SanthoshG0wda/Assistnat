from livekit.agents import llm

from .registry import register


@llm.function_tool
async def web_search(query: str) -> str:
    """Search the web for current information."""
    return f'Search results for "{query}": This is a simulated result.'


register(web_search)
