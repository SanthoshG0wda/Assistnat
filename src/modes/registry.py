import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from livekit.agents import llm
from livekit.agents.llm.tool_context import FunctionTool
from livekit.agents.voice import Agent

from tools.registry import agent_tools

logger = logging.getLogger("modes")

MODE_SYSTEM_TAG = "[MODE:"


@dataclass
class ModeConfig:
    id: str
    name: str
    description: str
    icon: str = "sparkles"
    system_prompt: str = ""
    tools: list[str] = field(
        default_factory=lambda: ["get_weather", "calculate", "web_search"]
    )
    voice: dict = field(
        default_factory=lambda: {
            "personality": "friendly",
            "speed": 1.0,
            "tone": "warm",
        }
    )
    model: dict = field(
        default_factory=lambda: {
            "id": "meta/llama-3.1-8b-instruct",
            "temperature": 0.7,
            "provider": "nvidia",
        }
    )
    vision_enabled: bool = True
    reasoning_enabled: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "vision_enabled": self.vision_enabled,
            "reasoning_enabled": self.reasoning_enabled,
            "voice": self.voice,
            "model": self.model,
            "tools": self.tools,
        }


class ModeRegistry:
    def __init__(self, config_dir: str | Path) -> None:
        self._modes: dict[str, ModeConfig] = {}
        self._current_id: str = "general"
        self._load_configs(Path(config_dir))

    def _load_configs(self, config_dir: Path) -> None:
        if not config_dir.is_dir():
            logger.warning("Mode config dir %s not found", config_dir)
            return
        for f in sorted(config_dir.glob("*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                mode = ModeConfig(**data)
                self._modes[mode.id] = mode
                logger.info("Loaded mode: %s (%s)", mode.id, f.name)
            except Exception:
                logger.exception("Failed to load mode config: %s", f)

        if not self._modes:
            logger.warning("No mode configs loaded — adding default General mode")
            self._modes["general"] = ModeConfig(
                id="general",
                name="General",
                description="Default all-purpose assistant",
            )
        self._current_id = "general"

    @property
    def current(self) -> ModeConfig:
        return self._modes[self._current_id]

    @property
    def current_id(self) -> str:
        return self._current_id

    @property
    def all(self) -> list[ModeConfig]:
        return list(self._modes.values())

    def all_public(self) -> list[dict[str, Any]]:
        return [m.to_public_dict() for m in self.all]

    def switch_to(self, mode_id: str) -> ModeConfig:
        if mode_id not in self._modes:
            logger.warning("Unknown mode '%s' — falling back to general", mode_id)
            mode_id = "general"
        self._current_id = mode_id
        logger.info("Switched to mode: %s (%s)", mode_id, self.current.name)
        return self.current

    def get_filtered_tools(self) -> list[FunctionTool]:
        mode = self.current
        tool_names = mode.tools
        return [t for t in agent_tools if t.name in tool_names]

    def get_system_prompt(self) -> str:
        return self.current.system_prompt

    def apply_to_agent(self, agent: Agent) -> None:
        mode = self.current
        old_tag = f"{MODE_SYSTEM_TAG}"
        new_messages: list[llm.ChatMessage] = []
        for msg in agent.chat_ctx.messages:
            text = msg.text or ""
            if msg.role == "system" and old_tag in text:
                continue
            new_messages.append(msg)
        new_messages.append(
            llm.ChatMessage(
                role="system",
                text=f"{MODE_SYSTEM_TAG}{mode.name}]\n{mode.system_prompt}",
            )
        )
        ctx = agent.chat_ctx.copy()
        ctx.messages = new_messages
        agent._chat_ctx = ctx
        agent.tools = self.get_filtered_tools()

    def build_init_chat_ctx(self) -> llm.ChatContext:
        mode = self.current
        ctx = llm.ChatContext(
            messages=[
                llm.ChatMessage(
                    role="system",
                    text=(
                        f"{MODE_SYSTEM_TAG}{mode.name}]\n"
                        f"{mode.system_prompt}\n\n"
                        f"Current mode: {mode.name}. "
                        f"Voice personality: {mode.voice.get('personality', 'friendly')}."
                    ),
                ),
            ]
        )
        return ctx
