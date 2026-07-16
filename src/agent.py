import logging
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import nvidia, openai, silero

import tools  # noqa: F401 – registers agent_tools
from memory.short_term import ShortTermMemory
from memory.summarizer import ConversationSummarizer
from tools.registry import agent_tools

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger("agent")


def _create_llm() -> openai.LLM:
    return openai.LLM(
        model=os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def prewarm(proc: JobProcess) -> None:
    logger.info("Prewarming agent")
    proc.userdata["short_term"] = ShortTermMemory(50)
    proc.userdata["summarizer"] = ConversationSummarizer()


async def entrypoint(job: JobContext) -> None:
    logger.info("Agent entering room %s", job.room.name)
    await job.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    short_term: ShortTermMemory = job.proc.userdata["short_term"]
    summarizer: ConversationSummarizer = job.proc.userdata["summarizer"]

    agent = Agent(
        instructions=(
            "You are a helpful AI Voice Assistant. "
            "You can see the user via their camera and screen share. "
            "Speak naturally. Keep responses concise. "
            "Only use tools when the user explicitly asks for weather, "
            "calculations, or web search. Otherwise just reply conversationally."
        ),
        tools=agent_tools or None,
    )

    llm_instance = _create_llm()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=nvidia.STT(),
        llm=llm_instance,
        tts=nvidia.TTS(),
    )

    @session.on("agent_state_changed")
    async def on_agent_state(state):
        await job.room.local_participant.publish_data(
            json.dumps({"type": "agent_state", "data": state}),
            reliable=True,
        )

    @session.on("user_input_transcribed")
    async def on_user_transcript(evt):
        await job.room.local_participant.publish_data(
            json.dumps({
                "type": "transcript",
                "data": {
                    "role": "user",
                    "text": evt.transcript,
                    "is_final": evt.is_final,
                    "id": f"user-{id(evt):x}",
                },
            }),
            reliable=True,
        )

    @session.on("conversation_item_added")
    def on_conversation_item(evt):
        item = evt.item
        if item.type != "message" or item.role != "assistant":
            return

        text = item.text_content or ""
        short_term.add_turn({"role": "assistant", "content": text})

        if short_term.turn_count >= 20 and short_term.is_full:
            logger.info("Triggering conversation summarization")
            _generate_summary(summarizer, short_term, llm_instance)

    await session.start(agent=agent, room=job.room)

    # Signal the UI that we're ready
    await job.room.local_participant.publish_data(
        json.dumps({"type": "agent_state", "data": "listening"}),
        reliable=True,
    )

    logger.info("Agent ready")


async def _generate_summary(
    summarizer: ConversationSummarizer,
    short_term: ShortTermMemory,
    llm_instance: openai.LLM,
) -> None:
    history = short_term.get_history()[-30:]
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    try:
        chat_ctx = llm.ChatContext(
            messages=[
                llm.ChatMessage(role="system", text="Summarize the following conversation concisely."),
                llm.ChatMessage(role="user", text=conversation),
            ],
        )
        stream = llm_instance.chat(chat_ctx=chat_ctx)
        response = await stream.collect()
        summary = response.text
        if summary:
            summarizer.set_summary(summary)
            short_term.clear()
    except Exception:
        logger.exception("Summarization failed")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="voice-agent",
        ),
    )
