import asyncio
import logging
import time
from typing import Any

from livekit import rtc
from livekit.agents import llm
from livekit.agents.llm import ChatContext, ImageContent
from livekit.agents.voice import Agent

logger = logging.getLogger("vision")

FRAME_INTERVAL = 2.0
SCENE_CHANGE_THRESHOLD = 30.0
INFERENCE_WIDTH = 512
SOURCE_CAMERA_STR = "camera"
SOURCE_SCREEN_SHARE_STR = "screen_share"


def _mse(a: rtc.VideoFrame, b: rtc.VideoFrame) -> float:
    if a.width != b.width or a.height != b.height:
        return float("inf")
    fa = a.data
    fb = b.data
    if len(fa) != len(fb):
        return float("inf")
    n = len(fa) // 4
    total = 0.0
    for i in range(n):
        diff = abs(int(fa[i * 4]) - int(fb[i * 4]))
        total += diff * diff
    return total / n if n else float("inf")


class VisionService:
    def __init__(self, room: rtc.Room, agent: Agent, vision_llm: llm.LLM | None = None):
        self._room = room
        self._agent = agent
        self._vision_llm = vision_llm
        self._tasks: list[asyncio.Task[Any]] = []
        self._last_camera_frame: rtc.VideoFrame | None = None
        self._last_screen_frame: rtc.VideoFrame | None = None
        self._last_inject_time: float = 0.0
        self._last_analysis: str | None = None

    async def start(self) -> None:
        self._room.on("track_published", self._on_track_published)
        for participant in self._room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.source in (SOURCE_CAMERA_STR, SOURCE_SCREEN_SHARE_STR):
                    self._start_track_consumer(pub, participant)

    def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    def _on_track_published(
        self,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if pub.source in (SOURCE_CAMERA_STR, SOURCE_SCREEN_SHARE_STR):
            self._start_track_consumer(pub, participant)

    def _start_track_consumer(
        self,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        async def _wait_and_consume() -> None:
            try:
                track = await pub.wait_for_subscription()
                stream = rtc.VideoStream(track)
                task = asyncio.create_task(
                    self._consume_frames(stream, pub.source),
                    name=f"vision-{pub.source}-{participant.identity}",
                )
                self._tasks.append(task)
                task.add_done_callback(self._tasks.remove)
            except Exception:
                logger.exception("Failed to start vision consumer")

        task = asyncio.create_task(_wait_and_consume())
        self._tasks.append(task)

    async def _consume_frames(
        self, stream: rtc.VideoStream, source: str
    ) -> None:
        try:
            async for event in stream:
                frame = event.frame
                now = time.time()
                if now - self._last_inject_time < FRAME_INTERVAL:
                    continue

                if source == SOURCE_CAMERA_STR:
                    prev = self._last_camera_frame
                    self._last_camera_frame = frame
                else:
                    prev = self._last_screen_frame
                    self._last_screen_frame = frame

                if prev is None or _mse(prev, frame) > SCENE_CHANGE_THRESHOLD:
                    self._last_inject_time = now
                    await self._inject_visual_context(frame, source)
        except Exception:
            logger.debug("Vision frame consumer stopped")

    async def _inject_visual_context(
        self, frame: rtc.VideoFrame, source: str
    ) -> None:
        label = "camera" if source == SOURCE_CAMERA_STR else "screen share"

        if self._vision_llm:
            try:
                vision_ctx = ChatContext(
                    messages=[
                        llm.ChatMessage(
                            role="system",
                            text="Describe what you see in this image briefly.",
                        ),
                        llm.ChatMessage(
                            role="user",
                            content=[
                                ImageContent(
                                    image=frame,
                                    inference_width=INFERENCE_WIDTH,
                                    inference_detail="low",
                                )
                            ],
                        ),
                    ]
                )
                stream_resp = self._vision_llm.chat(chat_ctx=vision_ctx)
                resp = await stream_resp.collect()
                self._last_analysis = resp.text
                logger.info("Vision analysis (%s): %s", label, self._last_analysis[:100])
            except Exception:
                logger.exception("Vision LLM analysis failed")
                self._last_analysis = None

        image_content = ImageContent(
            image=frame,
            inference_width=INFERENCE_WIDTH,
            inference_detail="low",
        )
        try:
            ctx = self._agent.chat_ctx.copy()
            ctx.add_message(
                role="user",
                content=[
                    f"[{label} frame at {time.strftime('%H:%M:%S')}]",
                    image_content,
                ],
            )
            await self._agent.update_chat_ctx(ctx)
        except Exception:
            logger.exception("Failed to inject visual context")

    @property
    def last_analysis(self) -> str | None:
        return self._last_analysis
