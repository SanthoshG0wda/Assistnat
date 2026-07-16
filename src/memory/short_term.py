import time
from typing import Any


class ShortTermMemory:
    def __init__(self, max_turns: int = 50):
        self._turns: list[dict[str, Any]] = []
        self._max_turns = max_turns

    def add_turn(self, turn: dict[str, Any]) -> None:
        entry = {**turn, "timestamp": turn.get("timestamp", time.time())}
        self._turns.append(entry)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._turns)

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self._turns[-n:]

    def clear(self) -> None:
        self._turns = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_full(self) -> bool:
        return len(self._turns) >= self._max_turns
