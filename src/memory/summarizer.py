class ConversationSummarizer:
    def __init__(self):
        self._summary: str | None = None

    @property
    def summary(self) -> str | None:
        return self._summary

    def set_summary(self, summary: str) -> None:
        self._summary = summary

    def clear(self) -> None:
        self._summary = None
