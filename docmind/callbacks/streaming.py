"""A minimal queue-backed streaming callback handler.

LangGraph's ``stream_mode="messages"`` is the primary streaming mechanism used by
the app, but this official ``BaseCallbackHandler`` subclass is provided for
callers that want to consume tokens through the standard callbacks interface
(e.g. wiring a raw chain into Streamlit).
"""

from __future__ import annotations

import queue
from typing import Any, Iterator

from langchain_core.callbacks.base import BaseCallbackHandler

_DONE = object()


class QueueCallbackHandler(BaseCallbackHandler):
    """Collect streamed LLM tokens into a thread-safe queue."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Any]" = queue.Queue()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self._queue.put(token)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._queue.put(_DONE)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._queue.put(_DONE)

    def iter_tokens(self) -> Iterator[str]:
        while True:
            item = self._queue.get()
            if item is _DONE:
                return
            yield item
