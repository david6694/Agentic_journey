"""Contract between the agent loop and the LLM client.

The loop only depends on this interface; the real multi-provider client
implements it, tests use a scripted fake.
"""

from typing import Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """One successful completion, with the metrics StepMetrics needs."""

    text: str
    input_tokens: int
    output_tokens: int
    request_time_ms: float
    retries: int = 0
    api_url: str = ""
    model_name: str = ""


class LLMClient(Protocol):
    """Anything that can complete a chat conversation."""

    def complete(
        self, messages: list[dict[str, str]], stop: list[str],
        max_tokens: int, timeout: float,
    ) -> LLMResponse:
        """Return the model's answer to `messages`.

        Generation stops at any string of `stop` (not included in text).
        `timeout` is the time left for the task, in seconds: retries and
        fallbacks included, the call must return or raise before it.
        Raises an exception once every retry and fallback has failed.
        """
        ...
