"""What the agent loop needs from an LLM client.

smith.llm.client.LLMClient implements it; tests use a scripted fake.
"""

from typing import Protocol

from smith.models import StepMetrics


class ChatModel(Protocol):
    """Anything that can complete a chat conversation."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        step: int,
        stop: list[str] | None,
        max_tokens: int | None,
        timeout: float | None,
    ) -> StepMetrics:
        """Return the answer as a StepMetrics with the LLM fields filled.

        Generation stops at any string of `stop`. `timeout` caps the whole
        call, retries included. Raises once every retry has failed.
        """
        ...
