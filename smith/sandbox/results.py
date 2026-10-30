"""Result of one sandbox execution and its rendering for the LLM."""

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Everything the agent loop needs to know about one execution.

    Each boolean flag maps to a situation the LLM must be told about
    explicitly, so it never has to guess what happened.
    """

    output: str = Field(default="", description="Captured stdout/stderr")
    error: str | None = Field(default=None, description="Formatted error")
    timed_out: bool = False
    memory_exceeded: bool = False
    output_truncated: bool = False
    worker_restarted: bool = Field(
        default=False,
        description="Worker was replaced: all variables were lost",
    )
    final_answer_called: bool = False
    final_answer: str | None = None
    notes: list[str] = Field(
        default_factory=list,
        description="Extra explicit feedback (e.g. truncated tool output)",
    )

    def to_observation(
        self,
        empty_message: str = "(no output: use print() to display values)",
    ) -> str:
        """Render the result as the text fed back to the LLM.

        Args:
            empty_message: Text returned when there is nothing to report.
        """
        parts: list[str] = []
        if self.output:
            parts.append(self.output.rstrip("\n"))
        if self.error:
            parts.append(f"[Error]\n{self.error.rstrip()}")
        if self.output_truncated:
            parts.append(
                "[Note] Output was truncated due to size limits. "
                "Print less (e.g. slices) to see the rest."
            )
        if self.timed_out:
            parts.append(
                "[Note] Execution hit the timeout and was killed: "
                "the output above is partial."
            )
        if self.memory_exceeded:
            parts.append("[Note] Execution exceeded the memory limit.")
        if self.worker_restarted:
            parts.append(
                "[Note] The sandbox was restarted: all variables defined "
                "in previous steps were lost."
            )
        parts.extend(f"[Note] {note}" for note in self.notes)
        if self.final_answer_called:
            parts.append("[Note] final_answer() was called.")
        if not parts:
            return empty_message
        return "\n".join(parts)
