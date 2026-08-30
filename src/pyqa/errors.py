"""Exception hierarchy. Every failure pyqa raises deliberately inherits PyqaError."""

from __future__ import annotations


class PyqaError(Exception):
    """Base for all pyqa errors."""


class ConfigError(PyqaError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ParseError(PyqaError):
    """The PDF could not be read or produced no usable text."""


class VisionError(PyqaError):
    """A page could not be rendered or read by the vision engine."""


class ChunkError(PyqaError):
    """The document could not be chunked."""


class GenerationError(PyqaError):
    """The LLM did not return usable output after exhausting retries."""


class StageBudgetExceeded(PyqaError):
    """A stage exceeded its allowed failure budget and aborted the run."""

    def __init__(self, stage: str, failures: int, budget: int) -> None:
        self.stage = stage
        self.failures = failures
        self.budget = budget
        super().__init__(
            f"stage {stage!r} exceeded its failure budget: "
            f"{failures} failures with a budget of {budget}"
        )
