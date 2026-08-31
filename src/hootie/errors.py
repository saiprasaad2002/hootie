"""Exception hierarchy. Every failure hootie raises deliberately inherits HootieError."""

from __future__ import annotations


class HootieError(Exception):
    """Base for all hootie errors."""


class ConfigError(HootieError):
    """Configuration is missing, malformed, or internally inconsistent."""


class ParseError(HootieError):
    """The PDF could not be read or produced no usable text."""


class VisionError(HootieError):
    """A page could not be rendered or read by the vision engine."""


class ChunkError(HootieError):
    """The document could not be chunked."""


class GenerationError(HootieError):
    """The LLM did not return usable output after exhausting retries."""


class StageBudgetExceeded(HootieError):
    """A stage exceeded its allowed failure budget and aborted the run."""

    def __init__(self, stage: str, failures: int, budget: int) -> None:
        self.stage = stage
        self.failures = failures
        self.budget = budget
        super().__init__(
            f"stage {stage!r} exceeded its failure budget: "
            f"{failures} failures with a budget of {budget}"
        )
