"""Configuration models and TOML loading.

Credentials are never passed on the command line. Config files reference
environment variables with `${VAR}` or `${VAR:-fallback}`, which are expanded at
load time; API keys are held as `SecretStr` so they cannot leak into logs or
repr output by accident.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from .errors import ConfigError

load_dotenv()

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand `${VAR}` and `${VAR:-default}` against the environment.

    A missing variable with no default is an error rather than an empty string,
    because silently sending an empty API key produces a confusing 401 much
    later in the run.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        env = os.environ.get(name)
        if env is not None:
            return env
        if default is not None:
            return default
        raise ConfigError(
            f"environment variable {name!r} is referenced in the config but is not set"
        )

    return _ENV_PATTERN.sub(replace, value)


def _expand_tree(node: Any) -> Any:
    """Recursively expand env references through a parsed TOML tree."""
    if isinstance(node, str):
        return expand_env(node)
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    return node


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointConfig(Base):
    """An OpenAI-compatible inference endpoint.

    `base_url` covers vLLM, TGI, Together, Groq, OpenRouter, Ollama and anything
    else speaking the same protocol. Leave it unset for OpenAI itself.
    """

    model: str
    base_url: str | None = None
    api_key: SecretStr = SecretStr("")
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class FigureConfig(Base):
    """Thresholds for deciding a page carries visual content worth describing.

    Defaults are tuned to catch charts and diagrams while ignoring rules,
    borders, and letterhead. Tune them with `pyqa inspect` before spending money.
    """

    min_raster_area_ratio: float = Field(default=0.05, gt=0, le=1)
    min_vector_paths: int = Field(default=12, ge=1)
    margin_band_ratio: float = Field(default=0.10, ge=0, lt=0.5)
    boilerplate_page_ratio: float = Field(default=0.30, gt=0, le=1)


class VisionConfig(Base):
    """The single stage that talks to a vision model."""

    endpoint: EndpointConfig | None = None
    describe_figures: bool = True
    table_reread: bool = True
    dpi: int = Field(default=200, ge=72, le=600)
    max_image_edge: int = Field(default=2000, ge=256)
    concurrency: int = Field(default=8, ge=1)
    max_vision_pages: int | None = Field(default=None, ge=0)
    failure_budget: int = Field(default=10, ge=0)
    figures: FigureConfig = Field(default_factory=FigureConfig)


class ChunkConfig(Base):
    """Chonkie SemanticChunker settings.

    Defaults mirror the library's own, except `chunk_size`, which we lower so a
    chunk comfortably fits a generation prompt alongside instructions.
    """

    embedding_model: str = "minishlab/potion-base-32M"
    threshold: float = Field(default=0.8, gt=0, lt=1)
    chunk_size: int = Field(default=1024, ge=64)
    similarity_window: int = Field(default=3, ge=1)
    min_sentences_per_chunk: int = Field(default=1, ge=1)
    min_characters_per_sentence: int = Field(default=24, ge=1)
    skip_window: int = Field(default=0, ge=0)
    min_chunk_chars: int = Field(default=80, ge=0)

    @field_validator("threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        # Chonkie rejects 0.0 and 1.0 with an unhelpful message; catch it here.
        if not (0.0 < v < 1.0):
            raise ValueError("threshold must be strictly between 0 and 1 (exclusive)")
        return v


class GenerateConfig(Base):
    endpoint: EndpointConfig | None = None
    pairs_per_chunk: int = Field(default=5, ge=1, le=50)
    concurrency: int = Field(default=8, ge=1)
    failure_budget: int = Field(default=10, ge=0)
    system_prompt: str = (
        "You are a helpful assistant answering questions about the provided document."
    )


class GroundConfig(Base):
    """Grounding filter. Falls back to the generation endpoint when unset."""

    enabled: bool = True
    endpoint: EndpointConfig | None = None
    concurrency: int = Field(default=8, ge=1)
    failure_budget: int = Field(default=10, ge=0)
    keep_on_error: bool = True  # a checker failure shouldn't silently drop good data


class OutputConfig(Base):
    dir: Path = Path("./out")
    dataset_name: str = "dataset.jsonl"
    rejected_name: str = "rejected.jsonl"
    manifest_name: str = "manifest.json"
    include_provenance: bool = False


class Config(Base):
    vision: VisionConfig = Field(default_factory=VisionConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    ground: GroundConfig = Field(default_factory=GroundConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cache_dir: Path = Path(".pyqa-cache")
    use_cache: bool = True

    @model_validator(mode="after")
    def _ground_endpoint_fallback(self) -> Config:
        if self.ground.enabled and self.ground.endpoint is None:
            self.ground.endpoint = self.generate.endpoint
        return self

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a TOML config, expanding environment references."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            raw = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {p}: {exc}") from exc
        return cls.model_validate(_expand_tree(raw))
