"""CLI behaviour, including output hygiene."""

from __future__ import annotations

import json
import logging

from typer.testing import CliRunner

from pyqa.cli import app

runner = CliRunner()


def test_inspect_needs_no_credentials(native_pdf):
    result = runner.invoke(app, ["inspect", str(native_pdf)])
    assert result.exit_code == 0
    assert "3 pages" in result.stdout
    assert "No vision calls needed" in result.stdout


def test_inspect_projects_vision_cost(figures_pdf):
    result = runner.invoke(app, ["inspect", str(figures_pdf)])
    assert result.exit_code == 0
    assert "describe" in result.stdout
    assert "2 of 4 pages" in result.stdout


def test_no_figures_flag_zeroes_the_projection(figures_pdf):
    result = runner.invoke(app, ["inspect", str(figures_pdf), "--no-figures"])
    assert result.exit_code == 0
    assert "0 of 4 pages" in result.stdout


def test_max_vision_pages_caps_the_projection(figures_pdf):
    result = runner.invoke(app, ["inspect", str(figures_pdf), "--max-vision-pages", "1"])
    assert result.exit_code == 0
    assert "1 of 4 pages" in result.stdout


def test_http_client_loggers_are_quiet():
    """The openai 3.x SDK logs through `httpx2`; leaving it at INFO buries progress."""
    from pyqa.cli import main

    main(verbose=False)
    for name in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub"):
        assert logging.getLogger(name).level >= logging.WARNING, name


def test_run_writes_a_dataset(native_pdf, stub_config, tmp_path):
    """Drive the real `run` command through a config file, as a user would."""
    config_file = tmp_path / "pyqa.toml"
    endpoint = stub_config.generate.endpoint
    config_file.write_text(
        f"""
[generate.endpoint]
model = "{endpoint.model}"
base_url = "{endpoint.base_url}"

[ground.endpoint]
model = "{endpoint.model}"
base_url = "{endpoint.base_url}"

[output]
dir = "{tmp_path / "out"}"
"""
    )
    result = runner.invoke(app, ["run", str(native_pdf), "-c", str(config_file)])
    assert result.exit_code == 0, result.stdout

    dataset = tmp_path / "out" / "dataset.jsonl"
    assert dataset.is_file()
    assert json.loads(dataset.read_text().splitlines()[0])["messages"]


def test_missing_config_file_is_a_clean_error(native_pdf, tmp_path):
    result = runner.invoke(app, ["inspect", str(native_pdf), "-c", str(tmp_path / "nope.toml")])
    assert result.exit_code != 0
