from __future__ import annotations

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))


@pytest.fixture(scope="session")
def native_pdf() -> Path:
    """A 3-page text-based PDF with no figures."""
    return FIXTURES / "native_3page.pdf"


@pytest.fixture(scope="session")
def figures_pdf() -> Path:
    """A 4-page PDF: a logo on every page, a vector flowchart, a raster chart.

    Regenerated on demand so the suite works from a clean checkout.
    """
    path = FIXTURES / "figures_4page.pdf"
    if not path.exists():
        import make_fixtures

        make_fixtures.build(path)
    return path


@pytest.fixture
def stub():
    """A running OpenAI-compatible stub server. Yields (state, base_url)."""
    import threading

    from stub_server import StubState, make_server

    state = StubState()
    server = make_server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield state, f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def stub_config(stub, tmp_path):
    """A Config wired to the stub, writing into a temp directory."""
    from pyqa.config import Config, EndpointConfig

    _, base_url = stub
    endpoint = EndpointConfig(model="stub-model", base_url=base_url)
    cfg = Config()
    cfg.generate.endpoint = endpoint
    cfg.ground.endpoint = endpoint
    cfg.vision.endpoint = endpoint
    cfg.output.dir = tmp_path / "out"
    cfg.cache_dir = tmp_path / "cache"
    return cfg
