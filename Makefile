# macOS + uv workaround: uv writes the editable-install .pth with the UF_HIDDEN
# flag, and CPython's site.addpackage skips hidden .pth files, so `import hootie`
# breaks after every sync. Unhiding it restores the editable install.
.PHONY: sync test lint fixtures logo

sync:
	uv sync
	@chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

test: sync
	uv run pytest -q

lint:
	uv run ruff check src tests assets
	uv run ruff format --check src tests assets

fixtures:
	uv run python tests/fixtures/make_fixtures.py

logo:
	uv run python assets/make_logo.py
