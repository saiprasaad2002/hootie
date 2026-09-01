# macOS + uv workaround: uv writes the editable-install .pth with the UF_HIDDEN
# flag, and CPython's site.addpackage skips hidden .pth files, so `import hootie`
# breaks after every sync. Unhiding it restores the editable install.
.PHONY: sync test lint fixtures logo dist release-check

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

# Always build from a clean dist/. This directory sits in an iCloud-synced
# folder, which silently leaves "name 2.whl" duplicates behind; `twine upload
# dist/*` would then try to upload the stale copies too.
dist:
	rm -rf dist
	uv build
	uvx twine check --strict dist/*

# Everything that must be green before tagging a release.
release-check: sync
	uv run pytest -q
	uv run ruff check src tests assets
	uv run ruff format --check src tests assets
	$(MAKE) dist
	@ls -1 dist/
