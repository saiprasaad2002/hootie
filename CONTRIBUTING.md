# Contributing

Thanks for considering a contribution. Issues, bug reports, and pull requests
are all welcome.

## Getting set up

```bash
git clone https://github.com/saiprasaad2002/pyqa
cd pyqa
make sync        # uv sync + a macOS workaround, see below
make test
```

`make sync` runs `uv sync` and then unhides the editable-install `.pth` file.
On macOS, uv writes that file with the `UF_HIDDEN` flag and CPython's
`site.addpackage` deliberately skips hidden `.pth` files, so `import hootie`
breaks silently after any sync. If you hit `ModuleNotFoundError: No module
named 'hootie'` on a fresh sync, that is why:

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth
```

## Running the checks

```bash
make test    # pytest
make lint    # ruff check + ruff format --check
```

The suite runs against a stub OpenAI-compatible server (`tests/stub_server.py`),
so **no API keys are needed**. It does download a small embedding model
(`minishlab/potion-base-32M`, a few MB) from HuggingFace the first time the
chunker runs; after that the cache is warm and the suite is offline.

## Fixtures

PDF fixtures are generated, not committed by hand — `tests/fixtures/make_fixtures.py`
writes raw PDF syntax so the files contain exactly the object types the figure
detector looks for (real vector path objects, real raster XObjects, and a logo
repeated on every page for boilerplate suppression).

```bash
make fixtures
```

## Things worth knowing before you change them

A few behaviours are subtle enough that they have dedicated tests. If one of
these fails, read the test before "fixing" the code.

- **`pdf-inspector` mixes page index conventions.** `extract_pages_markdown`
  is 0-indexed for both its argument and its result; `extract_text_with_positions`
  is 1-indexed for both. Getting this wrong returns the wrong page *silently*.
  Everything is normalized to 1-indexed at the parser boundary, and the adapter
  asserts the invariant. See `tests/test_page_indexing.py`.
- **The vision planner must never assign two tasks to one page.** `ocr`
  subsumes `table_reread` and `describe`; a page is rendered once and billed
  once. See `tests/test_planner.py`.
- **Boilerplate suppression is load-bearing.** Without it, a logo on every page
  bills a vision call on every page.
- **Structured output must degrade, not fail.** Servers disagree about
  `response_format`; the fallback ladder is in
  `src/hootie/generation/openai_compat.py` and must downgrade only on errors that
  genuinely indicate missing support, never on a rate limit.

## Adding an engine

The four extension points are `Protocol`s. To add a parser, chunker, vision
engine, or LLM client, implement the protocol and pass an instance in — there is
no registry to update. Page numbers crossing any boundary are **1-indexed**.

## Pull requests

- Keep `make test` and `make lint` green.
- Add a test for behaviour changes. Prefer testing the pure helpers over the
  network paths where you can; the stub server is there for the rest.
- Add a `CHANGELOG.md` entry under `## [Unreleased]`.
- Commit messages: a short imperative subject, and a body explaining *why*
  where it isn't obvious.

## Reporting bugs

For parsing or detection issues, `hootie inspect yourfile.pdf` output is the most
useful thing to include — it needs no credentials and shows the per-page routing
decisions. Please don't attach confidential documents; a minimal reproducing PDF
is far more useful.
