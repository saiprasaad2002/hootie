# pyqa

Turn a PDF into a finetuning-ready QA dataset, using **your own** OCR and LLM
endpoints.

You point pyqa at a document and an OpenAI-compatible inference server (vLLM,
TGI, Together, Groq, OpenRouter, Ollama, OpenAI itself). It parses the PDF,
reads the parts only a vision model can read, chunks the result semantically,
generates question-answer pairs, discards the ones that aren't supported by the
source, and writes JSONL to your disk.

Built for anyone finetuning a small model on documents they already have — a
research corpus, product manuals, internal policy, case files. No pyqa-owned
network calls, no credentials on the command line, no telemetry.

```bash
pyqa inspect  policy.pdf                        # what will this cost? (free)
pyqa chunk    policy.pdf                        # how does it chunk? (free)
pyqa run      policy.pdf -c pyqa.toml -o ./out  # build the dataset
```

## How it works

```
PDF ─▶ Parse ─▶ Vision ─▶ Assemble ─▶ Chunk ─▶ Generate ─▶ Ground ─▶ Write
       Rust      your VLM   page index  local     your LLM   your LLM  JSONL
       (free)    (as needed)  (free)    (free)
```

**Parse** uses [`pdf-inspector`](https://github.com/firecrawl/pdf-inspector), a
Rust library that returns per-page Markdown *and* tells you which pages actually
need OCR. Most pages of most documents parse natively and cost nothing.

**Vision** is the only stage that sends page images anywhere, and it runs at most
one call per page. A planner assigns each page exactly one task:

| Task | When | Cost |
|---|---|---|
| `ocr` | the page has no text layer | 1 call |
| `table_reread` | a table on a complex layout, where heuristics are unreliable | 1 call |
| `describe` | the page carries a chart or diagram | 1 call |
| `none` | the page parsed fine | free |

`ocr` subsumes the others, so a scanned page with a chart is never billed twice.

**Figure detection looks for vector drawings, not just embedded images.** Most
PDF flowcharts are drawn with vector paths and contain no image object at all, so
image-only detection misses exactly the diagrams that matter. Repeated images
(letterheads, logos) are fingerprinted and suppressed — otherwise a logo would
bill a call on every page.

Descriptions are spliced back **at the figure's position**, so the figure travels
into the same chunk as the prose introducing it:

````markdown
Applications are routed by score band as shown below.

```figure page=12
A five-stage flowchart: intake, scoring, review, decision, notification...
```
````

**Ground** re-checks every generated answer against its source chunk and drops
unsupported ones into `rejected.jsonl` with reasons, so you can audit yield
instead of trusting it.

## Install

```bash
uv sync
```

Requires Python 3.12+. No system dependencies — PDF parsing and rendering ship as
prebuilt wheels.

## Configure

Copy `pyqa.example.toml` and edit. Credentials come from the environment:

```toml
[generate.endpoint]
model    = "Qwen/Qwen2.5-14B-Instruct"
base_url = "http://localhost:8000/v1"
api_key  = "${PYQA_API_KEY}"
```

`${VAR}` is expanded at load time; a referenced variable that is not set is an
error immediately, rather than a confusing 401 an hour into a long run.

Check your endpoints answer before starting:

```bash
pyqa check-connection -c pyqa.toml
```

## Output

`out/dataset.jsonl` — chat/messages format, ingested directly by TRL, Axolotl,
Unsloth, and OpenAI finetuning:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"What is the maximum LTV for a second-lien HELOC?"},{"role":"assistant","content":"..."}]}
```

Alongside it: `rejected.jsonl` (dropped pairs with reasons) and `manifest.json`
(page counts, vision calls broken down by task, chunk count, yield, timings,
models used).

## Controlling cost

`pyqa inspect` projects the exact number of vision calls **before** you spend
anything, and it is also where you tune figure-detection thresholds against your
own documents:

```
Page  Vision task  Figures  Tables  Cols  Chars  Why
   1  none               -    -      -      847
   2  describe           1    -      -      612  1 vector figure region(s)
   3  ocr                -    -      -        0  page has no extractable text layer

Projected vision calls
  ocr               1
  describe          1
  total             2 of 3 pages
```

Then: `--no-figures`, `--no-table-reread`, `--max-vision-pages N` (which trims
optional work but never drops OCR), or `--no-ground`.

Vision results are cached against the document hash, so an interrupted run
resumes without paying for OCR twice.

## Use as a library

```python
import asyncio
from pyqa import Config, run_async

config = Config.load("pyqa.toml")
result = await run_async("policy.pdf", config, progress=print)
print(result.manifest["pairs"])
```

Every engine is a protocol you can replace — `Parser`, `VisionEngine`,
`Chunker`, `ChatClient` — so you can swap in a different parser or route
requests through your own client without forking.

## Notes

**Air-gapped environments.** Chunking uses a small local embedding model
(`minishlab/potion-base-32M`) downloaded once from HuggingFace. Pre-seed the HF
cache before going offline, or point `chunk.embedding_model` at a local path.

**Structured output.** OpenAI-compatible servers disagree about
`response_format`. pyqa negotiates once per run — `json_schema` → `json_object` →
prompt-only — and validates responses locally regardless. The mode it settled on
is recorded in the manifest.

**macOS + uv.** `uv` writes the editable-install `.pth` with the macOS hidden
flag, and CPython skips hidden `.pth` files, so `import pyqa` breaks after a
sync. `make sync` handles it; or run `chflags nohidden .venv/lib/python*/site-packages/*.pth`.

## Development

```bash
make test   # sync, then run the suite
make lint   # ruff check + format
```

The suite runs against a stub OpenAI-compatible server, so it needs no
credentials and no network.
