# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-01

Initial release.

### Added

- Seven-stage pipeline turning a PDF into a chat/messages JSONL dataset:
  parse, vision, assemble, chunk, generate, ground, write.
- Native parsing via `pdf-inspector`, which reports a per-page OCR verdict, so
  only pages that genuinely need a model are sent to one.
- A vision planner assigning **at most one task per page** (`ocr`,
  `table_reread`, `describe`, `none`), so no page is rendered or billed twice.
- Figure detection covering both raster images and dense vector-path clusters,
  since most PDF diagrams are drawn with path operators and contain no image
  object at all. Repeated images are fingerprinted and suppressed so page
  furniture such as a letterhead does not bill a call on every page.
- Position-aware splicing of figure descriptions, so a figure chunks together
  with the prose that introduces it.
- A grounding pass that drops answers unsupported by their source chunk into
  `rejected.jsonl` with reasons.
- Structured-output negotiation across OpenAI-compatible servers
  (`json_schema` → `json_object` → prompt-only), with local validation
  regardless of which path succeeded.
- Vision-result caching keyed on the document hash, so an interrupted run
  resumes without paying for OCR twice.
- CLI: `inspect`, `chunk`, `check-connection`, `run`. The first two need no
  credentials, so detection thresholds can be tuned before spending anything.
- Swappable protocols for `Parser`, `VisionEngine`, `Chunker`, and `ChatClient`.
- `.env` support via `python-dotenv`, and an OpenRouter example config.

[Unreleased]: https://github.com/saiprasaad2002/hootie/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/saiprasaad2002/hootie/releases/tag/v0.1.0
