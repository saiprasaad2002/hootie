"""Prompts and JSON schemas for QA generation and grounding.

Schemas are hand-written rather than derived from Pydantic because strict
`json_schema` mode rejects `$ref`/`$defs`, which `model_json_schema()` emits for
any nested model.
"""

from __future__ import annotations

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["supported", "reason"],
    "additionalProperties": False,
}

QA_SYSTEM = """\
You write question-answer pairs for finetuning a language model on a specific document.

The pairs become training data, so quality matters more than quantity.

Rules:
- Every answer must be fully supported by the passage. Never use outside knowledge.
- Questions must be self-contained. A reader who has not seen the passage must be
  able to understand what is being asked, so avoid "this document", "the above",
  "as mentioned", or bare pronouns referring to the passage.
- Prefer questions a real user would ask: specifics, thresholds, definitions,
  conditions, exceptions, and procedures.
- Answers should be complete but not padded. Include the specific figures, names,
  and conditions stated in the passage.
- Vary the form: some factual lookups, some "under what conditions", some
  procedural, some asking to explain a defined term.
- If the passage contains a ```figure block, treat its description as part of the
  document and write questions about what the figure shows.
- If the passage is too thin to support the requested number of pairs, return
  fewer. Never invent content to reach a count."""

GROUNDING_SYSTEM = """\
You check whether an answer is supported by a source passage.

Given a passage, a question, and an answer, decide whether every factual claim in
the answer is stated in or directly derivable from the passage.

Mark it unsupported when the answer:
- states facts, figures, or names absent from the passage,
- contradicts the passage,
- or answers from general knowledge instead of the passage.

Minor rephrasing, summarising, and reasonable inference from what is stated are
supported. Be strict about invented specifics and lenient about wording.

Give a one-sentence reason."""


def qa_user_prompt(passage: str, pairs: int, page_label: str) -> str:
    return (
        f"Write up to {pairs} question-answer pairs from this passage "
        f"({page_label} of the source document).\n\n---\n{passage}\n---"
    )


def grounding_user_prompt(passage: str, question: str, answer: str) -> str:
    return (
        f"Passage:\n---\n{passage}\n---\n\n"
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        "Is the answer fully supported by the passage?"
    )
