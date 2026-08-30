"""Generate the pyqa logo assets.

Two theme variants are generated from one template so they cannot drift. GitHub
picks between them with <picture media="(prefers-color-scheme: dark)">, which is
more reliable than relying on a media query inside an SVG loaded via <img>.

The mark is a document with a conversation bubble emerging from it: a page
becoming question-and-answer pairs, which is the whole product in one glyph.
"""

from __future__ import annotations

from pathlib import Path

# Font stack rather than embedded outlines: the wordmark is simple enough that
# small metric differences between platforms are not noticeable.
FONT = "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Helvetica,Arial,sans-serif"

THEMES = {
    # name:   (accent,   wordmark,  page outline, page rules)
    "light": ("#4F46E5", "#0F172A", "#475569", "#CBD5E1"),
    "dark": ("#818CF8", "#E2E8F0", "#94A3B8", "#475569"),
}


def mark(accent: str, outline: str, rules: str, x: float = 0, y: float = 0, s: float = 1) -> str:
    """The icon: a page with a conversation bubble overlapping its lower right."""
    return f"""  <g transform="translate({x} {y}) scale({s})">
    <!-- page, with a folded top-right corner -->
    <path d="M6 4.5h20.5L38 16v25.5a4 4 0 0 1-4 4H10a4 4 0 0 1-4-4V8.5a4 4 0 0 1 4-4z"
          fill="none" stroke="{outline}" stroke-width="3" stroke-linejoin="round"/>
    <path d="M26 4.5V13a3 3 0 0 0 3 3h9" fill="none" stroke="{outline}"
          stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    <!-- lines of text on the page -->
    <g stroke="{rules}" stroke-width="3" stroke-linecap="round">
      <path d="M13 24h13"/>
      <path d="M13 31h8"/>
    </g>
    <!-- conversation bubble: the page turning into question-and-answer pairs -->
    <path d="M32 27h20a5 5 0 0 1 5 5v13a5 5 0 0 1-5 5H43l-7.5 6.5V50h-3.5a5 5 0 0 1-5-5V32a5 5 0 0 1 5-5z"
          fill="{accent}"/>
    <g stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.95">
      <path d="M36 36h13"/>
      <path d="M36 43h8"/>
    </g>
  </g>"""


def lockup(theme: str) -> str:
    accent, word, outline, rules = THEMES[theme]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 76" width="300" height="76" role="img" aria-label="pyqa">
  <title>pyqa</title>
{mark(accent, outline, rules, x=2, y=6)}
  <text x="82" y="44" font-family="{FONT}" font-size="38" font-weight="700"
        letter-spacing="-1.2" fill="{word}">pyqa</text>
  <text x="83" y="62" font-family="{FONT}" font-size="12.5" font-weight="500"
        letter-spacing="0.2" fill="{outline}">PDFs into finetuning data</text>
</svg>
"""


def icon(theme: str) -> str:
    accent, _word, outline, rules = THEMES[theme]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" aria-label="pyqa">
  <title>pyqa</title>
{mark(accent, outline, rules, x=1, y=1)}
</svg>
"""


def main() -> None:
    here = Path(__file__).parent
    for theme in THEMES:
        (here / f"logo-{theme}.svg").write_text(lockup(theme), encoding="utf-8")
        (here / f"icon-{theme}.svg").write_text(icon(theme), encoding="utf-8")
    for path in sorted(here.glob("*.svg")):
        print(f"wrote {path.relative_to(here.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
