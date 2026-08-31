"""Generate the hootie logo, icon, and avatar assets.

Every variant is drawn from one `owl()` routine so the mascot cannot drift
between the README lockup, the favicon, and the GitHub avatar. Two theme
variants are written for each, because GitHub picks between them with
<picture media="(prefers-color-scheme: dark)">, which is more reliable than a
media query inside an SVG loaded via <img>.

Cuteness here is deliberate geometry, not decoration: one round body, eyes
large relative to the head, pupils converged slightly inward, and a small beak.
Those are the proportions that read as friendly, and they survive being scaled
down to a 20px favicon, which fussier detail would not.
"""

from __future__ import annotations

from pathlib import Path

FONT = "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Helvetica,Arial,sans-serif"

THEMES = {
    #          body       belly      beak/feet  pupil      wordmark   tagline
    "light": ("#4F46E5", "#E0E7FF", "#F59E0B", "#1E1B4B", "#0F172A", "#475569"),
    "dark": ("#818CF8", "#EEF2FF", "#FBBF24", "#1E1B4B", "#E2E8F0", "#94A3B8"),
}


def owl(
    body: str, belly: str, beak: str, pupil: str, x: float = 0, y: float = 0, s: float = 1
) -> str:
    """The mascot, drawn in a 64x64 box before transform."""
    return f"""  <g transform="translate({x} {y}) scale({s})">
    <!-- ear tufts, drawn under the body so they read as part of the silhouette -->
    <path d="M13 24 C12 16 14 11 18 9 C22 12 25 16 26 21 Z" fill="{body}"/>
    <path d="M51 24 C52 16 50 11 46 9 C42 12 39 16 38 21 Z" fill="{body}"/>
    <!-- body: one rounded blob, which is what makes an owl an owl -->
    <path d="M32 9 C46 9 56 20 56 34 C56 48 45 58 32 58 C19 58 8 48 8 34 C8 20 18 9 32 9 Z"
          fill="{body}"/>
    <!-- belly -->
    <ellipse cx="32" cy="45" rx="13" ry="12" fill="{belly}"/>
    <!-- eye discs: large, and overlapping in the middle -->
    <circle cx="23" cy="30" r="11" fill="#FFFFFF"/>
    <circle cx="41" cy="30" r="11" fill="#FFFFFF"/>
    <!-- pupils, converged slightly inward -->
    <circle cx="25.5" cy="31" r="5.4" fill="{pupil}"/>
    <circle cx="38.5" cy="31" r="5.4" fill="{pupil}"/>
    <circle cx="27.4" cy="29" r="1.9" fill="#FFFFFF"/>
    <circle cx="40.4" cy="29" r="1.9" fill="#FFFFFF"/>
    <!-- beak -->
    <path d="M32 36 L27.5 41.5 L36.5 41.5 Z" fill="{beak}" stroke-linejoin="round"/>
    <!-- feet -->
    <path d="M24 57 l0 4 M21.5 61 h5 M40 57 l0 4 M37.5 61 h5"
          stroke="{beak}" stroke-width="2.6" stroke-linecap="round" fill="none"/>
  </g>"""


AVATAR_GROUND = "#FDE9A9"  # warm cream: contrasts on white and on dark alike


def avatar() -> str:
    """Square, single fixed variant — GitHub avatars are one uploaded image.

    Deliberately not themed. The ground is warm rather than another indigo so
    the owl still reads once GitHub crops it to a circle, on a light profile
    page or a dark repository header.
    """
    body, belly, beak, pupil, _w, _t = THEMES["light"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128" role="img" aria-label="hootie">
  <title>hootie</title>
  <rect width="128" height="128" rx="30" fill="{AVATAR_GROUND}"/>
{owl(body, belly, beak, pupil, x=14, y=15, s=1.55)}
</svg>
"""


def icon(theme: str) -> str:
    """Transparent ground, for favicons and inline use."""
    body, belly, beak, pupil, _w, _t = THEMES[theme]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 68" width="64" height="68" role="img" aria-label="hootie">
  <title>hootie</title>
{owl(body, belly, beak, pupil, x=0, y=2)}
</svg>
"""


def lockup(theme: str) -> str:
    body, belly, beak, pupil, word, tag = THEMES[theme]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 84" width="320" height="84" role="img" aria-label="hootie">
  <title>hootie</title>
{owl(body, belly, beak, pupil, x=4, y=8, s=1.05)}
  <text x="86" y="47" font-family="{FONT}" font-size="38" font-weight="700"
        letter-spacing="-1.2" fill="{word}">hootie</text>
  <text x="87" y="65" font-family="{FONT}" font-size="12.5" font-weight="500"
        letter-spacing="0.2" fill="{tag}">PDFs into finetuning data</text>
</svg>
"""


def social() -> str:
    """1280x640 card. GitHub renders this when the repo is linked anywhere."""
    body, belly, beak, pupil, word, tag = THEMES["light"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640" role="img" aria-label="hootie">
  <title>hootie</title>
  <rect width="1280" height="640" fill="{AVATAR_GROUND}"/>
  <rect x="40" y="40" width="1200" height="560" rx="36" fill="#FFFFFF"/>
{owl(body, belly, beak, pupil, x=493, y=92, s=4.6)}
  <text x="640" y="500" text-anchor="middle" font-family="{FONT}" font-size="72"
        font-weight="700" letter-spacing="-2.4" fill="{word}">hootie</text>
  <text x="640" y="548" text-anchor="middle" font-family="{FONT}" font-size="26"
        font-weight="500" fill="{tag}">PDFs into finetuning data</text>
</svg>
"""


def main() -> None:
    here = Path(__file__).parent
    for theme in THEMES:
        (here / f"logo-{theme}.svg").write_text(lockup(theme), encoding="utf-8")
        (here / f"icon-{theme}.svg").write_text(icon(theme), encoding="utf-8")

    (here / "avatar.svg").write_text(avatar(), encoding="utf-8")
    (here / "social-preview.svg").write_text(social(), encoding="utf-8")
    # GitHub rejects SVG for avatars and social previews, so raster those two.
    try:
        import cairosvg
    except ImportError:
        print("  (cairosvg not installed; skipping PNG export — run `make sync`)")
    else:
        for name, size in (("avatar", (512, 512)), ("social-preview", (1280, 640))):
            cairosvg.svg2png(
                url=str(here / f"{name}.svg"),
                write_to=str(here / f"{name}.png"),
                output_width=size[0],
                output_height=size[1],
            )

    for path in sorted(list(here.glob("*.svg")) + list(here.glob("*.png"))):
        print(f"  wrote {path.relative_to(here.parent)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
