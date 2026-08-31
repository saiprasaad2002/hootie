# Brand assets

Everything here is generated. Edit `make_logo.py`, never the SVGs — the theme
variants are drawn from one `owl()` routine so the mascot cannot drift between
the README lockup, the favicon, and the avatar.

```bash
make logo
```

| File | Use |
|---|---|
| `avatar.svg` / `avatar.png` | GitHub profile and organisation avatar. **Upload the PNG** — GitHub rejects SVG for avatars. Framed for a circular crop. |
| `social-preview.png` | GitHub repository social preview (Settings → General → Social preview). 1280×640, as GitHub requires. |
| `icon-light.svg` / `icon-dark.svg` | Favicons and inline use. Transparent ground. |
| `logo-light.svg` / `logo-dark.svg` | Mascot plus wordmark, for the README header and docs. |

Light and dark variants exist because GitHub selects between them with
`<picture media="(prefers-color-scheme: dark)">`, which is more reliable than a
media query inside an SVG loaded through `<img>`. The avatar and social preview
are deliberately *not* themed: each is a single uploaded image, so both carry a
warm ground that stays readable on a light profile page and a dark repo header
alike.

The PNGs are rasterized with `cairosvg`, which does not resolve the SVG font
stack, so the wordmark renders lighter there than in the SVG lockup. That only
affects the social preview.
