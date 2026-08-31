"""Render PDF pages to PNG for the vision model.

`pdf-inspector` does not rasterize, so this fills the gap. Pages are rendered one
at a time and released immediately: a 400-page scan at 200 DPI held in memory all
at once would be gigabytes.

Rendering is serialized behind a lock. PDFium state is not safe to drive from
several threads at once, and serializing costs nothing here — rendering is fast
relative to the network call that follows, and the parallelism worth having is
on the requests, not the pixels.
"""

from __future__ import annotations

import contextlib
import io
import threading
from pathlib import Path

import pypdfium2 as pdfium

from ..errors import VisionError


class PageRenderer:
    """Renders 1-indexed pages of one PDF to PNG bytes."""

    def __init__(self, path: Path, dpi: int = 200, max_edge: int = 2000) -> None:
        self.path = Path(path)
        self.dpi = dpi
        self.max_edge = max_edge
        self._lock = threading.Lock()
        try:
            self._pdf = pdfium.PdfDocument(str(self.path))
        except Exception as exc:
            raise VisionError(f"could not open {self.path} for rendering: {exc}") from exc

    def __len__(self) -> int:
        return len(self._pdf)

    def render(self, page: int) -> bytes:
        """Render a 1-indexed page to PNG bytes."""
        if page < 1 or page > len(self._pdf):
            raise VisionError(f"page {page} is out of range for {self.path}")

        with self._lock:
            try:
                bitmap = self._pdf[page - 1].render(scale=self.dpi / 72)
                image = bitmap.to_pil()
            except Exception as exc:
                raise VisionError(f"failed to render page {page}: {exc}") from exc

        try:
            image = _downscale(image, self.max_edge)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
        finally:
            image.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # best-effort cleanup
            self._pdf.close()

    def __enter__(self) -> PageRenderer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _downscale(image, max_edge: int):
    """Shrink to fit `max_edge`, preserving aspect ratio.

    A full-resolution 200 DPI page is ~1700x2200; base64-encoded that is a large
    payload to send per page, and vision models downscale it anyway.
    """
    from PIL import Image

    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)
