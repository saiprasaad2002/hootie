"""Generate deterministic PDF fixtures by hand-writing PDF syntax.

Hand-written rather than produced by a rendering library so the fixtures contain
exactly the object types the figure detector looks for: real vector path objects
(a drawn flowchart, which has no image object at all) and real raster XObjects,
including a logo repeated on every page to exercise boilerplate suppression.
"""

from __future__ import annotations

import zlib
from pathlib import Path

WIDTH, HEIGHT = 612, 792


def _stream_obj(data: bytes, extra: str = "") -> bytes:
    comp = zlib.compress(data)
    return (
        f"<< /Length {len(comp)} /Filter /FlateDecode {extra} >>\nstream\n".encode()
        + comp
        + b"\nendstream"
    )


def _rgb_image(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    raw = bytes(rgb) * (w * h)
    return _stream_obj(
        raw,
        f"/Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8",
    )


def _flowchart_ops() -> str:
    """A vector 'flowchart': boxes and connectors, drawn with path operators."""
    ops = ["1 w 0 0 0 RG"]
    y = 620
    for _ in range(5):
        ops.append(f"200 {y} 200 60 re S")  # box
        ops.append(f"300 {y} m 300 {y - 30} l S")  # connector
        ops.append(f"280 {y - 10} m 320 {y - 10} l S")  # arrowhead
        y -= 110
    for i in range(6):  # branch arms, to push the path count up
        ops.append(f"150 {480 - i * 40} m 450 {500 - i * 40} l S")
    return "\n".join(ops)


def build(path: Path) -> None:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # 1-based object number

    logo = add(_rgb_image(8, 8, (200, 30, 30)))  # small, repeated on every page
    chart = add(_rgb_image(64, 48, (30, 90, 200)))  # large raster chart, page 3 only
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pages_id = len(objects) + 1  # reserved; filled in after page objects exist
    add(b"PLACEHOLDER")

    page_ids: list[int] = []
    specs = [
        ("Page One Alpha. Credit policy overview.", "", False),
        ("Page Two Bravo. Maximum LTV is 80 percent.", _flowchart_ops(), False),
        ("Page Three Charlie. See the chart below.", "", True),
        ("Page Four Delta. Underwriting rules conclude.", "", False),
    ]

    for text, extra_ops, with_chart in specs:
        ops = [
            "q 1 0 0 1 40 750 cm 20 0 0 20 0 0 cm /ImLogo Do Q",  # header logo
            f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET",
        ]
        if extra_ops:
            ops.append(extra_ops)
        if with_chart:
            ops.append("q 1 0 0 1 150 300 cm 300 0 0 220 0 0 cm /ImChart Do Q")

        content = add(_stream_obj("\n".join(ops).encode()))
        xobjects = f"/ImLogo {logo} 0 R" + (f" /ImChart {chart} 0 R" if with_chart else "")
        page = add(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {WIDTH} {HEIGHT}] "
            f"/Resources << /Font << /F1 {font} 0 R >> /XObject << {xobjects} >> >> "
            f"/Contents {content} 0 R >>".encode()
        )
        page_ids.append(page)

    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode()
    catalog = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode()
    )

    path.write_bytes(bytes(out))


if __name__ == "__main__":
    target = Path(__file__).parent / "figures_4page.pdf"
    build(target)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
