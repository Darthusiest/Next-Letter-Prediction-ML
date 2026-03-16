"""
Utilities to convert PDF and EPUB files to plain text (.txt) so they can be
used by the character-level next-letter model.

These helpers are **optional** and keep heavy dependencies out of the main
requirements. To use them, install the libraries you need:

PDF:
    pip install pypdf

EPUB:
    pip install ebooklib

Then run from the project root, for example:

    python tools/convert_to_txt.py input.pdf data/raw/input_from_pdf.txt
    python tools/convert_to_txt.py input.epub data/raw/input_from_epub.txt
"""

import sys
from pathlib import Path
from typing import Optional


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf (if installed)."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Install it with `pip install pypdf` to convert PDFs."
        ) from exc

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def epub_to_text(epub_path: Path) -> str:
    """Extract text from an EPUB using ebooklib (if installed)."""
    try:
        import ebooklib  # type: ignore
        from ebooklib import epub  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ebooklib and beautifulsoup4 are required. Install with "
            "`pip install ebooklib beautifulsoup4` to convert EPUBs."
        ) from exc

    book = epub.read_epub(str(epub_path))
    parts = []
    # Use isinstance checks instead of ITEM_DOCUMENT constant to be robust
    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            html = item.get_content()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            parts.append(text)
    return "\n\n".join(parts)


def convert_to_txt(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert a .pdf or .epub file to .txt.
    Returns the path to the written .txt file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".txt")
    output_path = Path(output_path)
    ext = input_path.suffix.lower()

    if ext == ".pdf":
        text = pdf_to_text(input_path)
    elif ext == ".epub":
        text = epub_to_text(input_path)
    else:
        raise ValueError(f"Unsupported input extension: {ext}. Use .pdf or .epub.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main(argv: list[str]) -> None:
    if len(argv) < 2 or len(argv) > 3:
        print(
            "Usage: python tools/convert_to_txt.py INPUT.(pdf|epub) [OUTPUT.txt]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    input_path = Path(argv[1])
    output_path = Path(argv[2]) if len(argv) == 3 else None
    out = convert_to_txt(input_path, output_path)
    print(f"Wrote text to {out}")


if __name__ == "__main__":
    main(sys.argv)

