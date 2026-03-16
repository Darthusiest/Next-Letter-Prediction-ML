"""
Batch convert all PDFs in data/raw/nlp-ebooks/ to individual .txt files in data/raw/.

This is a safer alternative to ad-hoc shell loops and ensures each book gets
its own .txt file (e.g. moby_dick.txt, frankenstein.txt, treasure_island.txt).
"""

from pathlib import Path

from convert_to_txt import convert_to_txt


SOURCE_DIR = Path("data/raw/nlp-ebooks")
TARGET_DIR = Path("data/raw")


def slugify(name: str) -> str:
    # Simple filename normalization: keep alnum, space, underscore, hyphen
    cleaned = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    return cleaned.strip().lower().replace(" ", "_")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in SOURCE_DIR.glob("*.pdf"):
        base = slugify(pdf.stem)
        out_path = TARGET_DIR / f"{base}.txt"
        print(f"Converting {pdf} -> {out_path}")
        convert_to_txt(pdf, out_path)


if __name__ == "__main__":
    main()

