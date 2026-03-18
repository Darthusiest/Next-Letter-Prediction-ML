"""
Strong corpus cleaning for OCR/boilerplate-heavy ebook conversions.

Reads .txt files from:
  - data/raw/*.txt
  - data/raw/nlp-ebooks/*.txt

Writes cleaned per-file outputs to:
  - data/processed/clean_texts/*.txt

Also writes:
  - data/processed/cleaning_report.json
  - data/processed/clean_texts/merged_corpus.txt
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EBOOK_DIR = RAW_DIR / "nlp-ebooks"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "clean_texts"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_report.json"


BOILERPLATE_PATTERNS = [
    r"\bcopyright\b",
    r"\ball rights reserved\b",
    r"\bprinted in\b",
    r"\bcatalog\b",
    r"\bcatalogue\b",
    r"\bcatalog card\b",
    r"\blibrary\b",
    r"\bdate due\b",
    r"\bproject gutenberg\b",
    r"\btranscrib",
    r"\bebook\b",
    r"\bscanned\b",
    r"\bpublication\b",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Long runs of layout punctuation or OCR separators
SEPARATOR_RE = re.compile(r"(\.{8,}|-{8,}|_{8,}|={6,}|\*{6,}|~{6,})")

# Whitespace normalization: collapse runs of spaces/tabs; preserve newlines
SPACE_TAB_RE = re.compile(r"[ \t]+")


NOISY_FILENAME_HINTS = [
    "golden comics",
    "comic",
    "illustrated",
    "golden press",
    "create space",
    "independent publishing",
]


@dataclass
class FileReport:
    path: str
    skipped: bool
    skip_reason: str | None
    input_chars: int
    output_chars: int
    input_lines: int
    kept_lines: int
    dropped_lines: int


def _digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    d = sum(ch.isdigit() for ch in s)
    return d / len(s)


def _letter_ratio(s: str) -> float:
    if not s:
        return 0.0
    a = sum(ch.isalpha() for ch in s)
    return a / len(s)


def _non_ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    na = sum(ord(ch) > 127 for ch in s)
    return na / len(s)


def should_skip_file(path: Path) -> tuple[bool, str | None]:
    name = path.name.lower()
    for hint in NOISY_FILENAME_HINTS:
        if hint in name:
            return True, f"filename_hint:{hint}"
    return False, None


def normalize_line(line: str) -> str:
    # normalize unicode quotes/dashes lightly without heavy dependencies
    line = line.replace("\u2019", "'").replace("\u2018", "'")
    line = line.replace("\u201c", '"').replace("\u201d", '"')
    line = line.replace("\u2014", "-").replace("\u2013", "-")
    line = SPACE_TAB_RE.sub(" ", line)
    return line.strip()


def is_noise_line(
    line: str,
    *,
    max_digit_ratio: float,
    min_letter_ratio: float,
    max_non_ascii_ratio: float,
) -> tuple[bool, str | None]:
    if not line:
        return True, "empty"
    if BOILERPLATE_RE.search(line):
        return True, "boilerplate_keyword"
    if SEPARATOR_RE.search(line):
        return True, "separator_run"

    dr = _digit_ratio(line)
    if dr > max_digit_ratio:
        return True, f"digit_ratio>{max_digit_ratio}"

    # Layout-like OCR artifacts often look like: many 1-2 character tokens
    # separated by spaces (e.g. "Z i a f g ie ...") with relatively high
    # whitespace ratio but no “real” word-like tokens (length >= 3).
    #
    # This aims to reduce residual uncertainty concentrated in
    # `whitespace_dense` / `newline_whitespace_dense` categories downstream.
    spaces = sum(1 for ch in line if ch == " ")
    whitespace_ratio = spaces / len(line) if line else 0.0
    tokens = line.split()
    if len(tokens) >= 8:
        short_token_count = sum(1 for t in tokens if len(t) <= 2)
        has_word_like_token = any(
            (len(t) >= 3) and any(ch.isalpha() for ch in t) for t in tokens
        )
        if whitespace_ratio >= 0.35 and (not has_word_like_token) and short_token_count >= 6:
            return True, "whitespace_dense_layout_ocr"

    lr = _letter_ratio(line)
    if lr < min_letter_ratio and len(line) >= 20:
        return True, f"letter_ratio<{min_letter_ratio}"

    nar = _non_ascii_ratio(line)
    if nar > max_non_ascii_ratio:
        return True, f"non_ascii_ratio>{max_non_ascii_ratio}"

    # Very short “lines” that are mostly punctuation
    if len(line) <= 3 and not any(ch.isalpha() for ch in line):
        return True, "tiny_nonalpha"

    return False, None


def clean_text(
    text: str,
    *,
    max_digit_ratio: float,
    min_letter_ratio: float,
    max_non_ascii_ratio: float,
) -> tuple[str, int, int]:
    kept = 0
    dropped = 0
    out_lines: list[str] = []

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        noisy, _reason = is_noise_line(
            line,
            max_digit_ratio=max_digit_ratio,
            min_letter_ratio=min_letter_ratio,
            max_non_ascii_ratio=max_non_ascii_ratio,
        )
        if noisy:
            dropped += 1
            continue
        kept += 1
        out_lines.append(line)

    # Keep paragraph breaks but collapse excessive blank lines
    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned, kept, dropped


def iter_input_txts() -> Iterable[Path]:
    for p in sorted(RAW_DIR.glob("*.txt")):
        yield p
    if EBOOK_DIR.exists():
        for p in sorted(EBOOK_DIR.glob("*.txt")):
            yield p


def slugify_filename(name: str) -> str:
    base = name
    base = base.replace(".txt", "")
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base[:120] or "file"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean OCR/boilerplate-heavy corpora.")
    parser.add_argument("--max-digit-ratio", type=float, default=0.30)
    parser.add_argument("--min-letter-ratio", type=float, default=0.25)
    parser.add_argument("--max-non-ascii-ratio", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--report-path", type=str, default=str(REPORT_PATH))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[FileReport] = []
    merged_parts: list[str] = []

    inputs = list(iter_input_txts())
    if not inputs:
        raise FileNotFoundError(
            f"No .txt files found in {RAW_DIR} or {EBOOK_DIR}. Convert PDFs/EPUBs first."
        )

    for path in inputs:
        skip, reason = should_skip_file(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        input_chars = len(text)
        input_lines = text.count("\n") + 1 if text else 0

        if skip:
            reports.append(
                FileReport(
                    path=str(path),
                    skipped=True,
                    skip_reason=reason,
                    input_chars=input_chars,
                    output_chars=0,
                    input_lines=input_lines,
                    kept_lines=0,
                    dropped_lines=input_lines,
                )
            )
            continue

        cleaned, kept, dropped = clean_text(
            text,
            max_digit_ratio=args.max_digit_ratio,
            min_letter_ratio=args.min_letter_ratio,
            max_non_ascii_ratio=args.max_non_ascii_ratio,
        )
        out_name = slugify_filename(path.name) + ".txt"
        out_path = out_dir / out_name
        out_path.write_text(cleaned, encoding="utf-8")
        merged_parts.append(cleaned)

        reports.append(
            FileReport(
                path=str(path),
                skipped=False,
                skip_reason=None,
                input_chars=input_chars,
                output_chars=len(cleaned),
                input_lines=input_lines,
                kept_lines=kept,
                dropped_lines=dropped,
            )
        )

    merged = "\n".join(merged_parts).strip() + "\n"
    (out_dir / "merged_corpus.txt").write_text(merged, encoding="utf-8")

    report_obj = {
        "inputs": [r.path for r in reports],
        "outputs_dir": str(out_dir),
        "total_input_chars": sum(r.input_chars for r in reports),
        "total_output_chars": sum(r.output_chars for r in reports),
        "skipped_files": [asdict(r) for r in reports if r.skipped],
        "file_reports": [asdict(r) for r in reports],
        "params": {
            "max_digit_ratio": args.max_digit_ratio,
            "min_letter_ratio": args.min_letter_ratio,
            "max_non_ascii_ratio": args.max_non_ascii_ratio,
            "noisy_filename_hints": NOISY_FILENAME_HINTS,
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
        },
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")

    print("Wrote cleaned texts to:", out_dir)
    print("Wrote cleaning report to:", report_path)
    print("Merged corpus:", out_dir / "merged_corpus.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

