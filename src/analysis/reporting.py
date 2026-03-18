"""
Summary reporting for post-training analysis.
Writes a short text file describing key metrics and generated artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def write_summary_report(
    metrics: Dict,
    generated_files: List[Path],
    skipped_reasons: List[str],
    reports_dir: Path,
) -> Path:
    """
    Write a human-readable summary_report.txt for a given run, including:
    - metrics from metrics.json (if present)
    - list of generated plot/report files
    - reasons why any analyses were skipped
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "summary_report.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write("Post-training analysis summary\n")
        f.write("================================\n\n")

        if metrics:
            f.write("Metrics (from metrics.json):\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if generated_files:
            f.write("Generated files:\n")
            for p in generated_files:
                f.write(f"  {p}\n")
            f.write("\n")

        if skipped_reasons:
            f.write("Skipped analyses:\n")
            for reason in skipped_reasons:
                f.write(f"  - {reason}\n")

    return path

