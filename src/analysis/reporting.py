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
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "summary_report.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write("Post-training analysis summary\n")
        f.write("================================\n\n")
\n        if metrics:\n            f.write(\"Metrics (from metrics.json):\\n\")\n            for k, v in metrics.items():\n                f.write(f\"  {k}: {v}\\n\")\n            f.write(\"\\n\")\n\n        if generated_files:\n            f.write(\"Generated files:\\n\")\n            for p in generated_files:\n                f.write(f\"  {p}\\n\")\n            f.write(\"\\n\")\n\n        if skipped_reasons:\n            f.write(\"Skipped analyses:\\n\")\n            for reason in skipped_reasons:\n                f.write(f\"  - {reason}\\n\")\n\n    return path\n+\n*** End Patch```} ***!
