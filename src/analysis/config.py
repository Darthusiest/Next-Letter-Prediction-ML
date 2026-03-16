from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AnalysisConfig:
    run_id: str
    output_root: Path = Path("outputs")

    generate_training_plots: bool = True
    generate_data_plots: bool = True
    generate_prediction_plots: bool = True
    generate_confusion_plots: bool = True
    generate_embedding_plots: bool = True
    show_plots: bool = False

    # Model / prediction settings
    sample_contexts: List[str] = field(
        default_factory=lambda: ["th", "qu", "pre", "walki", "tion", "re", "un", "ing"]
    )
    top_k: int = 10

    # Advanced options (for later versions)
    enable_3d: bool = False
    enable_attention_plots: bool = True


