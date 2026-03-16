from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class AnalysisConfig:
    run_id: str
    output_root: Path = Path("outputs")

    # Core v1 plots
    generate_training_plots: bool = True
    generate_data_plots: bool = True
    generate_prediction_plots: bool = True
    generate_confusion_plots: bool = True
    generate_embedding_plots: bool = True
    show_plots: bool = False

    # Extended diagnostics
    generate_entropy_plots: bool = False
    generate_context_viz: bool = False
    generate_similarity_report: bool = False
    generate_transition_graph: bool = False

    # Model / prediction settings
    sample_contexts: List[str] = field(
        default_factory=lambda: ["th", "qu", "pre", "walki", "tion", "re", "un", "ing"]
    )
    top_k: int = 10

    # Entropy/confidence
    max_entropy_samples: int = 2000

    # Context viz and similarity
    context_viz_contexts: List[str] = field(
        default_factory=lambda: [
            "th",
            "sh",
            "ch",
            "qu",
            "ing",
            "ed",
            "tion",
            "pre",
            "re",
            "un",
            "walk",
            "play",
            "ness",
            "ly",
        ]
    )
    similarity_chars: List[str] = field(
        default_factory=lambda: ["e", "a", "t", " ", ".", ","]
    )
    similarity_contexts: List[str] = field(
        default_factory=lambda: ["th", "ing", "re", "un", "tion"]
    )

    # Attention (for transformers)
    attention_contexts: List[str] = field(
        default_factory=lambda: ["the", "question", "walking", "preparation", "running"]
    )

    # Advanced options
    enable_3d: bool = False
    enable_attention_plots: bool = True

