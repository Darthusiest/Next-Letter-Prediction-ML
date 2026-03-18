"""
Analysis package for post-training diagnostics.

Exports:
- AnalysisConfig: configuration for which analyses to run.
- run_post_training_analysis: orchestrates all analysis modules for a run.
"""

from .config import AnalysisConfig
from .run_analysis import run_post_training_analysis

__all__ = ["AnalysisConfig", "run_post_training_analysis"]

