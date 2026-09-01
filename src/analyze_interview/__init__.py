"""
analyze-interview: 本地化、零云端费用、隐私保护的面试录屏智能复盘系统。
基于 Apple Silicon MLX 硬件加速与 Whisper。
"""

__version__ = "0.1.0"
__author__ = "zos1n"
__email__ = "zos1n@outlook.com"

from analyze_interview.config import AnalyzerConfig, load_config
from analyze_interview.pipeline import InterviewAnalysisPipeline
from analyze_interview.state import ExecutionStage, ExecutionState

__all__ = [
    "AnalyzerConfig",
    "ExecutionStage",
    "ExecutionState",
    "InterviewAnalysisPipeline",
    "load_config",
]
