"""Третья часть практического задания: многопоточность и процессы."""

from .augment import OPERATIONS, augment_batch, augment_image
from .performance import ComparisonResult, compare
from .producer_consumer import ParallelPipeline, PipelineStats, run_sequential

__all__ = [
    "OPERATIONS",
    "ComparisonResult",
    "ParallelPipeline",
    "PipelineStats",
    "augment_batch",
    "augment_image",
    "compare",
    "run_sequential",
]
