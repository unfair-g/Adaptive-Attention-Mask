"""Semantic segmentation learners for continual learning."""
from .base_seg import BaseSegmentationLearner
from .er_seg import ERSegmentationLearner

__all__ = [
    'BaseSegmentationLearner',
    'ERSegmentationLearner',
]

