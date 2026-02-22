"""
PyTorch-based MRC implementations.

This module provides PyTorch implementations of Minimax Risk Classifiers
with GPU acceleration and deep neural network support.
"""

from MRCpy.pytorch.mgce.classifier import mgce_clf
from MRCpy.pytorch.mgce.loss import mgce_loss

__all__ = ['mgce_clf', 'mgce_loss']
