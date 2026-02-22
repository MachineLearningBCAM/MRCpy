r"""
Minimax Generalized Cross-Entropy (MGCE) module.

This module implements the Minimax Generalized Cross-Entropy (MGCE)
classification framework for PyTorch-based deep learning. MGCE
generalizes the standard cross-entropy (log-loss) through the
:math:`\alpha`-loss family (:math:`\ell_\alpha`), a parametric family of loss
functions indexed by :math:`\alpha \geq 1` that interpolate between
the 0-1 loss (:math:`\alpha = 1`) and the log-loss
(:math:`\alpha \to \infty`). The minimax formulation provides robust
performance guarantees over an uncertainty set of distributions.

The module provides:

- :class:`mgce_loss`: :math:`\alpha`-loss function with gradient computation
  for the MGCE framework.
- :class:`mgce_clf`: MGCE classifier wrapping a PyTorch neural network
  with :math:`\alpha`-loss based minimax risk training.
"""

from MRCpy.pytorch.mgce.classifier import mgce_clf
from MRCpy.pytorch.mgce.loss import mgce_loss

__all__ = ['mgce_clf', 'mgce_loss']
