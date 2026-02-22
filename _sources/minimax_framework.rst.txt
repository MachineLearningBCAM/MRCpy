Minimax Framework
=================

MRCpy [8]_ implements minimax risk classifiers (MRCs) that obtain
classification rules by minimizing the worst-case expected loss over an
uncertainty set of distributions. This section provides a general overview
of the framework underlying all methods in the library, following the
formulation in [9]_.

Minimax Risk Classification
---------------------------

Given a loss function :math:`\ell`, an MRC obtains a classification rule
:math:`\mathrm{h}^{\mathcal{U}}` that solves:

.. math::

    \mathrm{h}^{\mathcal{U}} \in \arg\min_{\mathrm{h}}
    \max_{\mathrm{p} \in \mathcal{U}} \ell(\mathrm{h}, \mathrm{p})

where the maximization is over distributions :math:`\mathrm{p}` in an
uncertainty set :math:`\mathcal{U}` defined by expectation constraints on
feature mappings :math:`\Phi(x, y)`.

Uncertainty Sets
----------------

MRCpy implements two types of uncertainty sets:

* **Uncertainty set** :math:`\mathcal{U}_1`: Defined by constraints that
  bound the expectations of a feature mapping :math:`\Phi(x, y)`:

  .. math::

      \mathcal{U}_1 = \left\{ \mathrm{p} :
      \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)] - \boldsymbol{\tau} \right|
      \leq \boldsymbol{\lambda} \right\}

  MRCs using :math:`\mathcal{U}_1` (the ``MRC`` class) provide upper and
  lower bounds on the expected loss [3]_. Efficient learning algorithms
  for high-dimensional settings are available via column generation [4]_
  and constraint generation [2]_.

* **Uncertainty set** :math:`\mathcal{U}_2`: Adds an additional constraint
  that fixes the instances' marginal distribution to coincide with the
  empirical marginal:

  .. math::

      \mathcal{U}_2 = \left\{ \mathrm{p} :
      \mathrm{p}_x = \hat{\mathrm{p}}_x, \;
      \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)] - \boldsymbol{\tau} \right|
      \leq \boldsymbol{\lambda} \right\}

  MRCs using :math:`\mathcal{U}_2` (the ``CMRC`` class) correspond to
  popular techniques such as L1-regularized logistic regression, zero-one
  adversarial classifiers, and maximum entropy machines [6]_.

In both cases, :math:`\boldsymbol{\tau}` denotes the empirical mean estimates of the
feature mappings and :math:`\boldsymbol{\lambda}` controls the size of the uncertainty
set based on the estimation accuracy.

The library also provides methods for adaptation to concept drift [7]_ and
covariate shift [5]_ within the minimax risk classification framework.

Feature Mappings
----------------

MRCs use feature mappings to represent instance-label pairs as real vectors.
MRCpy implements several feature mappings:

* **Linear (identity)**: Direct use of input features.
* **Random Fourier features**: Approximation of kernel methods via random projections.
* **Random ReLU features**: Non-linear random features using ReLU activations.
* **Threshold features**: Binary features based on thresholding input dimensions.

All feature mappings can be combined with any MRC variant.

Loss Functions
--------------

MRCpy supports two main types of loss functions:

* **0-1 loss**: Directly quantifies the probability of classification error.
  Unlike common techniques that rely on surrogate losses, MRCs can utilize
  the 0-1 loss directly.
* **Log-loss**: Quantifies the negative log-likelihood for a classification
  rule (cross-entropy).

The PyTorch MGCE module extends this framework by supporting the full
alpha-loss family (:math:`\ell_\alpha`) [1]_, which smoothly interpolates
between 0-1 loss (:math:`\alpha = 1`) and log-loss
(:math:`\alpha \to \infty`) through the parameter :math:`\alpha`.
See `mgce_clf` in the API reference for details.

The library implements multiple techniques within the minimax risk
classification framework; see the references below for details.

References
----------

.. [1] Bondugula, K., Mazuelas, S., Pérez, A., & Liu, A. (2026). Minimax
       Generalized Cross-Entropy. In Proceedings of the International
       Conference on Artificial Intelligence and Statistics (AISTATS).

.. [2] Bondugula, K., Mazuelas, S., & Pérez, A. (2025). Efficient
       Large-Scale Learning of Minimax Risk Classifiers. IEEE International
       Conference on Data Mining (ICDM).

.. [3] Mazuelas, S., Romero, M., & Grunwald, P. (2023). Minimax Risk
       Classifiers with 0-1 Loss. Journal of Machine Learning Research,
       24(208), 1-48.

.. [4] Bondugula, K., Mazuelas, S., & Pérez, A. (2023). Efficient Learning
       of Minimax Risk Classifiers in High Dimensions. The 39th Conference
       on Uncertainty in Artificial Intelligence (UAI), 206-215.

.. [5] Segovia-Martín, J.I., Mazuelas, S., & Liu, A. (2023).
       Double-Weighting for Covariate Shift Adaptation. In Proceedings of
       the 40th International Conference on Machine Learning (ICML),
       pp. 30439-30457.

.. [6] Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized Maximum
       Entropy for Supervised Classification. IEEE Transactions on
       Information Theory, 68(4), 2530-2550.

.. [7] Álvarez, V., Mazuelas, S., & Lozano, J.A. (2022). Minimax
       Classification under Concept Drift with Multidimensional Adaptation
       and Performance Guarantees. In Proceedings of the 39th International
       Conference on Machine Learning (ICML), pp. 486-499.

.. [8] Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library
       for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952.

.. [9] Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification
       with 0-1 Loss and Performance Guarantees. Advances in Neural
       Information Processing Systems, 33, 302-312.
