<!-- ---
layout: default
title: MRCpy
nav_exclude: true
---

# MRCpy: A Library for Minimax Risk Classifiers

MRCpy is an open-source Python library for Supervised Classification in Machine Learning. It provides implementation for recently proposed Minimax Risk Classifiers (MRCs) based on robust risk minimization and 0-1-loss, and provide bounds on the classification error. It provides a unified user-friendly interface for different variant of MRCs. The library also comes with a feature mapping module that provides implementation for some recent feature mapping methods.

## Documentation outline

- [Getting Started](getting_started.html)
	- [Installation and Setup](#installation-setup)
	- [Quick Start](#start)

- User Guide
	- What is Minimax Risk Classification
	- Variants of MRCs
	- Feature Mappings

- Examples
	- MRC
	- Feature Mappings

 -->

metric-learn: Metric Learning in Python
=======================================
|Travis-CI Build Status| |License| |PyPI version| |Code coverage|

`metric-learn <https://github.com/scikit-learn-contrib/metric-learn>`_
contains efficient Python implementations of several popular supervised and
weakly-supervised metric learning algorithms. As part of `scikit-learn-contrib
<https://github.com/scikit-learn-contrib>`_, the API of metric-learn is compatible with `scikit-learn
<https://scikit-learn.org/>`_, the leading library for machine learning in
Python. This allows to use all the scikit-learn routines (for pipelining,
model selection, etc) with metric learning algorithms through a unified
interface.

If you use metric-learn in a scientific publication, we would appreciate
citations to the following paper:

`metric-learn: Metric Learning Algorithms in Python
<http://www.jmlr.org/papers/volume21/19-678/19-678.pdf>`_, de Vazelhes
*et al.*, Journal of Machine Learning Research, 21(138):1-6, 2020.

Bibtex entry::

  @article{metric-learn,
    title = {metric-learn: {M}etric {L}earning {A}lgorithms in {P}ython},
    author = {{de Vazelhes}, William and {Carey}, CJ and {Tang}, Yuan and
              {Vauquier}, Nathalie and {Bellet}, Aur{\'e}lien},
    journal = {Journal of Machine Learning Research},
    year = {2020},
    volume = {21},
    number = {138},
    pages = {1--6}
  }


Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   user_guide

.. toctree::
   :maxdepth: 2

   Package Contents <metric_learn>

.. toctree::
   :maxdepth: 2

   auto_examples/index

:ref:`genindex` | :ref:`search`

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/scikit-learn-contrib/metric-learn.svg?branch=master
   :target: https://travis-ci.org/scikit-learn-contrib/metric-learn
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |Code coverage| image:: https://codecov.io/gh/scikit-learn-contrib/metric-learn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/metric-learn