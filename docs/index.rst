.. MRCpy documentation master file, created by
   sphinx-quickstart on Sat Jun  5 15:07:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MRCpy: A Library for Minimax Risk Classifiers
=============================================

|Travis-CI Build Status| |Code coverage|

`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ implements Minimax
Risk Classifiers (MRCs), which are based on the robust risk minimization
(RRM) framework. Unlike empirical risk minimization (ERM), RRM accounts for
uncertainty in the underlying data distribution by optimizing the worst-case
risk over a set of plausible distributions. These techniques give rise to a
broad family of classification methods that provide guarantees in terms of
an upper bound on the classification error at training.

MRCpy provides a unified interface for different variants of MRCs, following
the design standards of popular Python machine learning libraries. The
library includes efficient implementations of MRC-based methods designed to
scale to large datasets and high-dimensional problems. It also provides
implementations of established techniques that can be formulated as MRCs,
including L1-regularized logistic regression, zero-one adversarial
classification, and maximum entropy machines. In addition, MRCpy includes
PyTorch-based classifiers that enable the integration of MRC objectives with
deep neural networks, allowing users to train DNNs using minimax risk-based
learning objectives.

.. code-block:: bash

   pip install MRCpy

.. code-block:: python

   from MRCpy import MRC
   from MRCpy.datasets import load_mammographic
   from sklearn.model_selection import train_test_split

   X, Y = load_mammographic(with_info=False)
   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

   clf = MRC().fit(X_train, y_train)

   lower_error, upper_error = clf.get_lower_bound(), clf.get_upper_bound()
   accuracy = clf.score(X_test, y_test)

.. grid:: 1 2 2 4
   :gutter: 3
   :class-container: sd-mb-4

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Installation, dependencies, and a quick-start example.

   .. grid-item-card:: User Guide
      :link: minimax_framework
      :link-type: doc

      The minimax risk framework behind MRCpy's classifiers.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Detailed description of every class and function in MRCpy.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc

      Worked examples of MRCpy applied to real datasets.

If you use MRCpy in your research, please see :doc:`citing` for the relevant
references and BibTeX entries.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   minimax_framework
   api
   auto_examples/index
   citing

Funding
-------

Funding in direct support of this work has been provided through different research projects by the following institutions.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item::

      .. figure:: fund_logo.png
         :align: center
         :width: 150
         :alt: Spanish Ministry of Science and Innovation logo

         Spanish Ministry of Science and Innovation through the project
         **PID2019-105058GA-I00** funded by **MCIN/AEI/10.13039/501100011033**.

   .. grid-item::

      .. figure:: axalogo.png
         :align: center
         :width: 150
         :alt: AXA Research Fund logo

         AXA Research Fund through the project **"Early Prognosis of
         COVID-19 Infections via Machine Learning"** funded in the
         Exceptional Flash Call **"Mitigating risk in the wake of the
         COVID-19 pandemic"**.

   .. grid-item::

      .. figure:: logogobiernovasco.png
         :align: center
         :width: 150
         :alt: Basque Government logo

         Basque Government through the project **"Mathematical Modeling
         Applied to Health"**, and through the **"ELKARTEK Program"**.

.. |Travis-CI Build Status| image:: https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield
   :target: https://circleci.com/gh/MachineLearningBCAM/MRCpy
.. |Code coverage| image:: https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy
   :target: https://codecov.io/gh/MachineLearningBCAM/MRCpy
