###############
Getting started
###############

Installation and Setup
======================

**Installation**

The latest built version of ``MRCpy`` can be installed using `pip` as
```
pip install MRCpy
```

Alternatively, the latest code (under development) of MRCpy can be installed by downloading the source GitHub repository as 
```
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
python3 setup.py install
```
You may then use ``pytest tests`` to run all the checks (you will need to have the ``pytest`` package installed).

**Dependencies**

- ``Python`` >= 3.9
- ``numpy`` >= 1.19
- ``scipy`` >= 1.4.1
- ``scikit-learn`` >= 0.22
- ``cvxpy`` >= 1.1
- ``pandas`` >= 1.0
- ``pyarrow``
- ``gurobipy`` (requires license)
- ``pycddlib`` >= 3.0.2 (required only for LMRC)

.. note::
   Installing ``pycddlib`` requires the GMP library, and ``pip install pycddlib``
   alone may not be sufficient. See the
   `pycddlib installation guide <https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation>`_
   for details on installing GMP and other dependencies.

Optional (for PyTorch MGCE classifier):

- ``torch`` >= 1.9.0
- ``tqdm`` >= 4.50.0

Quick start
===========

This example loads the mammographic dataset, and trains the `MRC` classifier
using 0-1 loss (i.e., the default loss).

::

    from MRCpy import MRC
    from MRCpy.datasets import load_mammographic
    from sklearn.model_selection import train_test_split

    # Load the mammographic dataset
    X, Y = load_mammographic(with_info=False)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create the MRC classifier using default loss (0-1)
    clf = MRC()

    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Bounds on the classification error (only for MRC)
    lower_error = clf.get_lower_bound()
    upper_error = clf.get_upper_bound()

    # Compute the accuracy on the test set
    accuracy = clf.score(X_test, y_test)

Dataset Loaders
===============
`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ provides convenient loader functions for a variety of standard datasets via ``MRCpy.datasets``. Note that the datasets ``adult``, ``magic``, ``mnist``, ``cats vs dogs`` and ``yearbook`` are not available in the built version (installed through pip) of ``MRCpy``.

Usage
-----

All loaders follow the same interface. By default they return ``(X, y)`` arrays:

::

    from MRCpy.datasets import load_mammographic

    # Returns (X, y) arrays
    X, y = load_mammographic(with_info=False)

    # Returns a Bunch object with .data, .target, .DESCR, .filename
    dataset = load_mammographic(with_info=True)

Available Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10

   * - Loader
     - Classes
     - Samples
     - Features
   * - ``load_iris``
     - 3
     - 150
     - 4
   * - ``load_haberman``
     - 2
     - 306
     - 3
   * - ``load_mammographic``
     - 2
     - 961
     - 5
   * - ``load_diabetes``
     - 2
     - 668
     - 8
   * - ``load_credit``
     - 2
     - 690
     - 15
   * - ``load_indian_liver``
     - 2
     - 583
     - 10
   * - ``load_glass``
     - 6
     - 214
     - 9
   * - ``load_ecoli``
     - 8
     - 336
     - 8
   * - ``load_vehicle``
     - 4
     - 846
     - 18
   * - ``load_segment``
     - 7
     - 2310
     - 19
   * - ``load_redwine``
     - 10
     - 6497
     - 11
   * - ``load_satellite``
     - 6
     - 6435
     - 36
   * - ``load_optdigits``
     - 10
     - 5620
     - 64
   * - ``load_usenet2``
     - 2
     - 1500
     - 99
   * - ``load_adult`` :sup:`*`
     - 2
     - 48842
     - 14
   * - ``load_magic`` :sup:`*`
     - 2
     - 19020
     - 10
   * - ``load_forestcov`` :sup:`*`
     - 7
     - 581012
     - 54
   * - ``load_letterrecog`` :sup:`*`
     - 26
     - 20000
     - 16

Pre-extracted Feature Datasets
------------------------------

These datasets provide features extracted using a pretrained ResNet18 neural network.
They are not available in the pip-installed version.

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 10

   * - Loader
     - Classes
     - Samples
     - Features
   * - ``load_mnist_features_resnet18`` :sup:`*`
     - 10
     - 70000
     - 512
   * - ``load_catsvsdogs_features_resnet18`` :sup:`*`
     - 2
     - 23262
     - 512
   * - ``load_yearbook_features_resnet18`` :sup:`*`
     - 2
     - 37921
     - 512

:sup:`*` Not available in the pip-installed version of MRCpy.

Newsgroup Text Datasets
-----------------------

Binary text classification datasets derived from the 20 Newsgroups corpus,
with TF-IDF features (1000 dimensions).

.. list-table::
   :header-rows: 1
   :widths: 40 10 10

   * - Loader
     - Classes
     - Features
   * - ``load_comp_vs_sci``
     - 2
     - 1000
   * - ``load_comp_vs_talk``
     - 2
     - 1000
   * - ``load_rec_vs_sci``
     - 2
     - 1000
   * - ``load_rec_vs_talk``
     - 2
     - 1000
   * - ``load_sci_vs_talk``
     - 2
     - 1000

