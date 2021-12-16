###############
Getting started
###############

Installation and Setup
======================

**Installation**

The latest code of MRCpy can be installed by downloading the source repository and running ``python setup.py install``. You may then run ``pytest tests`` to run all tests (you will need to have the ``pytest`` package installed).

**Dependencies**

- `Python` :math:`\geq` 3.6
- `numpy` :math:`\geq` 1.18.1, `scipy`:math:`\geq` 1.4.1, `scikit-learn` :math:`\geq` 0.21.0, `cvxpy`, `mosek`

Quick start
===========

This example loads the mammographic dataset, and trains the `MRC` classifier
using 0-1 loss (i.e., the default loss).

::

    from MRCpy import MRC
    from MRCpy.datasets import load_mammographic
    from sklearn.model_selection import train_test_split

    # Load the mammographic dataset
    X, Y = load_mammographic(return_X_y=True)

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

