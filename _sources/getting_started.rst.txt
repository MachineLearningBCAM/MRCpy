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
`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ library incorporates a variety of datasets, along with descriptions and convenient loader functions for each dataset. Next, we show the description of the functions you can find and import from `MRCpy.datasets`.


``normalizeLabels(origY)``
--------------------------

Normalize the labels of the instances in the range 0,..., r-1 for r classes.


``load_adult(with_info=False)``
---------------------------------

Load and return the adult incomes prediction dataset (classification).

=================   ==============
Classes                          2
Samples per class    [37155,11687]
Samples total                48882
Dimensionality                  14
Features             int, positive
=================   ==============

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_diabetes(with_info=False)``
-----------------------------------
Load and return the Pima Indians Diabetes dataset (classification).

=================   =====================
Classes                                 2
Samples per class               [500,268]
Samples total                         668
Dimensionality                          8
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_iris(with_info=False)``
-------------------------------
Load and return the Iris Plants Dataset (classification).

=================   =====================
Classes                                 3
Samples per class              [50,50,50]
Samples total                         150
Dimensionality                          4
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_redwine(with_info=False)``
----------------------------------
Load and return the Red Wine Dataset (classification).

=================   =====================
Classes                                10
Samples per class            [1599, 4898]
Samples total                        6497
Dimensionality                         11
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_forestcov(with_info=False)``
------------------------------------
Load and return the Forestcov Plants Dataset (classification).

=================   =====================
Classes                                 7
Samples per class [211840,283301,35754,
                 2747,9493,17367,20510,0]
Samples total                      581012
Dimensionality                         54
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_letterrecog(with_info=False)``
--------------------------------------
Load and return the Letter Recognition Dataset (classification).

=================   =====================
Classes                                26
Samples total                       20000
Dimensionality                         16
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_ecoli(with_info=False)``
--------------------------------
Load and return the Ecoli Dataset (classification).

=================   =====================
Classes                                 8
Samples per class [143,77,52,35,20,5,2,2]
Samples total                         336
Dimensionality                          8
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_vehicle(with_info=False)``
----------------------------------
Load and return the Vehicle Dataset (classification).

=================   =====================
Classes                                 4
Samples per class       [240,240,240,226]
Samples total                         846
Dimensionality                         18
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of the dataset.

(data, target) : tuple if ``with_info`` is True


``load_segment(with_info=False)``
----------------------------------
Load and return the Segment prediction dataset (classification).

=================   =====================
Classes                                 7
Samples per class              [383, 307]
Samples total                        2310
Dimensionality                         19
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of adult csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_satellite(with_info=False)``
------------------------------------
Load and return the Satellite prediction dataset (classification).

=================   =====================
Classes                                 6
Samples per class               383, 307]
Samples total                        6435
Dimensionality                         36
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of adult csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_optdigits(with_info=False)``
------------------------------------
Load and return the Optdigits prediction dataset (classification).

=================   =====================
Classes                                10
Samples per class               383, 307]
Samples total                        5620
Dimensionality                         64
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of adult csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_credit(with_info=False)``
---------------------------------
Load and return the Credit Approval prediction dataset (classification).

=================   =====================
Classes                                 2
Samples per class               383, 307]
Samples total                         690
Dimensionality                         15
Features             int, float, positive
=================   =====================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of adult csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_magic(with_info=False)``
--------------------------------
Load and return the Magic Gamma Telescope dataset (classification).

=================== ======================
Classes                                 2
Samples per class            [6688,12332]
Samples total                       19020
Dimensionality                         10
Features                            float
=================== ======================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of adult csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_glass(with_info=False)``
--------------------------------
Load and return the Glass Identification Data Set (classification).

==================== =======================
Classes                                   6
Samples per class    [70, 76, 17, 29, 13, 9]
Samples total                           214
Dimensionality                            9
Features                              float
==================== =======================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of glass csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_haberman(with_info=False)``
-----------------------------------
Load and return the Haberman's Survival Data Set (classification).

============= =================
Classes                      2
Samples per class    [225, 82]
Samples total              306
Dimensionality               3
Features                   int
============= =================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of haberman csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_mammographic(with_info=False)``
---------------------------------------
Load and return the Mammographic Mass Data Set (classification).

============ ==================
Classes                      2
Samples per class    [516, 445]
Samples total              961
Dimensionality               5
Features                   int
============ ==================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of mammographic csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_indian_liver(with_info=False)``
---------------------------------------
Load and return the Indian Liver Patient Data Set
(classification).

========================== ===============================
Classes                                                 2
Samples per class                              [416, 167]
Samples total                                         583
Dimensionality                                         10
Features                                       int, float
Missing Values                                     4 (nan)
========================== ===============================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

data : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of satellite csv dataset.

(data, target) : tuple if ``with_info`` is True

``load_yearbook_path()``
------------------------
Returns the path of Yearbook Image Dataset


``load_mnist_features_resnet18(with_info=False, split=False)``
--------------------------------------------------------------
Load and return the MNIST Data Set features extracted using a
pretrained ResNet18 neural network (classification).

======================= ===========================
Classes                                          2
Samples per class Train [5923,6742,5958,6131,5842,            
                         5421,5918,6265,5851,5949]
Samples per class Test    [980,1135,1032,1010,982,
                            892,958,1028,974,1009]
Samples total Train                          60000
Samples total Test                           10000
Samples total                                70000
Dimensionality                                 512
Features                                     float
======================= ===========================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.
split : boolean, default=False.
    If True, returns a dictionary instead of an array in the place of the
    data.

**Returns**

bunch : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of MNIST ResNet18 features
    csv dataset. If `split=False`, data is
    an array. If `split=True` data is a dictionary with 'train' and 'test'
    splits.

(data, target) : tuple if ``with_info`` is True. If `split=False`, data is
    an array. If `split=True` data is a dictionary with 'train' and 'test'
    splits.


``load_catsvsdogs_features_resnet18(with_info=False)``
------------------------------------------------------
Load and return the Cats vs Dogs Data Set features extracted using a
pretrained ResNet18 neural network (classification).

==================== =======================
Classes                                   2
Samples per class             [11658,11604]
Samples total                         23262
Dimensionality                          512
Features                              float
==================== =======================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

**Returns**

bunch : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of Cats vs Dogs ResNet18 features
    csv dataset.

(data, target) : tuple if ``with_info`` is True


``load_yearbook_features_resnet18(with_info=False, with_attributes=False)``
----------------------------------------------------
Load and return the Yearbook Data Set features extracted using a
pretrained ResNet18 neural network (classification).

==================== =======================
Classes                                   2
Samples per class             [20248,17673]
Samples total                         37921
Dimensionality                          512
Features                              float
==================== =======================

**Parameters**

with_info : boolean, default=False.
    If True, returns ``(data, target)`` instead of a Bunch object.
    See below for more information about the `data` and `target` object.

with_attributes : boolean, default=False.
    If True, returns an additional dictionary containing information of
    additional attributes: year, state, city, school of the portraits.
    The key 'attr_labels' in the dictionary contains these labels
    corresponding to each columns, while 'attr_data' corresponds to
    the attribute data in form of numpy array.

**Returns**

bunch : Bunch
    Dictionary-like object, the interesting attributes are:
    'data', the data to learn, 'target', the classification targets,
    'DESCR', the full description of the dataset,
    and 'filename', the physical location of Yearbook ResNet18 features
    csv dataset.

(data, target) : tuple if ``with_info`` is True



