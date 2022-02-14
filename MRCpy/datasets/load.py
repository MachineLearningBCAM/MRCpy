# -*- coding: utf-8 -*-
"""
.. _load:

Set of loaders and convenient functions to access Dataset
=========================================================
"""
import csv
import zipfile
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch


def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y


def load_adult(with_info=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'adult.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)
        # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_diabetes(with_info=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_iris(with_info=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 3
    Samples per class              [50,50,50]
    Samples total                         150
    Dimensionality                          4
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'iris.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'iris.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_redwine(with_info=False):
    """Load and return the Red Wine Dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class            [1599, 4898]
    Samples total                        6497
    Dimensionality                         11
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'redwine.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'redwine.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray([np.float(i) for i in d[:-1]],
                                 dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_forestcov(with_info=False):
    """Load and return the Forestcov Dataset (classification).

    =========================   =============================================
    Classes                                                                7
    Samples per class           [211840,283301,35754,2747,9493,17367,20510,0]
    Samples total                                                      581012
    Dimensionality                                                         54
    Features                                             int, float, positive
    =========================   =============================================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'forestcov.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'forestcov.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)
        # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_letterrecog(with_info=False):
    """Load and return the Letter Recognition Dataset (classification).

    =================   =====================
    Classes                                26
    Samples total                       20000
    Dimensionality                         16
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'letter-recognition.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'letter-recognition.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)
        # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[1:], dtype=np.float64)
            if d[0] in classes:
                index = classes.index(d[0])
                target[i] = np.asarray(index, dtype=int)
            else:
                classes.append(d[0])
                target[i] = np.asarray(classes.index(d[0]), dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_ecoli(with_info=False):
    """Load and return the Ecoli Dataset (classification).

    =================== =========================
    Classes                                     8
    Samples per class     [143,77,52,35,20,5,2,2]
    Samples total                             336
    Dimensionality                              8
    Features                 int, float, positive
    =================== =========================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'ecoli.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'ecoli.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray([float(i) for i in d[1:-1]], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_vehicle(with_info=False):
    """Load and return the Vehicle Dataset (classification).

    =================   =====================
    Classes                                 4
    Samples per class       [240,240,240,226]
    Samples total                         846
    Dimensionality                         18
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'vehicle.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'vehicle.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)
    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_segment(with_info=False):
    """Load and return the Segment prediction dataset (classification).

    =================   =====================
    Classes                                 7
    Samples total                        2310
    Dimensionality                         19
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'segment.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'segment.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray([np.float(i) for i in d[:-1]],
                                     dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_satellite(with_info=False):
    """Load and return the Satellite prediction dataset (classification).

    =================   =====================
    Classes                                 6
    Samples total                        6435
    Dimensionality                         36
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'satellite.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'satellite.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_optdigits(with_info=False):
    """Load and return the Optdigits prediction dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class               383, 307]
    Samples total                        5620
    Dimensionality                         64
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'optdigits.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'optdigits.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_credit(with_info=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'credit.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_magic(with_info=False):
    """Load and return the Magic Gamma Telescope dataset (classification).

    =================== ======================
    Classes                                 2
    Samples per class            [6688,12332]
    Samples total                       19020
    Dimensionality                         10
    Features                            float
    =================== ======================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'magic.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'magic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_glass(with_info=False):
    """Load and return the Glass Identification Data Set (classification).

    ================== =========================
    Classes                                   6
    Samples per class    [70, 76, 17, 29, 13, 9]
    Samples total                           214
    Dimensionality                            9
    Features                              float
    ================== =========================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of glass csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'glass.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'glass.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=descr_text,
                 feature_names=['RI: refractive index',
                                "Na: Sodium (unit measurement: "
                                "weight percent in corresponding oxide, "
                                "as are attributes 4-10)",
                                'Mg: Magnesium ',
                                'Al: Aluminim',
                                'Si: Silicon',
                                'K: Potassium',
                                'Ca: Calcium',
                                'Ba: Barium',
                                'Fe: Iron'])


def load_haberman(with_info=False):
    """Load and return the Haberman's Survival Data Set (classification).

    ================== ============
    Classes                      2
    Samples per class    [225, 82]
    Samples total              306
    Dimensionality               3
    Features                   int
    ================== ============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of haberman csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'haberman.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'haberman.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=descr_text,
                 feature_names=['PatientAge',
                                'OperationYear',
                                'PositiveAxillaryNodesDetected'])


def load_mammographic(with_info=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ================== ============
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ================== ============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    # fdescr_name = join(module_path, 'descr', 'mammographic.rst')
    # with open(fdescr_name) as f:
    #     descr_text = f.read()

    data_file_name = join(module_path, 'data', 'mammographic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=None,
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'])


def load_indian_liver(with_info=False):
    """Load and return the Indian Liver Patient Data Set
    (classification).

    ============================ =============================
    Classes                                                 2
    Samples per class                              [416, 167]
    Samples total                                         583
    Dimensionality                                         10
    Features                                       int, float
    Missing Values                                     4 (nan)
    =========================== ==============================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'indianLiverPatient.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)
    # with open(join(module_path, 'descr',
    #                'indianLiverPatient.rst')) as rst_file:
    #     fdescr = [line.decode('utf-8').strip() \
    #               for line in rst_file.readlines()]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=['Age of the patient',
                                'Gender of the patient',
                                'Total Bilirubin',
                                'Direct Bilirubin',
                                'Alkaline Phosphotase',
                                'Alamine Aminotransferase',
                                'Aspartate Aminotransferase',
                                'Total Protiens',
                                'Albumin',
                                'A/G Ratio'])


def load_yearbook_path():
    """
    Returns the path of Yearbook Image Dataset
    """
    module_path = dirname(__file__)
    path = join(module_path, 'data', 'yearbook')
    return path


def load_mnist_features_resnet18(with_info=False, split=False):
    """Load and return the MNIST Data Set features extracted using a
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

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    split : boolean, default=False.
        If True, returns a dictionary instead of an array in the place of the
        data.

    Returns
    -------
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
    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'mnist_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('mnist_features_resnet18_1.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('mnist_features_resnet18_2.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_3.csv.zip'))
    df3 = pd.read_csv(zf.open('mnist_features_resnet18_3.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_4.csv.zip'))
    df4 = pd.read_csv(zf.open('mnist_features_resnet18_4.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'mnist_features_resnet18_5.csv.zip'))
    df5 = pd.read_csv(zf.open('mnist_features_resnet18_5.csv'), header=None)

    dataset = np.array(pd.concat([df1, df2, df3, df4, df5]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    target = normalizeLabels(target)
    if not with_info:
        if split:
            # X_train, X_test, Y_train, Y_test
            X_train = data[:60000, :]
            Y_train = target[:60000]
            X_test = data[60000:, :]
            Y_test = target[60000:]
            return X_train, X_test, Y_train, Y_test
        else:
            return data, target
    else:
        if split:
            data = {'train': data[:60000, :], 'test': data[60000:, :]}
            target = {'train': target[:60000], 'test': target[60000:]}
        return Bunch(data=data, target=target, DESCR=descr_text)
    return 0


def load_catsvsdogs_features_resnet18(with_info=False):
    """Load and return the Cats vs Dogs Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ===========================================
    Classes                                   2
    Samples per class             [11658,11604]
    Samples total                         23262
    Dimensionality                          512
    Features                              float
    ===========================================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of Cats vs Dogs ResNet18 features
        csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr',
                       'catsvsdogs_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'catsvsdogs_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_1.csv'))
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'catsvsdogs_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('catsvsdogs_features_resnet18_2.csv'))

    dataset = np.array(pd.concat([df1, df2]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=descr_text)


def load_yearbook_features_resnet18(with_info=False, with_attributes=False):
    """Load and return the Yearbook Data Set features extracted using a
    pretrained ResNet18 neural network (classification).

    ===========================================
    Classes                                   2
    Samples per class             [20248,17673]
    Samples total                         37921
    Dimensionality                          512
    Features                              float
    ===========================================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    with_attributes : boolean, default=False.
        If True, returns an additional dictionary containing information of
        additional attributes: year, state, city, school of the portraits.
        The key 'attr_labels' in the dictionary contains these labels
        corresponding to each columns, while 'attr_data' corresponds to
        the attribute data in form of numpy array.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of Yearbook ResNet18 features
        csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'yearbook_features_resnet18.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    zf = zipfile.ZipFile(join(module_path, 'data',
                              'yearbook_features_resnet18_1.csv.zip'))
    df1 = pd.read_csv(zf.open('yearbook_features_resnet18_1.csv'), header=None)
    zf = zipfile.ZipFile(join(module_path, 'data',
                              'yearbook_features_resnet18_2.csv.zip'))
    df2 = pd.read_csv(zf.open('yearbook_features_resnet18_2.csv'), header=None)

    dataset = np.array(pd.concat([df1, df2]))
    data = dataset[:, :-1]
    target = dataset[:, -1]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if with_attributes:
        attr = pd.read_csv(join(module_path, 'data',
                                'yearbook_attributes.csv'))
        attr_labels = attr.columns.values
        attr_val = attr.values
        attr = {'attr_labels': attr_labels, 'attr_data': attr_val}

        if not with_info:
            return data, normalizeLabels(target), attr

        return Bunch(data=data, target=normalizeLabels(target),
                     attributes=attr, DESCR=descr_text)

    else:
        if not with_info:
            return data, normalizeLabels(target)

        return Bunch(data=data, target=normalizeLabels(target),
                     DESCR=descr_text)
