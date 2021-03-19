""" Unit tests for the constrained minimax risk classifiers """

import unittest

import numpy as np

# Import the dataset
from MRCpy import CMRC
from MRCpy.datasets import load_iris


class TestCMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(return_X_y=True)

    def CMRC_training(self, phi, loss, use_cvx):
        r = np.unique(self.y).shape[0]
        clf = CMRC(n_classes=r, phi=phi, loss=loss, use_cvx=use_cvx)
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))

    # Without using cvxpy
    # Training test for MRC with 0-1 loss and threshold features.
    def test_MRC0_1_with_threshold(self):
        self.CMRC_training(phi='threshold', loss='0-1', use_cvx=False)

    # Training test for MRC with 0-1 loss and linear kernel.
    def test_MRC0_1_with_linear(self):
        self.CMRC_training(phi='linear', loss='0-1', use_cvx=False)

    # Training test for MRC with 0-1 loss and gaussian kernel.
    def test_MRC0_1_with_gaussian(self):
        self.CMRC_training(phi='gaussian', loss='0-1', use_cvx=False)

    # Training test for MRC with log loss and threshold features.
    def test_MRClog_with_threshold(self):
        self.CMRC_training(phi='threshold', loss='log', use_cvx=False)

    # Training test for MRC with log loss and linear kernel.
    def test_MRClog_with_linear(self):
        self.CMRC_training(phi='linear', loss='log', use_cvx=False)

    # Training test for MRC with log loss and gaussian kernel.
    def test_MRClog_with_gaussian(self):
        self.CMRC_training(phi='gaussian', loss='log', use_cvx=False)

    # Using cvxpy
    # Training test for MRC with 0-1 loss and threshold features.
    def test_MRC0_1_with_threshold_cvx(self):
        self.CMRC_training(phi='threshold', loss='0-1', use_cvx=True)

    # Training test for MRC with 0-1 loss and linear kernel.
    def test_MRC0_1_with_linear_cvx(self):
        self.CMRC_training(phi='linear', loss='0-1', use_cvx=True)

    # Training test for MRC with 0-1 loss and gaussian kernel.
    def test_MRC0_1_with_gaussian_cvx(self):
        self.CMRC_training(phi='gaussian', loss='0-1', use_cvx=True)

    # Training test for MRC with log loss and threshold features.
    def test_MRClog_with_threshold_cvx(self):
        self.CMRC_training(phi='threshold', loss='log', use_cvx=True)

    # Training test for MRC with log loss and linear kernel.
    def test_MRClog_with_linear_cvx(self):
        self.CMRC_training(phi='linear', loss='log', use_cvx=True)

    # Training test for MRC with log loss and gaussian kernel.
    def test_MRClog_with_gaussian_cvx(self):
        self.CMRC_training(phi='gaussian', loss='log', use_cvx=True)
