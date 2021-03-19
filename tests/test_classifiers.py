""" Unit tests for the minimax risk classifiers """

import unittest

import numpy as np

# Import the dataset
from MRCpy import CMRC, MRC
from MRCpy.datasets import load_iris


class TestClassifiers(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(return_X_y=True)

    # Simple training test for the MRC.
    def test_MRC_train(self):
        r = np.unique(self.y).shape[0]
        clf = MRC(n_classes=r, phi='threshold', loss='0-1')
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))

    # Simple training test for CMRC
    def test_CMRC_train(self):
        r = np.unique(self.y).shape[0]
        clf = CMRC(n_classes=r, phi='threshold', loss='0-1')
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))
