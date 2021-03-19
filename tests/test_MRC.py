""" Unit tests for the minimax risk classifiers """

import unittest

import numpy as np

# Import the dataset
from MRCpy import MRC
from MRCpy.datasets import load_iris


class TestMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(return_X_y=True)

    def MRC_training(self, phi, loss, use_cvx):
        r = np.unique(self.y).shape[0]
        clf = MRC(n_classes=r, phi=phi, loss=loss, use_cvx=use_cvx)
        clf.fit(self.X, self.y)
        upper = clf.upper_
        lower = clf.getLowerBound()
        self.assertTrue(lower <= upper)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        if loss == 'log':
            hy_x = clf.predict_proba(self.X)
            self.assertTrue(hy_x.shape == (self.X.shape[0], r))
            self.assertTrue(np.all(np.sum(hy_x, axis=1)==1))

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Without using cvxpy
    # Training test for MRC with 0-1 loss.
    def test_MRC0_1(self):
        self.MRC_training(phi='threshold', loss='0-1', use_cvx=False)
        self.MRC_training(phi='linear', loss='0-1', use_cvx=False)
        self.MRC_training(phi='gaussian', loss='0-1', use_cvx=False)

    # Training test for MRC with log loss.
    def test_MRClog(self):
        self.MRC_training(phi='threshold', loss='log', use_cvx=False)
        self.MRC_training(phi='linear', loss='log', use_cvx=False)
        self.MRC_training(phi='gaussian', loss='log', use_cvx=False)

    # Using cvxpy
    # Training test for MRC with 0-1 loss.
    def test_MRC0_1_cvx(self):
        self.MRC_training(phi='threshold', loss='0-1', use_cvx=True)
        self.MRC_training(phi='linear', loss='0-1', use_cvx=True)
        self.MRC_training(phi='gaussian', loss='0-1', use_cvx=True)

    # Training test for MRC with log loss.
    def test_MRClog_cvx(self):
        self.MRC_training(phi='threshold', loss='log', use_cvx=True)
        self.MRC_training(phi='linear', loss='log', use_cvx=True)
        self.MRC_training(phi='gaussian', loss='log', use_cvx=True)
