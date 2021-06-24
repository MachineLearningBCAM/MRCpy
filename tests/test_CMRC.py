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
        clf = CMRC(phi=phi, loss=loss,
                   use_cvx=use_cvx, solver='SCS')
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        if loss == 'log':
            hy_x = clf.predict_proba(self.X)
            self.assertTrue(hy_x.shape == (self.X.shape[0], r))
            self.assertTrue(np.all(np.sum(hy_x, axis=1)))

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))

    # Without using cvxpy
    # Training test for CMRC with 0-1 loss.
    def test_CMRC0_1(self):
        self.CMRC_training(phi='threshold', loss='0-1', use_cvx=False)
        self.CMRC_training(phi='linear', loss='0-1', use_cvx=False)
        self.CMRC_training(phi='fourier', loss='0-1', use_cvx=False)
        self.CMRC_training(phi='relu', loss='0-1', use_cvx=False)

    # Training test for CMRC with log loss.
    def test_CMRClog(self):
        self.CMRC_training(phi='threshold', loss='log', use_cvx=False)
        self.CMRC_training(phi='linear', loss='log', use_cvx=False)
        self.CMRC_training(phi='fourier', loss='log', use_cvx=False)
        self.CMRC_training(phi='relu', loss='log', use_cvx=False)

    # Using cvxpy
    # Training test for CMRC with 0-1 loss.
    def test_CMRC0_1_cvx(self):
        self.CMRC_training(phi='threshold', loss='0-1', use_cvx=True)
        self.CMRC_training(phi='linear', loss='0-1', use_cvx=True)
        self.CMRC_training(phi='fourier', loss='0-1', use_cvx=True)
        self.CMRC_training(phi='relu', loss='0-1', use_cvx=True)

    # Training test for CMRC with log loss.
    def test_CMRClog_cvx(self):
        self.CMRC_training(phi='threshold', loss='log', use_cvx=True)
        self.CMRC_training(phi='linear', loss='log', use_cvx=True)
        self.CMRC_training(phi='fourier', loss='log', use_cvx=True)
        self.CMRC_training(phi='relu', loss='log', use_cvx=True)
