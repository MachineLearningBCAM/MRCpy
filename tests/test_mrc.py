""" Unit tests for the minimax risk classifiers """

import unittest

import numpy as np

# Import the dataset
from MRCpy import MRC
from MRCpy.datasets import load_iris


class TestMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(with_info=False)

    def MRC_training(self, phi, loss, solver):
        r = np.unique(self.y).shape[0]
        clf = MRC(phi=phi,
                  loss=loss,
                  max_iters=500,
                  solver=solver)
        clf.fit(self.X, self.y)
        upper = clf.get_upper_bound()
        lower = clf.get_lower_bound()
        self.assertTrue(lower <= upper)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        hy_x = clf.predict_proba(self.X)
        self.assertTrue(hy_x.shape == (self.X.shape[0], r))
        self.assertTrue(np.all(np.sum(hy_x, axis=1)))

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Without using cvxpy
    # Training test for MRC with 0-1 loss.
    def test_MRC0_1(self):
        self.MRC_training(phi='threshold', loss='0-1', solver='subgrad')
        self.MRC_training(phi='linear', loss='0-1', solver='subgrad')
        self.MRC_training(phi='fourier', loss='0-1', solver='subgrad')
        self.MRC_training(phi='relu', loss='0-1', solver='subgrad')

    # Training test for MRC with log loss.
    def test_MRClog(self):
        self.MRC_training(phi='threshold', loss='log', solver='subgrad')
        self.MRC_training(phi='linear', loss='log', solver='subgrad')
        self.MRC_training(phi='fourier', loss='log', solver='subgrad')
        self.MRC_training(phi='relu', loss='log', solver='subgrad')

    # Using cvxpy
    # Training test for MRC with 0-1 loss.
    def test_MRC0_1_cvx(self):
        self.MRC_training(phi='threshold', loss='0-1', solver='cvx')
        self.MRC_training(phi='linear', loss='0-1', solver='cvx')
        self.MRC_training(phi='fourier', loss='0-1', solver='cvx')
        self.MRC_training(phi='relu', loss='0-1', solver='cvx')

    # Training test for MRC with log loss.
    def test_MRClog_cvx(self):
        self.MRC_training(phi='threshold', loss='log', solver='cvx')
        self.MRC_training(phi='linear', loss='log', solver='cvx')
        self.MRC_training(phi='fourier', loss='log', solver='cvx')
        self.MRC_training(phi='relu', loss='log', solver='cvx')
