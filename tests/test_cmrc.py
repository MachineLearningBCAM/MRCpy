""" Unit tests for the constrained minimax risk classifiers """

import unittest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the dataset
from MRCpy import CMRC
from MRCpy.datasets import load_iris


class TestCMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(with_info=False)

    def CMRC_training_check(self, phi, loss, solver):

        # Simple functional check.
        r = np.unique(self.y).shape[0]
        clf = CMRC(phi=phi,
                   loss=loss,
                   solver=solver,
                   max_iters=100)
        clf.fit(self.X, self.y)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        hy_x = clf.predict_proba(self.X)
        self.assertTrue(hy_x.shape == (self.X.shape[0], r))
        self.assertTrue(np.all(np.sum(hy_x, axis=1)))

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))


    # The predicted label is the label with the largest score
    # Phi(x, y)^T mu, for the 0-1 loss
    # (h(y|x) = (1 + Phi(x, y)^T mu - varphi(x))_+) and the log-loss
    # (softmax of the scores), in multiclass and binary classification.
    # With standardized instances and ReLU features, varphi(x) < 0 for
    # many instances.
    def test_CMRC_predict_largest_score(self):
        X = StandardScaler().fit_transform(self.X)
        for y in (self.y, (self.y == 0).astype(int)):
            for loss in ('0-1', 'log'):
                for phi in ('linear', 'relu'):
                    clf = CMRC(phi=phi, loss=loss, solver='adam',
                               max_iters=100, random_state=0)
                    clf.fit(X, y)
                    scores = clf.scores(clf.compute_features(X), clf.mu_)
                    np.testing.assert_array_equal(clf.predict(X),
                                                  np.argmax(scores, axis=1))

    # Without using cvxpy
    # Training test for CMRC with 0-1 loss.
    def test_CMRC0_1_grad(self):
        self.CMRC_training_check(phi='threshold', loss='0-1', solver='sgd')
        self.CMRC_training_check(phi='linear', loss='0-1', solver='sgd')
        self.CMRC_training_check(phi='fourier', loss='0-1', solver='sgd')
        self.CMRC_training_check(phi='relu', loss='0-1', solver='sgd')

    # Training test for CMRC with log loss.
    def test_CMRClog_grad(self):
        self.CMRC_training_check(phi='threshold', loss='log', solver='sgd')
        self.CMRC_training_check(phi='linear', loss='log', solver='sgd')
        self.CMRC_training_check(phi='fourier', loss='log', solver='sgd')
        self.CMRC_training_check(phi='relu', loss='log', solver='sgd')

    # Training test for CMRC with 0-1 loss.
    def test_CMRC0_1_adam(self):
        self.CMRC_training_check(phi='threshold', loss='0-1', solver='adam')
        self.CMRC_training_check(phi='linear', loss='0-1', solver='adam')
        self.CMRC_training_check(phi='fourier', loss='0-1', solver='adam')
        self.CMRC_training_check(phi='relu', loss='0-1', solver='adam')

    # Training test for CMRC with log loss.
    def test_CMRClog_adam(self):
        self.CMRC_training_check(phi='threshold', loss='log', solver='adam')
        self.CMRC_training_check(phi='linear', loss='log', solver='adam')
        self.CMRC_training_check(phi='fourier', loss='log', solver='adam')
        self.CMRC_training_check(phi='relu', loss='log', solver='adam')

    # Using cvxpy
    # Training test for CMRC with 0-1 loss.
    def test_CMRC0_1_cvx(self):
        self.CMRC_training_check(phi='threshold', loss='0-1', solver='cvx')
        self.CMRC_training_check(phi='linear', loss='0-1', solver='cvx')
        self.CMRC_training_check(phi='fourier', loss='0-1', solver='cvx')
        self.CMRC_training_check(phi='relu', loss='0-1', solver='cvx')

    # Training test for CMRC with log loss.
    def test_CMRClog_cvx(self):
        self.CMRC_training_check(phi='threshold', loss='log', solver='cvx')
        self.CMRC_training_check(phi='linear', loss='log', solver='cvx')
        self.CMRC_training_check(phi='fourier', loss='log', solver='cvx')
        self.CMRC_training_check(phi='relu', loss='log', solver='cvx')
