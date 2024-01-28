""" Unit tests for the adaptive minimax risk classifiers """

import unittest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import the dataset
from MRCpy import AMRC
from MRCpy.datasets import load_usenet2


class TestAMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_usenet2(with_info=False)

        # Normalize data
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

    def AMRC_training_check(self, phi, loss):

        # Simple functional check.
        r = np.unique(self.y).shape[0]
        n = self.X.shape[0]

        n = 200
        Y_pred = np.zeros(n - 1)

        clf = AMRC(n_classes=r, phi=phi, loss=loss)

        mistakes = 0
        for i in range(n - 1):
            clf.fit(self.X[i, :], self.y[i])
            Y_pred[i] = clf.predict(self.X[i + 1, :])

            upper = clf.get_upper_bound()
            upper_accumulated = clf.get_upper_bound_accumulated()

            if Y_pred[i] != self.y[i + 1]:
                mistakes += 1

            accumulated_mistakes = mistakes / (i + 1)

            self.assertTrue(hasattr(clf, 'is_fitted_'))
            self.assertTrue(clf.is_fitted_)

            # Predict the probabilities for each class for the given instances.
            self.assertTrue(accumulated_mistakes <= upper_accumulated)

    # Without using cvxpy
    # Training test for AMRC.
    def test_AMRC(self):
        self.AMRC_training_check(phi='linear', loss='0-1')
        self.AMRC_training_check(phi='fourier', loss='0-1')