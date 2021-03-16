""" Unit tests for the minimax risk classifiers """

from minimax_risk_classifiers.MRC import MRC
from minimax_risk_classifiers.CMRC import CMRC
#import the dataset
from datasets import load_iris
import numpy as np

import unittest


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
        self.assertTrue(clf.is_fitted_ == True)

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0],))
