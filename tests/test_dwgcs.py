""" Unit tests for the constrained minimax risk classifiers """

import unittest

import numpy as np
from sklearn.model_selection import train_test_split

# Import the dataset
from MRCpy import DWGCS
from MRCpy.datasets import load_comp_vs_sci_short

class TestDWGCS(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X_TrainSet, self.Y_TrainSet, self.X_TestSet, self.Y_TestSet = \
                                        load_comp_vs_sci_short(with_info=False)

    def DWGCS_training_check(self, phi, loss, solver):

        # Simple functional check.
        r = np.unique(self.Y_TrainSet).shape[0]
        clf = DWGCS(phi=phi,
                    loss=loss,
                    solver=solver,
                    max_iters=500)
        clf.fit(self.X_TrainSet, self.Y_TrainSet, self.X_TestSet)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        hy_x = clf.predict_proba(self.X_TestSet)
        self.assertTrue(hy_x.shape == (self.X_TestSet.shape[0], r))
        self.assertTrue(np.all(np.sum(hy_x, axis=1)))

        y_pred = clf.predict(self.X_TestSet)
        self.assertTrue(y_pred.shape == (self.X_TestSet.shape[0],))

    # Skip tests for now. Need to fix bug with psd_wrap
    # Training test for DWGCS with 0-1 loss.
    def test_DWGCS0_1_adam(self):
        self.DWGCS_training_check(phi='linear', loss='0-1', solver='adam')
        # self.DWGCS_training_check(phi='fourier', loss='0-1', solver='adam')

    # Training test for DWGCS with log loss.
    def test_DWGCSlog_adam(self):
        self.DWGCS_training_check(phi='linear', loss='log', solver='adam')
        # self.DWGCS_training_check(phi='fourier', loss='log', solver='adam')
