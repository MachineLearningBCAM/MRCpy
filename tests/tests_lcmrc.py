import unittest

import numpy as np

# Import the dataset
from MRCpy.datasets import load_iris
from lcmrc import LCMRC

class TestMRC(unittest.TestCase):

    def setUp(self):
        # Get the sample data for testing.
        self.X, self.y = load_iris(with_info=False)
        self.n_classes = len(np.unique(self.y))

    def MRC_training(self, phi, loss, alpha = None, use_closed_form = False):
        d1 = np.unique(self.y).shape[0]
        if phi == 'linear':
            clf = LCMRC(phi=phi,
                      loss=loss,
                      n_classes = self.n_classes,
                      alpha = alpha,
                      use_closed_form=use_closed_form,
                      )
        elif phi == 'fourier':
            clf = LCMRC(phi=phi,
                       n_components = 32,
                       loss=loss,
                       alpha=alpha,
                       use_closed_form=use_closed_form,
                       n_classes=self.n_classes,
                       )
        else:
            print("Unknown phi")
            exit()
        clf.fit(self.X, self.y)
        d2 = clf.n_classification_classes_
        upper = clf.get_upper_bound()
        lower = clf.get_lower_bound(self.X,self.y)
        self.assertTrue(lower <= upper)
        #print("lower bound: ", lower, " | upper bound: ", upper)
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)

        # Predict the probabilities for each class for the given instances.
        hy_x = clf.predict_proba(self.X)
        #print("Average error: ", clf.error(self.X, self.y))
        self.assertTrue(hy_x.shape == (self.X.shape[0], d2))
        self.assertTrue(np.all(np.sum(hy_x, axis=1)))

        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Using cvxpy
    # Training test for MRC with 0-1 loss.
    def test_LMRC0_1_cvx(self):
        self.MRC_training(phi='linear', loss='0-1', alpha = 0.321)
        self.MRC_training(phi='fourier', loss='0-1', alpha = 0.321)

    # Training test for MRC with ordinal loss.
    def test_LMRC_ordinal_cvx(self):
        self.MRC_training(phi='linear', loss='ordinal')
        self.MRC_training(phi='fourier', loss='ordinal')

    # Training test for MRC with abstention loss.
    def test_LMRC_abstention_cvx(self):
        self.MRC_training(phi='linear',  loss='abstention')
        self.MRC_training(phi='fourier',  loss='abstention')

    #Trainning test for a user given matrix.
    def test_LMRC_expert_cvx(self):
        LT = np.array([[0, 0.66, 0.34], [0.84, 0, 0.16], [0.33, 0.67, 0]])
        self.MRC_training(phi='linear', loss=LT)
        self.MRC_training(phi='fourier', loss=LT)

    # Training test for MRC with squared-ordinal loss.
    def test_LMRC_squared_ordinal_cvx(self):
        self.MRC_training(phi='linear', loss='ordinal-squared', use_closed_form=True)
        self.MRC_training(phi='fourier', loss='ordinal-squared', use_closed_form=True)

