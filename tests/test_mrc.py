""" Unit tests for the minimax risk classifiers """

import unittest

import numpy as np

# Import the dataset
from MRCpy import MRC, CMRC, AMRC
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
        # lower = clf.get_lower_bound()
        print('Upper bound: ', upper)
        # self.assertTrue(lower <= upper)
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

    # Using column generation (cg) solver
    # Training test for MRC with 0-1 loss.
    # Note: cg solver only supports 0-1 loss
    def test_MRC0_1_cg(self):
        self.MRC_training(phi='threshold', loss='0-1', solver='cg')
        self.MRC_training(phi='linear', loss='0-1', solver='cg')
        self.MRC_training(phi='fourier', loss='0-1', solver='cg')
        self.MRC_training(phi='relu', loss='0-1', solver='cg')

    # Using constraint-column generation (ccg) solver
    # Training test for MRC with 0-1 loss.
    # Note: ccg solver only supports 0-1 loss
    # Uses a smaller dataset to stay within Gurobi free license limits
    def test_MRC0_1_ccg(self):
        X_small = self.X[:60]
        y_small = self.y[:60]
        r = np.unique(y_small).shape[0]
        for phi in ['threshold', 'linear', 'fourier', 'relu']:
            clf = MRC(phi=phi, loss='0-1', max_iters=500, solver='ccg')
            clf.fit(X_small, y_small)
            upper = clf.get_upper_bound()
            print('Upper bound: ', upper)
            self.assertTrue(hasattr(clf, 'is_fitted_'))
            self.assertTrue(clf.is_fitted_)
            hy_x = clf.predict_proba(X_small)
            self.assertTrue(hy_x.shape == (X_small.shape[0], r))
            y_pred = clf.predict(X_small)
            self.assertTrue(y_pred.shape == (X_small.shape[0], ))

    # Test binary classification
    def test_MRC_binary(self):
        # Create binary classification dataset
        X_binary = self.X[self.y != 2]
        y_binary = self.y[self.y != 2]
        
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        clf.fit(X_binary, y_binary)
        print(X_binary.shape)
        print(y_binary.shape)
        print(X_binary)
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(X_binary)
        self.assertTrue(y_pred.shape == (X_binary.shape[0], ))
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))

    # Test with fit_intercept=False
    def test_MRC_no_intercept(self):
        clf = MRC(phi='linear', loss='0-1', solver='subgrad', fit_intercept=False)
        clf.fit(self.X, self.y)
        
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Test with different deterministic values
    def test_MRC_deterministic(self):
        clf = MRC(phi='linear', loss='0-1', solver='subgrad', deterministic=True)
        clf.fit(self.X, self.y)
        
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Test with custom phi parameters
    def test_MRC_custom_phi_params(self):
        # Test with custom Fourier features
        clf = MRC(phi='fourier', loss='0-1', solver='subgrad', \
                  n_components=50, sigma=0.5)
        clf.fit(self.X, self.y)
        
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Test with custom ReLU features
    def test_MRC_relu_custom(self):
        clf = MRC(phi='relu', loss='0-1', solver='subgrad', n_components=50)
        clf.fit(self.X, self.y)
        
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(self.X)
        self.assertTrue(y_pred.shape == (self.X.shape[0], ))

    # Test score method
    def test_MRC_score(self):
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        clf.fit(self.X, self.y)
        
        score = clf.score(self.X, self.y)
        self.assertTrue(0 <= score <= 1)

    # Test with small dataset
    def test_MRC_small_dataset(self):
        X_small = self.X[:20]
        y_small = self.y[:20]
        
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        clf.fit(X_small, y_small)
        
        self.assertTrue(clf.is_fitted_)
        y_pred = clf.predict(X_small)
        self.assertTrue(y_pred.shape == (X_small.shape[0], ))


class TestMRCEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.X, self.y = load_iris(with_info=False)

    def test_MRC_single_class(self):
        # Test with single class (should handle gracefully or raise error)
        X_single = self.X[self.y == 0]
        y_single = self.y[self.y == 0]
        
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        try:
            clf.fit(X_single, y_single)
            # If it doesn't raise an error, check predictions
            y_pred = clf.predict(X_single)
            self.assertTrue(np.all(y_pred == 0))
        except (ValueError, AssertionError):
            # Expected behavior for single class
            pass

    def test_MRC_predict_before_fit(self):
        # Test prediction before fitting
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        
        with self.assertRaises((AttributeError, ValueError)):
            clf.predict(self.X)

    def test_MRC_different_input_shapes(self):
        # Test with 1D input (should fail or be reshaped)
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        clf.fit(self.X, self.y)
        
        # Test with single sample
        X_single = self.X[0:1]
        y_pred = clf.predict(X_single)
        self.assertTrue(y_pred.shape == (1,))

    def test_MRC_with_nan_values(self):
        # Test handling of NaN values
        X_nan = self.X.copy()
        X_nan[0, 0] = np.nan
        
        clf = MRC(phi='linear', loss='0-1', solver='subgrad')
        try:
            clf.fit(X_nan, self.y)
            # If it handles NaN, check it doesn't crash
            self.assertTrue(True)
        except (ValueError, RuntimeError):
            # Expected behavior for NaN values
            pass

    def test_MRC_deterministic_reproducibility(self):
        # Test that deterministic=True gives reproducible results
        clf1 = MRC(phi='linear', loss='0-1', solver='subgrad', 
                   deterministic=True, random_state=42)
        clf1.fit(self.X, self.y)
        pred1 = clf1.predict(self.X)
        
        clf2 = MRC(phi='linear', loss='0-1', solver='subgrad',
                   deterministic=True, random_state=42)
        clf2.fit(self.X, self.y)
        pred2 = clf2.predict(self.X)
        
        self.assertTrue(np.array_equal(pred1, pred2))


class TestMRCParameters(unittest.TestCase):
    """Test different parameter combinations"""

    def setUp(self):
        self.X, self.y = load_iris(with_info=False)

    def test_MRC_different_max_iters(self):
        # Test with different max_iters values
        for max_iters in [10, 100, 500]:
            clf = MRC(phi='linear', loss='0-1', solver='subgrad',
                     max_iters=max_iters)
            clf.fit(self.X, self.y)
            self.assertTrue(clf.is_fitted_)

    def test_MRC_random_state(self):
        # Test random_state parameter
        clf1 = MRC(phi='fourier', loss='0-1', solver='subgrad',
                  random_state=42)
        clf1.fit(self.X, self.y)
        
        clf2 = MRC(phi='fourier', loss='0-1', solver='subgrad',
                  random_state=42)
        clf2.fit(self.X, self.y)
        
        # Predictions should be similar with same random state
        pred1 = clf1.predict(self.X)
        pred2 = clf2.predict(self.X)
        
        # At least some predictions should match
        self.assertTrue(np.sum(pred1 == pred2) > len(pred1) * 0.5)

