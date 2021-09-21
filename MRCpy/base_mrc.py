'''Super class for Minimax Risk Classifiers.'''

import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi


class BaseMRC(BaseEstimator, ClassifierMixin):
    '''
    Base class for different minimax risk classifiers.

    This class is a parent class for different MRCs
    implemented in the library.
    It defines the different parameters and the layout.

    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default='0-1'
        The type of loss function to use for the risk minimization.

    s : float, default=0.3
        For tuning the estimation of expected values
        of feature mapping function.

    deterministic : bool, default=None
        For determining if the prediction of the labels
        should be done in a deterministic way or not.
        For '0-1' loss, the non-deterministic ('False') approach
        works well.
        For 'log' loss, the deterministic ('True') approach
        works well.
        If the user does not specify the value, the default value
        is set according to loss function.

    random_state : int, RandomState instance, default=None
        Used when 'fourier' and 'relu' options for feature mappings are used
        to produce the random weights.

    fit_intercept : bool, default=True
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    warm_start : bool, default=False
        When set to True,
        reuse the solution of the previous call to fit as initialization,
        otherwise, just erase the previous solution.

    use_cvx : bool, default=False
        If True, use CVXpy library for the optimization
        instead of the subgradient methods.

    solver : str {'SCS', 'ECOS', 'MOSEK'}, default='MOSEK'
        The type of CVX solver to use for solving the problem.
        In some cases, one solver might not work,
        so you might need to change solver depending on the problem.
        'MOSEK' is a commercial solver for which one might need to
        request for a license. A free license can be requested
        `here <https://www.mosek.com/products/academic-licenses/>`_

    max_iters : int, default=10000
        The maximum number of iterations to use
        for finding the solution of optimization
        using the subgradient approach.

    phi : str {'fourier', 'relu', 'threshold', 'linear'} or
          `BasePhi` instance (custom features), default='linear'
        The type of feature mapping function to use for mapping the input data
        'fourier', 'relu', 'threshold' and 'linear'
        are the currenlty available feature mapping methods.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

    **phi_kwargs : Additional parameters for feature mappings.
                Groups the multiple optional parameters
                for the corresponding feature mappings(phi).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument -
                `MRC(loss='log', phi='fourier', n_components=500)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : bool
        True if the classifier is fitted i.e., the parameters are learnt.

    tau_ : array-like of shape (n_features)
        The mean estimates for the expectations of feature mappings.

    lambda_ : array-like of shape (n_features)
        The variance in the mean estimates for the expectations
        of the feature mappings.

    classes_ : array-like of shape (n_classes)
        Labels in the given dataset.
        If the labels Y are not given during fit
        i.e., tau and lambda are given as input,
        then this array is None.
    '''

    def __init__(self, loss='0-1', s=0.3,
                 deterministic=None, random_state=None,
                 fit_intercept=True, warm_start=False, use_cvx=False,
                 solver='MOSEK', max_iters=10000, phi='linear', **phi_kwargs):

        self.loss = loss
        self.s = s
        self.deterministic = deterministic
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.use_cvx = use_cvx
        self.solver = solver
        self.max_iters = max_iters
        # Feature mapping and its parameters
        self.phi = phi
        self.phi_kwargs = phi_kwargs
        # Solver list for cvxpy
        self.solvers = ['MOSEK', 'SCS', 'ECOS']

    def fit(self, X, Y):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

            n_samples is the number of samples and
            n_features is the number of features.

        Y : array-like of shape (n_samples1), default = None
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X, Y = check_X_y(X, Y, accept_sparse=True)

        # Obtaining the number of classes and mapping the labels to integers
        origY = Y
        self.classes_ = np.unique(origY)
        n_classes = len(self.classes_)
        Y = np.zeros(origY.shape[0], dtype=int)

        # Map the values of Y from 0 to n_classes-1
        for i, y in enumerate(self.classes_):
            Y[origY == y] = i

        # Feature mappings
        if self.phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=n_classes,
                                        fit_intercept=self.fit_intercept,
                                        random_state=self.random_state,
                                        **self.phi_kwargs)
        elif self.phi == 'linear':
            self.phi = BasePhi(n_classes=n_classes,
                               fit_intercept=self.fit_intercept,
                               **self.phi_kwargs)
        elif self.phi == 'threshold':
            self.phi = ThresholdPhi(n_classes=n_classes,
                                    fit_intercept=self.fit_intercept,
                                    **self.phi_kwargs)
        elif self.phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=n_classes,
                                     fit_intercept=self.fit_intercept,
                                     random_state=self.random_state,
                                     **self.phi_kwargs)
        elif not isinstance(self.phi, BasePhi):
            raise ValueError('Unexpected feature mapping type ... ')

        # Fit the feature mappings
        self.phi.fit(X, Y)

        # Compute the expectation estimates
        tau_ = self.phi.est_exp(X, Y)
        lambda_ = (self.s * self.phi.est_std(X, Y)) / \
            np.sqrt(X.shape[0])

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory
        n_max = 5000
        n = X.shape[0]
        if n_max < n:
            n = n_max

        # Fit the MRC classifier
        self.minimax_risk(X[:n], tau_, lambda_, n_classes)

        return self

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Solves the minimax risk optimization problem
        for the corresponding variant of MRC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : array-like of shape (n_features * n_classes)
            The mean estimates
            for the expectations of feature mappings.

        lambda_ : array-like of shape (n_features * n_classes)
            The variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : int
            Number of labels in the dataset.

        Returns
        -------
        self :
            Fitted estimator

        '''

        # Variants of MRCs inheriting from this class should
        # extend this function to implement the solution to their
        # minimax risk optimization problem.

        raise NotImplementedError('BaseMRC is not an implemented classifier.' +
                                  ' It is base class for different MRCs.' +
                                  ' This functions needs to be implemented' +
                                  ' by a sub-class implementing a MRC.')

    def try_solvers(self, objective, constraints, mu):
        '''
        Solves the MRC problem
        using different types of solvers available in CVXpy

        Parameters
        ----------
        objective : cvxpy variable of float value
            Defines the minimization problem of the MRC.

        constraints : array-like of shape (n_constraints)
            Defines the constraints for the MRC optimization.

        mu : cvxpy array of shape (number of featuers in phi)
            Parameters used in the optimization problem

        Returns
        -------
        mu_ : array-like of shape (number of featuers in phi)
            The value of the parameters
            corresponding to the optimum value of the objective function.

        objective_value : float
            The optimized objective value.

        '''

        # Reuse the solution from previous call to fit.
        if self.warm_start:
            # Use a previous solution if it exists.
            try:
                mu.value = self.mu_
            except AttributeError:
                pass

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False,
                   warm_start=self.warm_start)

        mu_ = mu.value

        # if the solver could not find values of mu for the given solver
        if mu_ is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Reuse the solution from previous call to fit.
                    if self.warm_start:
                        # Use a previous solution if it exists.
                        try:
                            mu.value = self.mu_
                        except AttributeError:
                            pass

                    # Solve the problem
                    prob.solve(solver=s, verbose=False,
                               warm_start=self.warm_start)

                    # Check the values
                    mu_ = mu.value

                    # Break the loop once the solution is obtained
                    if mu_ is not None:
                        break

        # If no solution can be found for the optimization.
        if mu_ is None:
            raise ValueError('CVXpy solver couldn\'t find a solution .... ' +
                             'The problem is ', prob.status)

        objective_value = prob.value
        return mu_, objective_value

    def predict_proba(self, X):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Computes conditional probabilities corresponding
        to each class for the given unlabeled instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Testing instances for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        hy_x : array-like of shape (n_samples, n_classes)
            The conditional probabilities (p(y|x))
            corresponding to each class.
        '''

        # Variants of MRCs inheriting from this class
        # implement this function to compute the conditional
        # probabilities using the classifier obtained from minimax risk

        raise NotImplementedError('BaseMRC is not an implemented classifier.' +
                                  ' It is base class for different MRCs.' +
                                  ' This functions needs to be implemented' +
                                  ' by a sub-class implementing a MRC.')

    def predict(self, X):
        '''
        Returns the predicted classes for the given instances
        using the probabilities given by the function `predict_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : array-like of shape (n_samples)
            The predicted labels corresponding to the given instances.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Get the prediction probabilities for the classifier
        proba = self.predict_proba(X)

        if self.deterministic:
            y_pred = np.argmax(proba, axis=1)
        else:
            y_pred = [np.random.choice(self.n_classes, size=1, p=pc)[0]
                      for pc in proba]

        # Check if the labels were provided for fitting
        # (labels might be omitted if fitting is done through minimax_risk)
        # Otherwise return the default labels i.e., from 0 to n_classes-1.
        if hasattr(self, 'classes_'):
            y_pred = np.asarray([self.classes_[label] for label in y_pred])

        return y_pred
