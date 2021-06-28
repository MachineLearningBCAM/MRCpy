"""Super class for Minimax Risk Classifiers."""

import warnings

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
    Base Class for Minimax Risk Classifier

    The class implements minimax risk classfication
    using two types of commonly used loss functions,
    namely logistic loss and 0-1 loss.
    The class also provides different feature mapping functions
    that can enhance the performance of the classifier for different datasets.

    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default='0-1'
        The type of loss function to use for the risk minimization.

    s : float, default=0.3
        For tuning the estimation of expected values
        of feature mapping function(phi).
        Must be a positive float value and
        expected to be in the 0 to 1 in general cases.

    deterministic : bool, default=False
        For determining if the prediction of the labels
        should be done in a deterministic way or not.

    random_state : int, RandomState instance, default=None
        Used when 'fourier' option for feature mappings are used
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

    solver : {'SCS', 'ECOS'}, default='SCS'
        The type of CVX solver to use for solving the problem.
        In some cases, one solver might not work,
        so we might need to use the other solver from the set.

    max_iters : int, default=10000
        The maximum number of iterations to use
        for finding the solution of optimization
        using the subgradient approach.

    phi : `str` {'fourier', 'relu', 'threshold'} or
           `BasePhi` instance, default='linear'
        The type of feature mapping function to use for mapping the input data
        'fourier', 'relu' and 'threshold'
        are the currenlty available feature mapping methods.

    **phi_kwargs : Groups the multiple optional parameters
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
                 deterministic=False, random_state=None,
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

        # Feature mappings
        if phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=2,
                                        fit_intercept=fit_intercept,
                                        random_state=random_state,
                                        **phi_kwargs)
        elif phi == 'linear':
            self.phi = BasePhi(n_classes=2,
                               fit_intercept=fit_intercept)
        elif phi == 'threshold':
            self.phi = ThresholdPhi(n_classes=2,
                                    fit_intercept=fit_intercept,
                                    **phi_kwargs)
        elif phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=2,
                                     fit_intercept=fit_intercept,
                                     random_state=random_state,
                                     **phi_kwargs)
        elif isinstance(phi, BasePhi):
            self.phi = phi
        else:
            raise ValueError('Unexpected feature mapping type ... ')

        # Solver list available in cvxpy
        self.solvers = ['MOSEK', 'SCS', 'ECOS']

    def fit(self, X, Y=None, X_=None, n_classes=None, tau_=None, lambda_=None):
        '''
        Fit the MRC model.

        Parameters
        ----------
        X : array-like of shape (n_samples1, n_dimensions)
            Training instances used in
            - Calculating the estimates for the minimax risk classification
            - Also, used in optimization when X_ is not given

        Y_ : array-like of shape (n_samples1), default = None
            Labels corresponding to the training instances
            used to compute the estimates for the optimization.

            If the estimates used in the MRC are already given as a parameter
            to the this function, then the labels
            for the instances are not required
            unless and until the threshold feature mappings are used.

            When the threshold feature mappings are used,
            these labels are required to find the thresholds
            using the instance, label pair.

        X_ : array-like of shape (n_samples2, n_dimensions), default = None
            These instances are optional and
            when given, will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        n_classes : int
            Number of labels in the dataset.
            If the labels Y are not provided, then this argument is required.

        tau_ : array-like of shape (n_features * n_classes) or float,
               default=None
            The mean estimates
            for the expectations of feature mappings.
            If a single float value is passed,
            all the features in the feature mapping have the same estimates.

        lambda_ : array-like of shape (n_features * n_classes) or float,
                  defautl=None
            The variance in the mean estimates
            for the expectations of the feature mappings.
            If a single float value is passed,
            all the features in the feature mapping
            have the samve variance for the estimates.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X = check_array(X, accept_sparse=True)

        # Check if separate instances are given for the optimization
        if X_ is not None:
            X_opt = check_array(X_, accept_sparse=True)
            not_all_instances = False
        else:
            # Use the training samples used to compute estimate
            # for the optimization.
            X_opt = X

            # If the labels are not given, then these instances
            # are assumed to be given for optimization only and 
            # hence all the instances will be used.
            if Y is None:
                not_all_instances = False
            else:
                not_all_instances = True

        # Learn the classes
        # Classes will only be learnt if the estimates are not given
        self.classes_ = None
        if tau_ is None or lambda_ is None:
            X, Y = check_X_y(X, Y, accept_sparse=True)

            # Map the values of Y from 0 to n_classes-1
            origY = Y
            self.classes_ = np.unique(origY)
            self.n_classes = len(self.classes_)
            Y = np.zeros(origY.shape[0], dtype=int)

            for i, y in enumerate(self.classes_):
                Y[origY == y] = i

        elif type(n_classes) == int:
            self.n_classes = n_classes

        elif Y is not None:
            self.n_classes = len(np.unique(Y))

        else:
            raise ValueError("Expected the labels \'Y\' or " +
                             "a valid \'n_classes\' argument ... ")

        # Set the number of classes in phi
        self.phi.n_classes = self.n_classes

        # Fit the feature mappings
        if tau_ is None and lambda_ is None:
            self.phi.fit(X, Y)

        # Set the interval estimates if they are given
        # Otherwise compute the interval estimates
        if tau_ is not None:
            if isinstance(tau_, (float, int)):
                # Check if the input is an array or a single value.
                # If a single value is given as input,
                # it is converted to an array of size
                # equal to the number of features of phi
                # it imples that the estimates for all the features is same.
                tau_ = np.asarray([tau_] * self.phi.len_)
            self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)

        else:
            self.tau_ = self.phi.est_exp(X, Y)

        if lambda_ is not None:
            if isinstance(lambda_, (float, int)):
                # Check if the input is an array or a single value.
                # If a single value is given as input,
                # it is converted to an array of size
                # equal to the number of features of phi
                # it imples that the variance in the estimates
                # for all the features is same.
                lambda_ = np.asarray([lambda_] * self.phi.len_)
            self.lambda_ = check_array(lambda_, accept_sparse=True,
                                       ensure_2d=False)

        else:
            self.lambda_ = (self.s * \
                            self.phi.est_std(X, Y)) / \
                            np.sqrt(X.shape[0])

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory
        n_max = 5000
        n = X_opt.shape[0]
        if not_all_instances and n_max < n:
            n = n_max

        # Get the feature mapping corresponding to the training instances
        phi = self.phi.eval_x(X_opt[:n])

        # Supress the depreciation warnings
        warnings.simplefilter('ignore')

        # Fit the MRC classifier
        self.minimax_risk(phi)

        self.is_fitted_ = True

        return self

    def try_solvers(self, objective, constraints, mu, nu=None):
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

        nu : cvxpy variable of float value, default = None
            Parameter used in the optimization problem

        Returns
        -------
        mu_ : cvxpy array of shape (number of featuers in phi)
            The value of the parameters
            corresponding to the optimum value of the objective function.

        nu_ : cvxpy variable of float value, if None, not returned
            The value of the parameter
            corresponding to the optimum value of the objective function.
            The parameter is not returned
            in case the cvxpy variable is not defined (i.e, None) initially
            when it is passed as the argument to this function.
        '''

        # Reuse the solution from previous call to fit.
        if self.warm_start:
            # Use a previous solution if it exists.
            try:
                mu.value = self.mu_
                nu.value = self.nu_
            except AttributeError:
                pass

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False,
                   warm_start=self.warm_start)

        mu_ = mu.value

        # Solving the constrained MRC problem
        # which has only parameters mu
        if nu is not None:
            nu_ = nu.value
        else:
            nu_ = 0

        # if the solver could not find values of mu for the given solver
        if mu_ is None or nu_ is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Reuse the solution from previous call to fit.
                    if self.warm_start:
                        # Use a previous solution if it exists.
                        try:
                            mu.value = self.mu_
                            nu.value = self.nu_
                        except AttributeError:
                            pass

                    # Solve the problem
                    prob.solve(solver=s, verbose=False,
                               warm_start=self.warm_start)

                    # Check the values
                    mu_ = mu.value
                    if nu is not None:
                        nu_ = nu.value

                    # Break the loop once the solution is obtained
                    if mu_ is not None and nu_ is not None:
                        break

        # If no solution can be found for the optimization.
        if mu_ is None:
            raise ValueError('CVXpy solver couldn\'t find a solution .... \n \
                              The problem is ', prob.status)

        if nu is not None:
            return mu_, nu_
        else:
            return mu_

    def predict(self, X):
        '''
        Returns the predicted classes for X samples.

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
        n = X.shape[0]

        if self.loss == 'log':
            # In case of logistic loss function,
            # the classification is always deterministic

            phi = self.phi.eval_x(X)

            # Deterministic classification
            v = np.dot(phi, self.mu_)
            y_pred = np.argmax(v, axis=1)

        elif self.loss == '0-1':
            # In case of 0-1 loss function,
            # the classification can be done
            # in deterministic or non-deterministic way
            # but by default, it is prefered to do it in non-deterministic way
            # as it is designed for it.

            proba = self.predict_proba(X)

            if self.deterministic:
                y_pred = np.argmax(proba, axis=1)
            else:
                y_pred = [np.random.choice(self.n_classes, size=1, p=pc)[0]
                          for pc in proba]

        # Check if any labels are learnt
        # Otherwise return the default labels i.e., from 0 to n_classes-1.
        if self.classes_ is not None:
            y_pred = np.asarray([self.classes_[label] for label in y_pred])

        return y_pred
