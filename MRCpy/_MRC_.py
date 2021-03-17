"""Super class for Minimax Risk Classifiers."""
import cvxpy as cvx

import numpy as np

from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping
from MRCpy.phi import PhiGaussian, \
                      PhiThreshold, \
                      PhiLinear, \
                      Phi


class _MRC_():
    """
    Minimax Risk Classifier

    The class implements minimax risk classfication
    using two types of commonly used loss functions,
    namely logistic loss and 0-1 loss.
    The class also provides different feature mapping functions
    that can enhance the performance of the classifier for different datasets.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset
        used to determine the type constraints to be used
        during the optimization
        i.e., to use linear or nonlinear number of constraints
        to reduce the complexity.

    equality : bool, default=False
        The type of Learning. If true the LPC is asymptotically calibrated,
        if false the LPC is approximately calibrated.

    s : float, default=0.3
        For tuning the estimation of expected values
        of feature mapping function(phi).
        Must be a positive float value and
        expected to be in the 0 to 1 in general cases.

    deterministic : bool, default=False
        For determining if the prediction of the labels
        should be done in a deterministic way or not.

    random_state : int, RandomState instance, default=None
        Used when 'gaussian' option for feature mappings are used
        to produce the random weights.

    loss : {'0-1', 'log'}, default='0-1'
        The type of loss function to use for the risk minimization.

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

    warm_start : bool
            When set to True,
            reuse the solution of the previous call to fit as initialization,
            otherwise, just erase the previous solution.

    phi : {'gaussian','linear','threshold'} or phi instance, default='linear'
        The type of feature mapping function to use for mapping the input data
        The 'gaussian', 'linear' and 'threshold'are predefined feature mapping
        If the type is 'custom',
        it means that the user has to define his own feature mapping function

    **phi_kwargs : multiple optional parameters
                   for the corresponding feature mappings(phi)

    Attributes
    ----------
    is_fitted_ : bool
        True if the classifier is fitted i.e., the parameters are learnt.

    tau_ : array-like of shape (n_features) or float
        The mean estimates for the expectations of feature mappings.

    lambda_ : array-like of shape (n_features) or float
        The variance in the mean estimates for the expectations
        of the feature mappings.

    """

    def __init__(self, n_classes, equality=False, s=0.3,
                 deterministic=False, random_state=None, loss='0-1',
                 warm_start=False, use_cvx=False, solver='SCS',
                 max_iters=10000, phi='gaussian', **phi_kwargs):

        self.n_classes = n_classes
        self.equality = equality
        self.s = s
        self.deterministic = deterministic
        self.random_state = random_state
        self.loss = loss
        self.solver = solver
        self.use_cvx = use_cvx
        self.max_iters = max_iters
        self.warm_start = warm_start

        if phi == 'gaussian':
            self.phi = PhiGaussian(n_classes=n_classes,
                                   random_state=random_state, **phi_kwargs)
        elif phi == 'linear':
            self.phi = PhiLinear(n_classes=n_classes)
        elif phi == 'threshold':
            self.phi = PhiThreshold(n_classes=n_classes, **phi_kwargs)
        elif isinstance(phi, Phi):
            self.phi = phi
        else:
            raise ValueError('Unexpected feature mapping type ... ')

        # Solver list available in cvxpy
        self.solvers = ['SCS', 'ECOS', 'ECOS_BB']

    def fit(self, X, Y=None, X_=None, tau_=None, lambda_=None):
        """
        Fit the MRC model.

        Parameters
        ----------
        X : array-like of shape (n_samples1, n_dimensions)
            Training instances used in
            - Calculating the estimates for the minimax risk classification
            - Also, used in optimization when X_ is not given

        Y_ : array-like of shape (n_samples1/n_samples2), default = None
            Labels corresponding to the training instances
            used to compute the estimates for the optimization.

            If the estimates used in the MRC are already given as a parameter
            to the this function, then the labels
            for the instances are not required
            unless and until the threshold feature mappings are used.

            When the threshold feature mappings are used,
            these labels are required to find the thresholds
            using the instance, label pair.

        X_ : array-like of shape (n_samples, n_dimensions), default = None
            These instances are optional and
            when given, will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        tau_ : array-like of shape (n_features) or float, default=None
            The mean estimates
            for the expectations of feature mappings.
            If a single float value is passed,
            all the features in the feature mapping have the same estimates.

        lambda_ : array-like of shape (n_features) or float, defautl=None
            The variance in the mean estimates
            for the expectations of the feature mappings.
            If a single float value is passed,
            all the features in the feature mapping
            have the samve variance for the estimates.

        Returns
        -------
        self :
            Fitted estimator

        """

        X = check_array(X, accept_sparse=True)

        # Set the type of configuration to be learnt i.e.,
        # whether to learn duplicate instances or not,
        # based on the MRC/CMRC model
        self.setLearnConfigType()

        # Check if separate instances are given for the optimization
        if X_ is not None:
            X_opt = check_array(X_, accept_sparse=True)

        else:
            # Use the training samples used to compute estimate
            # for the optimization.
            X_opt = X

        # Fit the feature mappings
        self.phi.fit(X, Y)

        # Learn the classes
        # Classes will only be learnt if the estimates are not given
        self.classes_ = None
        if tau_ is None and lambda_ is None:
            # Map the values of Y from 0 to n_classes-1
            origY = Y
            self.classes_ = np.unique(origY)
            Y = np.zeros(origY.shape[0], dtype=int)

            for i, y in enumerate(self.classes_):
                Y[origY == y] = i

        # Set the interval estimates if they are given
        # Otherwise compute the interval estimates
        if tau_ is not None:
            if isinstance(tau_, (float, int)):
                # Check if the input is an array or a single value.
                # If a single value is given as input,
                # it is converted to an array of size
                # equal to the number of features of phi
                # it imples that the estimates for all the features is same.
                tau_ = np.asarray([tau_]*self.phi.len_)
            self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)

        else:
            X, Y = check_X_y(X, Y, accept_sparse=True)
            self.tau_ = self.phi.estExp(X, Y)

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
            X, Y = check_X_y(X, Y, accept_sparse=True)
            self.lambda_ = (self.s * self.phi.estStd(X, Y)) / \
                np.sqrt(X.shape[0])

        # a and b are needed for cvxpy optimization only.
        if self.use_cvx:
            self.a = self.tau_ - self.lambda_
            self.b = self.tau_ + self.lambda_
            print('Using CVXpy for optimization ...')

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory
        n_max = 5000
        n = X_opt.shape[0]
        if n_max < n:
            n = n_max

        # Fit the MRC classifier
        self._minimaxRisk(X_opt[:n, :])

        self.is_fitted_ = True

        return self

    def trySolvers(self, objective, constraints, mu, zhi, nu=None):
        """
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

        zhi : cvxpy array of shape (number of featuers in phi)
            Parameters used in the optimization problem

        nu : cvxpy variable of float value, default = None
            Parameter used in the optimization problem

        Returns
        -------
        mu_ : cvxpy array of shape (number of featuers in phi)
            The value of the parameters
            corresponding to the optimum value of the objective function.

        zhi_ : cvxpy array of shape (number of featuers in phi)
            The value of the parameters
            corresponding to the optimum value of the objective function.

        nu_ : cvxpy variable of float value, if None, not returned
            The value of the parameter
            corresponding to the optimum value of the objective function.
            The parameter is not returned
            in case the cvxpy variable is not defined (i.e, None) initially
            when it is passed as the argument to this function.
        """

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve(solver=self.solver, verbose=False)

        mu_ = mu.value
        zhi_ = zhi.value

        # Solving the constrained MRC problem
        # which has only parameters mu and zhi
        if nu is not None:
            nu_ = nu.value
        else:
            nu_ = 0

        # if the solver could not find values of mu for the given solver
        if mu_ is None or zhi_ is None or nu_ is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:
                    # Solve the problem
                    _ = prob.solve(solver=s, verbose=False)

                    # Check the values
                    mu_ = mu.value
                    zhi_ = zhi.value
                    if nu is not None:
                        nu_ = nu.value

                    # Break the loop once the solution is obtained
                    if mu_ is not None and zhi_ is not None and \
                            nu_ is not None:
                        break

        # If no solution can be found for the optimization.
        if mu_ is None or zhi_ is None:
            raise ValueError('CVXpy solver couldn\'t find a solution .... \n \
                              The problem is ', prob.status)

        if nu is not None:
            return mu_, zhi_, nu_
        else:
            return mu_, zhi_

    def predict(self, X):
        """
        Returns the predicted classes for X samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : array-like of shape (n_samples, )
            The predicted labels corresponding to the given instances.

        """

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        if self.loss == 'log':
            # In case of logistic loss function,
            # the classification is always deterministic

            phi = self.phi.eval(X)

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
