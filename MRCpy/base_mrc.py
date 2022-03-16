'''
.. _base_mrc:
Super class for Minimax Risk Classifiers.
'''

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
    Base class for different minimax risk classifiers

    This class is a parent class for different MRCs
    implemented in the library.
    It defines the different parameters and the layout.

    .. seealso:: For more information about MRC, one can refer to the
    following resources:

                    [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
                    Minimax Classification with
                    0-1 Loss and Performance Guarantees.
                    Advances in Neural Information Processing
                    Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

                    [2] `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
                    Generalized Maximum
                    Entropy for Supervised Classification.
                    arXiv preprint arXiv:2007.05447.
                    <https://arxiv.org/abs/2007.05447>`_

                    [3] `Bondugula, K., Mazuelas, S., & Pérez,
                    A. (2021). MRCpy: A Library for Minimax Risk Classifiers.
                    arXiv preprint arXiv:2108.01952.
                    <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    loss : `str`, default = '0-1'
        Type of loss function to use for the risk
        minimization.
        The options are 0-1 loss and logaritmic loss.
        '0-1'
            0-1 loss quantifies the probability of classification
            error at a certain example for a certain rule.
        'log'
            Log-loss quantifies the minus log-likelihood at a
            certain example for a certain rule.

    s : `float`, default = `0.3`
        Parameter that tunes the estimation of expected values
        of feature mapping function. It is used to calculate :math:`\lambda`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \\lambda = s * \\text{std}(\\phi(X,Y)) / \\sqrt{\\left| X \\right|}

        where (X,Y) is the dataset of training samples and their labels
        respectively and :math:`\\text{std}(\\phi(X,Y))` stands for
        standard deviation of :math:`\\phi(X,Y)` in the supervised
        dataset (X,Y).

    sigma : `str` or `float`, default = `scale`
        When given a string, it defines the type of heuristic to be used
        to calculate the scaling parameter `sigma` used in some feature
        mappings such as Random Fourier or ReLU features.
        For comparison its relation with parameter `gamma` used in
        other methods is :math:`\gamma=1/(2\sigma^2)`.
        When given a float, it is the value for the scaling parameter.

        'scale'
            Approximates `sigma` by
            :math:`\sqrt{\\frac{\\textrm{n_features} * \\textrm{var}(X)}{2}}`
            so that `gamma` is
            :math:`\\frac{1}{\\textrm{n_features} * \\textrm{var}(X)}`
            where `var` is the variance function.

        'avg_ann_50'
            Approximates `sigma` by the average distance to the
            :math:`50^{\\textrm{th}}`
            nearest neighbour estimated from 1000 samples of the dataset using
            the function `rff_sigma`.

    deterministic : `bool`, default = `True`
        Whether the prediction of the labels
        should be done in a deterministic way (given a fixed `random_state`
        in the case of using random Fourier or random ReLU features).

    random_state : `int`, RandomState instance, default = `None`
        Random seed used when 'fourier' and 'relu' options for feature mappings
        are used to produce the random weights.

    fit_intercept : `bool`, default = `True`
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    use_cvx : `bool`, default = `False`
        When set to True, use CVXpy library for the optimization
        instead of the subgradient methods.

    solver : `str`, default = 'MOSEK'
        Type of CVX solver to use for solving the problem.
        In some cases, one solver might not work,
        so you might need to change solver depending on the problem.

        'SCS'
            It uses Splitting Conic Solver (SCS).

        'ECOS'
            It uses Embedded Eonic Eolver (ECOS).

        'MOSEK'
            MOSEK is a commercial solver for which one might need to
            request for a license. A free license can be requested
            `here <https://www.mosek.com/products/academic-licenses/>`_.

    max_iters : `int`, default = `10000`
        Maximum number of iterations to use
        for finding the solution of optimization when
        using the subgradient approach.

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' or 'relu' feature mappings,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

        'relu'
            It uses Rectified Linear Unit (ReLU) features.
            See class `RandomReLUPhi`.

        'threshold'
            It uses Feature mappings obtained using a threshold.
            See class `ThresholdPhi`.

    **phi_kwargs : Additional parameters for feature mappings.
                Groups the multiple optional parameters
                for the corresponding feature mappings (`phi`).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument -
                `MRC(loss='log', phi='fourier', n_components=500)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : `bool`
        Whether the classifier is fitted i.e., the parameters are learnt.

    tau_ : `array`-like of shape (`n_features`)
        Mean estimates for the expectations of feature mappings.

    lambda_ : `array`-like of shape (`n_features`)
        Variance in the mean estimates for the expectations
        of the feature mappings.

    classes_ : `array`-like of shape (`n_classes`)
        Labels in the given dataset.
        If the labels Y are not given during fit
        i.e., tau and lambda are given as input,
        then this array is None.
    '''

    def __init__(self, loss='0-1', s=0.3,
                 deterministic=True, random_state=None,
                 fit_intercept=True, use_cvx=False,
                 solver='MOSEK', max_iters=10000, phi='linear',
                 sigma=None, **phi_kwargs):

        self.loss = loss
        self.s = s
        self.sigma = sigma
        self.deterministic = deterministic
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.use_cvx = use_cvx
        self.solver = solver
        self.max_iters = max_iters
        # Feature mapping and its parameters
        self.phi = phi
        self.phi_kwargs = phi_kwargs
        # Solver list for cvxpy
        self.solvers = ['MOSEK', 'SCS', 'ECOS']

    def fit(self, X, Y, X_=None):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

            `n_samples` is the number of training samples and
            `n_features` is the number of features.

        Y : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        X_ : array-like of shape (`n_samples2`, `n_dimensions`), default = None
            These instances are optional and
            when given, will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X, Y = check_X_y(X, Y, accept_sparse=True)

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
        n = X_opt.shape[0]
        if not_all_instances and n_max < n:
            n = n_max

        # Fit the MRC classifier
        self.minimax_risk(X_opt[:n], tau_, lambda_, n_classes)

        return self

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Solves the minimax risk optimization problem
        for the corresponding variant of MRC.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            Mean estimates
            for the expectations of feature mappings.

        lambda_ : `array`-like of shape (`n_features` * `n_classes`)
            Variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
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
        objective : `cvxpy` variable of `float` value
            Minimization objective function of the
            problem of the MRC.

        constraints : `array`-like of shape (`n_constraints`)
            Constraints for the MRC optimization.

        mu : `cvxpy` array of shape (number of featuers in `phi`)
            Parameters used in the optimization problem

        Returns
        -------
        mu_ : `array`-like of shape (number of featuers in `phi`)
            Value of the parameters
            corresponding to the optimum value of the objective function.

        objective_value : `float`
            Optimized objective value.

        '''

        # Solve the problem
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)

        mu_ = mu.value

        # if the solver could not find values of mu for the given solver
        if mu_ is None:

            # try with a different solver for solution
            for s in self.solvers:
                if s != self.solver:

                    # Solve the problem
                    prob.solve(solver=s, verbose=False)

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
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Testing instances for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        hy_x : `array`-like of shape (`n_samples`, `n_classes`)
            Conditional probabilities (:math:`p(y|x)`)
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
        Predicts classes for new instances using a fitted model.

        Returns the predicted classes for the given instances in `X`
        using the probabilities given by the function `predict_proba`.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : `array`-like of shape (`n_samples`)
            Predicted labels corresponding to the given instances.

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
