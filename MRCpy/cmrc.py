'''Marginally Constrained Minimax Risk Classification.'''

import itertools as it
import warnings

import cvxpy as cvx
import numpy as np
import scipy.special as scs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the MRC super class
from MRCpy import BaseMRC
from MRCpy.phi import \
    RandomReLUPhi, \
    ThresholdPhi


class CMRC(BaseMRC):
    '''
    Constrained Minimax Risk Classifier

    The class CMRC implements the method Minimimax Risk Classification
    (MRC) proposed in :ref:`[1] <ref1>`
    using the additional marginals constraints on the instances.
    It also implements two kinds of loss functions, namely 0-1 and log loss.

    This is a subclass of the super class BaseMRC.

    See :ref:`Examples of use` for futher applications of this class and
    its methods.

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

                    [3] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021).
                    MRCpy: A Library for Minimax Risk Classifiers.
                    arXiv preprint arXiv:2108.01952.
                    <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization. 0-1 loss
        quantifies the probability of classification error at a certain example
        for a certain rule. Log-loss quantifies the minus log-likelihood at a
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

    sigma : `str` or `float`, default = `sigma`
        When given a string, it defines the type of heuristic to be used
        to calculate the scaling parameter `sigma` used in some feature
        mappings such as Random Fourier or ReLU featuress.
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
        Used when 'fourier' and 'relu' options for feature mappings are used
        to produce the random weights.

    fit_intercept : `bool`, default = `True`
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    use_cvx : `bool`, default = `False`
        If True, use CVXpy library for the optimization
        instead of the subgradient or SGD methods.

    solver : `str`, default = 'MOSEK'
        The type of CVX solver to use for solving the problem.
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


    max_iters : `int`, default = `2000` or `30000`
        The maximum number of iterations to use
        for finding the solution of optimization when use_cvx=False.
        When None is given, default value is 30000 for Linear and RandomFourier
        feature mappings (optimization performed using SGD) and 2000 when
        using other feature mappings (optimization performed using nesterov
        subgradient approach).

    phi : `str` or `BasePhi` instance, default = 'linear'
        The type of feature mapping function to use for mapping the input data.
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
                for the corresponding feature mappings(`phi`).

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
    tau_ : `array`-like of shape (`n_features`) or `float`
        Mean estimates
        for the expectations of feature mappings.
    lambda_ : `array`-like of shape (`n_features`) or `float`
        Variance in the mean estimates
        for the expectations of the feature mappings.
    mu_ : `array`-like of shape (`n_features`) or `float`
        Parameters learnt by the optimization.
    params_ : `dict`
        Dictionary that stores the optimal points and best value of
        the function.

    Examples
    --------

    Simple example of using CMRC with default seetings: 0-1 loss and linear
    feature mapping. We first load the data and split it into train and
    test sets. We fit the model with the training samples using `fit` function.
    Then, we predict the class of some test samples with `predict`.
    We can also obtain the probabilities of each class with `predict_proba`.
    Finally, we calculate the score of the model over the test set
    using `score`.


    >>> from MRCpy import CMRC
    >>> from MRCpy.datasets import load_mammographic
    >>> from sklearn import preprocessing
    >>> from sklearn.model_selection import train_test_split
    >>> # Loading the dataset
    >>> X, Y = load_mammographic(return_X_y=True)
    >>> # Split the dataset into training and test instances
    >>> X_train, X_test, Y_train, Y_test =
    train_test_split(X, Y, test_size=0.2, random_state=0)
    >>> # Standarize the data
    >>> std_scale = preprocessing.StandardScaler().fit(X_train, Y_train)
    >>> X_train = std_scale.transform(X_train)
    >>> X_test = std_scale.transform(X_test)
    >>> # Fit the CMRC model
    >>> clf = CMRC().fit(X_train, Y_train)
    >>> # Prediction. The predicted values for the first 10 test instances are:
    >>> clf.predict(X_test[:10, :])
    [0 0 0 0 0 1 0 1 0 0]
    >>> # Predicted probabilities.
    >>> # The predicted probabilities for the first 10 test instances are:
    >>> clf.predict_proba(X_test[:10, :])
    [[0.62919573 0.37080427]
     [1.         0.        ]
     [0.95104266 0.04895734]
     [1.         0.        ]
     [0.99047735 0.00952265]
     [0.         1.        ]
     [1.         0.        ]
     [0.12378713 0.87621287]
     [1.         0.        ]
     [0.62290253 0.37709747]]
    >>> # Calculate the score of the predictor
    >>> # (mean accuracy on the given test data and labels)
    >>> clf.score(X_test, Y_test)
    0.8247422680412371
    '''

    # Redefining the init function
    # to reduce the default number for maximum iterations.
    # In case of CMRC, the convergence is observed to be fast
    # and hence less iterations should be sufficient
    def __init__(self, loss='0-1', s=0.3,
                 deterministic=True, random_state=None,
                 fit_intercept=True, use_cvx=False,
                 solver='SCS', max_iters=None, phi='linear',
                 stepsize='decay', **phi_kwargs):
        if max_iters is None:
            if phi == 'linear' or phi == 'fourier':
                max_iters = 100000
            else:
                max_iters = 2000
        self.stepsize = stepsize
        super().__init__(loss=loss,
                         s=s,
                         deterministic=deterministic,
                         random_state=random_state,
                         fit_intercept=fit_intercept,
                         use_cvx=use_cvx,
                         solver=solver,
                         max_iters=max_iters,
                         phi=phi, **phi_kwargs)

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

        if X_ is None:
            if self.phi == 'linear' or self.phi == 'fourier':
                super().fit(X, Y, X)
            else:
                super().fit(X, Y)
        else:
            super().fit(X, Y, X_)
        return self

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Solves the marginally constrained minimax risk
        optimization problem for
        different types of loss (0-1 and log loss).
        When use_cvx=False, it uses SGD optimization for linear and random
        fourier feature mappings and nesterov subgradient approach for
        the rest.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            The mean estimates
            for the expectations of feature mappings.

        lambda_ : `array`-like of shape (`n_features` * `n_classes`)
            The variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
            Number of labels in the dataset.

        Returns
        -------
        self :
            Fitted estimator

        '''

        # Set the parameters for the optimization
        self.n_classes = n_classes
        self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)
        self.lambda_ = check_array(lambda_, accept_sparse=True,
                                   ensure_2d=False)
        phi = self.phi.eval_x(X)

        # Constants
        n = phi.shape[0]
        m = phi.shape[2]

        # Supress the depreciation warnings
        warnings.simplefilter('ignore')

        # In case of 0-1 loss, learn constraints using the phi
        # These constraints are used in the optimization instead of phi

        if self.loss == '0-1':
            # Summing up the phi configurations
            # for all possible subsets of classes for each instance
            F = np.vstack((np.sum(phi[:, S, ], axis=1)
                           for numVals in range(1, self.n_classes + 1)
                           for S in it.combinations(np.arange(self.n_classes),
                                                    numVals)))

            # Compute the corresponding length of the subset of classes
            # for which sums computed for each instance
            cardS = np.arange(1, self.n_classes + 1).\
                repeat([n * scs.comb(self.n_classes, numVals)
                        for numVals in np.arange(1,
                        self.n_classes + 1)])

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)

            if self.loss == '0-1':
                # Constraints in case of 0-1 loss function

                M = F / (cardS[:, np.newaxis])
                h = 1 - (1 / cardS)

                # Calculate the psi function and
                # add it to the objective function
                # First we calculate the all possible values of psi
                # for all the points
                psi = M @ mu + h
                sum_psi = 0
                for i in range(n):
                    # Get psi for each data point and
                    # add the min value to objective
                    psi_xi = psi[np.arange(i, psi.shape[0], n)]
                    sum_psi = sum_psi + (1 / n) * cvx.max((psi_xi))

            elif self.loss == 'log':
                # Constraints in case of log loss function
                sum_psi = 0
                for i in range(n):
                    sum_psi = sum_psi + \
                        (1 / n) * cvx.log_sum_exp(phi[i, :, :] @ mu)

            # Objective function
            objective = cvx.Minimize(self.lambda_ @ cvx.abs(mu) -
                                     self.tau_ @ mu + sum_psi)

            self.mu_, self.obj_value = \
                self.try_solvers(objective, None, mu)

        elif not self.use_cvx:

            if self.loss == '0-1':
                # Define the objective function and
                # the gradient for the 0-1 loss function.
                M = F / (cardS[:, np.newaxis])
                h = 1 - (1 / cardS)

                # Function to calculate the psi subobjective
                # to be added to the objective function.
                def f_(mu):
                    # First we calculate the all possible values of psi
                    # for all the points.
                    psi = M @ mu + h
                    idx = []

                    for i in range(n):
                        # Get psi for each data point
                        # and return the max value over all subset
                        # and its corresponding index
                        xi_subsetInd = np.arange(i, psi.shape[0], n)
                        idx.append(xi_subsetInd[np.argmax(psi[xi_subsetInd])])
                    return (1 / n) * np.sum(psi[idx]), idx

                if isinstance(self.phi, RandomReLUPhi) or \
                   isinstance(self.phi, ThresholdPhi):
                    # Use the subgradient approach for the convex optimization
                    # The subgradient of the psi subobjective
                    # for all the datapoints
                    def g_(mu, idx):
                        return (1 / n) * np.sum(M.transpose()[:, idx], axis=1)
                else:
                    # Use SGD for the convex optimization
                    # Subgradient of the subobjective for one point
                    def g_(mu, sample_id):
                        xi_subsetInd = np.arange(sample_id, M.shape[0], n)
                        psi = M[xi_subsetInd] @ mu + h[xi_subsetInd]
                        idx = xi_subsetInd[np.argmax(psi)]
                        return M.transpose()[:, idx].flatten()

            elif self.loss == 'log':
                # Define the objective function and
                # the gradient for the log loss function.

                # The psi subobjective for all the datapoints
                def f_(mu):
                    return ((1 / n) *
                            np.sum(scs.logsumexp((phi @ mu), axis=1)),
                            None)

                if isinstance(self.phi, RandomReLUPhi) or \
                   isinstance(self.phi, ThresholdPhi):
                    # Use the subgradient approach for the convex optimization
                    # The subgradient of the psi subobjective
                    # for all the datapoints
                    def g_(mu, idx):
                        expPhi = np.exp(phi @ mu)[:, np.newaxis, :]
                        return (1 / n) *\
                            (np.sum(((expPhi @ phi)[:, 0, :] /
                                     np.sum(expPhi, axis=2)).transpose(),
                                    axis=1))

                else:
                    # Use SGD for the convex optimization
                    # Subgradient of the subobjective for one point
                    def g_(mu, sample_id):
                        expPhi = np.exp(phi[sample_id, :, :] @ mu
                                        )[np.newaxis, np.newaxis, :]
                        return (np.sum(((expPhi @ phi[sample_id, :, :])
                                        [:, 0, :] /
                                        np.sum(expPhi, axis=2)).transpose(),
                                       axis=1))

            if isinstance(self.phi, RandomReLUPhi) or \
               isinstance(self.phi, ThresholdPhi):
                self.params_ = \
                    self.nesterov_optimization(m, None, f_, g_)
            else:
                self.params_ = \
                    self.SGD_optimization(m, n, None, f_, g_)

            self.mu_ = self.params_['mu']
        self.is_fitted_ = True

        return self

    def nesterov_optimization(self, m, params_, f_, g_):
        '''
        Solution of the CMRC convex optimization(minimization)
        using the Nesterov accelerated approach.

        .. seealso:: [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                            The Strength of Nesterov’s Extrapolation
                            in the Individual Convergence of Nonsmooth
                            Optimization. IEEE transactions on
                            neural networks and learning systems,
                            31(7), 2557-2568.
                            <https://ieeexplore.ieee.org/document/8822632>`_

        Parameters
        ----------
        m : `int`
            Length of the feature mapping vector
        params_ : `dict`
            A dictionary of parameters values
            obtained from the previous call to fit
            used as the initial values for the current optimization
            when warm_start is True.
        f_ : a lambda function/ function of the form - `f_(mu)`
            It is expected to be a lambda function or a function
            calculating a part of the objective function
            depending on the type of loss function chosen
            by taking the parameters(mu) of the optimization as input.
        g_ : a lambda function of the form - `g_(mu, idx)`
            It is expected to be a lambda function
            calculating the part of the subgradient of the objective function
            depending on the type of the loss function chosen.
            It takes the as input -
            parameters (mu) of the optimization and
            the indices corresponding to the maximum value of subobjective
            for a given subset of Y (set of labels).

        Return
        ------
        new_params_ : `dict`
            Dictionary containing optimized values: mu (`array`-like,
            shape (`m`,)) - parameters corresponding to the optimized
            function value, f_best_value (`float` - optimized value of the
            function in consideration, w_k and w_k_prev (`array`-like,
            shape (`m`,)) - parameters corresponding to the last iteration.
        '''

        # Initial values for the parameters
        theta_k = 1
        theta_k_prev = 1

        y_k = np.zeros(m, dtype=np.float)
        w_k = np.zeros(m, dtype=np.float)
        w_k_prev = np.zeros(m, dtype=np.float)

        # Setting initial values for the objective function and other results
        psi, idx = f_(y_k)
        f_best_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + psi
        mu = y_k

        # Iteration for finding the optimal values
        # using Nesterov's extrapolation
        for k in range(1, (self.max_iters + 1)):
            y_k = w_k + theta_k * ((1 / theta_k_prev) - 1) * (w_k - w_k_prev)

            # Calculating the subgradient of the objective function at y_k
            psi, idx = f_(y_k)
            g_0 = self.lambda_ * np.sign(y_k) - self.tau_ + g_(y_k, idx)

            # Update the parameters
            theta_k_prev = theta_k
            theta_k = 2 / (k + 1)
            alpha_k = 1 / (np.power((k + 1), (3 / 2)))

            # Calculate the new points
            w_k_prev = w_k
            w_k = y_k - alpha_k * g_0

            # Check if there is an improvement
            # in the value of the objective function
            f_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + psi
            if f_value < f_best_value:
                f_best_value = f_value
                mu = y_k

        # Check for possible improvement of the objective valu
        # for the last generated value of w_k
        psi, idx = f_(w_k)
        f_value = self.lambda_ @ np.abs(w_k) - self.tau_ @ w_k + psi

        if f_value < f_best_value:
            f_best_value = f_value
            mu = w_k

        # Return the optimized values in a dictionary
        new_params_ = {'w_k': w_k,
                       'w_k_prev': w_k_prev,
                       'mu': mu,
                       'best_value': f_best_value,
                       }

        return new_params_

    def SGD_optimization(self, m, n, params_, f_, g_):
        '''
        Solution of the CMRC convex optimization(minimization)
        using SGD approach.

        Parameters
        ----------
        m : `int`
            Length of the feature mapping vector
        n : `int`
            Number of samples used for optimization
        params_ : `dict`
            A dictionary of parameters values
            obtained from the previous call to fit
            used as the initial values for the current optimization
            when warm_start is True.
        f_ : a lambda function/ function of the form - `f_(mu)`
            It is expected to be a lambda function or a function
            calculating a part of the objective function
            depending on the type of loss function chosen
            by taking the parameters(mu) of the optimization as input.
        g_ : a lambda function of the form - `g_(mu, idx)`
            It is expected to be a lambda function
            calculating the part of the subgradient of the objective function
            depending on the type of the loss function chosen.
            It takes the as input -
            parameters (mu) of the optimization and
            the indices corresponding to the maximum value of subobjective
            for a given subset of Y (set of labels).

        Return
        ------
        new_params_ : `dict`
            Dictionary containing optimized values: mu and w_k (`array`-like,
            shape (`m`,)) - parameters corresponding to the last iteration,
            best_value (`float` - optimized value of the
            function in consideration.
        '''

        # Initial values for points
        w_k = np.zeros(m, dtype=np.float)

        # Setting initial values for the objective function and other results

        sample_id = 0
        epoch_id = 0
        for k in range(1, (self.max_iters + 1)):

            g_0 = self.lambda_ * np.sign(w_k) - self.tau_ + g_(w_k, sample_id)

            if self.stepsize == 'decay':
                stepsize = 0.01 * (1 / 1 + 0.01 * epoch_id)
            elif type(self.stepsize) == float:
                stepsize = self.stepsize
            else:
                raise ValueError('Unexpected stepsize ... ')

            w_k = w_k - stepsize * g_0

            sample_id += 1
            epoch_id += sample_id // n
            sample_id = sample_id % n

        psi, idx = f_(w_k)
        f_value = self.lambda_ @ np.abs(w_k) - self.tau_ @ w_k + psi
        mu = w_k

        # Return the optimized values in a dictionary
        new_params_ = {'w_k': w_k,
                       'mu': mu,
                       'best_value': f_value,  # actually last value
                       }

        return new_params_

    def predict_proba(self, X):
        '''
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
            The conditional probabilities (p(y|x))
            corresponding to each class.
        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        n = X.shape[0]

        phi = self.phi.eval_x(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Summing up the phi configurations
            # for all possible subsets of classes for each instance
            F = np.vstack((np.sum(phi[:, S, ], axis=1)
                           for numVals in range(1, self.n_classes + 1)
                           for S in it.combinations(np.arange(self.n_classes),
                                                    numVals)))

            # Compute the corresponding length of the subset of classes
            # for which sums computed for each instance
            cardS = np.arange(1, self.n_classes + 1).\
                repeat([n * scs.comb(self.n_classes, numVals)
                        for numVals in np.arange(1,
                        self.n_classes + 1)])

            # Compute psi
            psi = np.zeros(n)

            # First we calculate the all possible values of psi
            # for all the points
            psi_arr = (np.ones(cardS.shape[0]) -
                       (F @ self.mu_ + cardS)) / cardS

            for i in range(n):
                # Get psi values for each data point and find the min value
                psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
                psi[i] = np.min(psi_arr_xi)

            # Conditional probabilities
            hy_x = np.clip(np.ones((n, self.n_classes)) +
                           np.dot(phi, self.mu_) +
                           np.tile(psi, (self.n_classes, 1)).transpose(),
                           0., None)

            # normalization constraint
            c = np.sum(hy_x, axis=1)
            # check when the sum is zero
            zeros = np.isclose(c, 0)
            c[zeros] = 1
            hy_x[zeros, :] = 1 / self.n_classes
            c = np.tile(c, (self.n_classes, 1)).transpose()
            hy_x = hy_x / c

            # Set the approach for prediction to non-deterministic
            # if not provided by user.
            if self.deterministic is None:
                self.deterministic = False

        elif self.loss == 'log':
            # Constraints in case of log loss function

            v = np.dot(phi, self.mu_)

            # Normalizing conditional probabilities
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes)).transpose()
            hy_x = np.reciprocal(hy_x)

            # Set the approach for prediction to deterministic
            # if not provided by user.
            if self.deterministic is None:
                self.deterministic = True

        return hy_x
