'''Minimax Risk Classification.'''

import itertools as it
import warnings

import cvxpy as cvx
import numpy as np
import scipy.special as scs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the MRC super class
from MRCpy import BaseMRC


class MRC(BaseMRC):
    '''
    Minimax Risk Classifier

    The class MRC implements the method Minimimax Risk Classification (MRC)
    proposed in :ref:`[1] <ref1>`
    using the default constraints. It implements two kinds of loss functions,
    namely 0-1 and log loss.

    The method MRC approximates the optimal classification rule by an
    optimization problem of the form

    .. math:: \\mathcal{P}_{\\text{MRC}}:
        \\min_{h\\in T(\\mathcal{X},\\mathcal{Y})}
        \\max_{p\\in\\mathcal{U}} \\ell(h,p)

    where we consider an uncertainty set :math:`\\mathcal{U}` of potential
    probabilities.
    These untertainty sets of distributions are given by constraints on the
    expectations of a vector-valued function :math:`\\phi : \\mathcal{X}
    \\times \\mathcal{Y} \\rightarrow \\mathbb{R}^m` referred to as feature
    mapping.


    This is a subclass of the super class `BaseMRC`.

    See :ref:`Examples of use` for futher applications of this class and its
    methods.

    .. _ref1:
    .. seealso:: For more information about MRC, one can refer to the following
    resources:

                    [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
                    Minimax Classification with
                    0-1 Loss and Performance Guarantees. Advances in Neural
                    Information Processing
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

        where (X,Y) is the dataset of training samples and their
        labels respectively and
        :math:`\\text{std}(\\phi(X,Y))` stands for standard deviation
        of :math:`\\phi(X,Y)` in the supervised dataset (X,Y).

    sigma : `str` or `float`, default = `scale`
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
        Type of CVX solver to be used for solving the optimization problem.
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
                for the corresponding feature mappings(`phi`).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument
                `MRC(loss='log', phi='fourier', n_components=500)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : `bool`
        Whether the classifier is fitted i.e., the parameters are learnt
        or not.

    tau_ : `array`-like of shape (`n_features`) or `float`
        Mean estimates
        for the expectations of feature mappings.

    lambda_ : `array`-like of shape (`n_features`) or `float`
        Variance in the mean estimates
        for the expectations of the feature mappings.

    mu_ : `array`-like of shape (`n_features`) or `float`
        Parameters learnt by the optimization.

    nu_ : `float`
        Parameter learnt by the optimization.

    mu_l_ : `array`-like of shape (`n_features`) or `float`
        Parameters learnt by solving the lower bound optimization of MRC.

    upper_ : `float`
        Optimized upper bound of the MRC classifier.

    lower_ : `float`
        Optimized lower bound of the MRC classifier.

    upper_params_ : `dict`
        Dictionary that stores the optimal points and best value
        for the upper bound of the function.

    params_ : `dict`
        Dictionary that stores the optimal points and best value
        for the lower bound of the function.


    Examples
    --------

    Simple example of using MRC with default seetings: 0-1 loss and linear
    feature mapping.
    We first load the data and split it into train and test sets.
    We fit the model with the training samples using `fit` function.
    Then, we predict the class of some test samples with `predict`.
    We can also obtain the probabilities of each class with `predict_proba`.
    Finally, we calculate the score of the model over the test set
    using `score`.


    >>> from MRCpy import MRC
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
    >>> # Fit the MRC model
    >>> clf = MRC().fit(X_train, Y_train)
    >>> # Prediction. The predicted values for the first 10 test instances are:
    >>> clf.pre (X_test[:10, :])
    [1 0 0 0 0 1 0 1 0 0]
    >>> # Predicted probabilities.
    >>> # The predicted probabilities for the first 10 test instances are:
    >>> clf.predict_proba(X_test[:10, :])
    [[2.80350905e-01 7.19649095e-01]
    [9.99996406e-01 3.59370941e-06]
    [8.78592959e-01 1.21407041e-01]
    [8.78593719e-01 1.21406281e-01]
    [8.78595619e-01 1.21404381e-01]
    [1.58950511e-01 8.41049489e-01]
    [9.99997060e-01 2.94047920e-06]
    [4.01753510e-01 5.98246490e-01]
    [8.78595322e-01 1.21404678e-01]
    [6.35793570e-01 3.64206430e-01]]
    >>> # Calculate the score of the predictor
    >>> # (mean accuracy on the given test data and labels)
    >>> clf.score(X_test, Y_test)
    0.7731958762886598

    '''

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Solves the minimax risk problem
        for different types of loss (0-1 and log loss).
        The solution of the default MRC optimization
        gives the upper bound of the error.

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

        # Set the parameters for the optimization
        self.n_classes = n_classes
        self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)
        self.lambda_ = check_array(lambda_, accept_sparse=True,
                                   ensure_2d=False)
        phi = self.phi.eval_x(X)

        phi = np.unique(phi, axis=0)

        # Constants
        m = phi.shape[2]
        n = phi.shape[0]

        # Save the phi configurations for finding the lower bounds
        self.lowerPhiConfigs = phi

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

            M = F / (cardS[:, np.newaxis])
            h = 1 - (1 / cardS)

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)

            if self.loss == '0-1':

                def neg_nu(mu):
                    return cvx.max(M @ mu + h)

            elif self.loss == 'log':
                numConstr = phi.shape[0]

                def neg_nu(mu):
                    return cvx.max(cvx.hstack(cvx.log_sum_exp(phi[i, :, :] @
                                                              mu)
                                              for i in range(numConstr)))

            else:
                raise ValueError('The given loss function is not available ' +
                                 'for this classifier')

            # Objective function
            objective = cvx.Minimize(self.lambda_ @ cvx.abs(mu) -
                                     self.tau_ @ mu +
                                     neg_nu(mu))

            self.mu_, self.upper_ = self.try_solvers(objective, None, mu)
            self.nu_ = (-1) * (neg_nu(self.mu_).value)

        elif not self.use_cvx:
            # Use the subgradient approach for the convex optimization of MRC

            if self.loss == '0-1':
                M_t = M.transpose()

                # Define the subobjective function and
                # its gradient for the 0-1 loss function.
                def f_(mu):
                    return M @ mu + h

                def g_(mu, idx):
                    return M_t[:, idx]

                # Calculate the upper bound
                self.upper_params_ = self.nesterov_optimization_minimized(M, h)

            elif self.loss == 'log':

                # Define the subobjective function and
                # its gradient for the log loss function.
                def f_(mu):
                    return scs.logsumexp((phi @ mu), axis=1)

                def g_(mu, idx):
                    phi_xi = phi[idx, :, :]
                    expPhi_xi = np.exp(phi_xi @ mu)
                    return (expPhi_xi @ phi_xi).transpose() / np.sum(expPhi_xi)

                # Calculate the upper bound
                self.upper_params_ = self.nesterov_optimization(m, f_, g_)

            else:
                raise ValueError('The given loss function is not available ' +
                                 'for this classifier')

            self.mu_ = self.upper_params_['mu']
            self.nu_ = self.upper_params_['nu']
            self.upper_ = self.upper_params_['best_value']

        self.is_fitted_ = True
        return self

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        return self.upper_

    def get_lower_bound(self):
        '''
        Obtains the lower bound on the expected loss for the fitted classifier.

        Returns
        -------
        lower : `float`
            Lower bound of the error for the fitted classifier.
        '''

        # Classifier should be fitted to obtain the lower bound
        check_is_fitted(self, "is_fitted_")

        # Learned feature mappings
        phi = self.lowerPhiConfigs

        # Variables
        n = phi.shape[0]
        m = phi.shape[2]

        if self.loss == '0-1':

            # To define the objective function and
            # the gradient for the 0-1 loss function.
            # epsilon
            eps = np.clip(1 + phi @ self.mu_ + self.nu_, 0, None)
            c = np.sum(eps, axis=1)
            zeros = np.isclose(c, 0)
            c[zeros] = 1
            eps[zeros, :] = 1 / self.n_classes
            eps = eps / (c[:, np.newaxis])
            # Using negative of epsilon
            # for the nesterov accelerated optimization
            eps = eps - 1

            # Reshape it for the optimization function
            eps = eps.reshape((n * self.n_classes,))

        elif self.loss == 'log':

            # To define the objective function and
            # the gradient for the log loss function.
            # Using negative of epsilon
            # for the nesterov accelerated optimization
            eps = phi @ self.mu_ - \
                scs.logsumexp(phi @ self.mu_, axis=1)[:, np.newaxis]
            eps = eps.reshape((n * self.n_classes,))

        else:
            raise ValueError('The given loss function is not available ' +
                             'for this classifier')

        phi = phi.reshape((n * self.n_classes, m))

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC

            low_mu = cvx.Variable(m)

            # Objective function
            objective = cvx.Minimize(self.lambda_ @ cvx.abs(low_mu) -
                                     self.tau_ @ low_mu +
                                     cvx.max(phi @ low_mu + eps))

            self.mu_l_, self.lower_ = \
                self.try_solvers(objective, None, low_mu)

            # Maximize the function
            self.lower_ = (-1) * self.lower_

        elif not self.use_cvx:
            # Use the subgradient approach for the convex optimization of MRC

            # Defining the partial objective and its gradient.
            def f_(mu):
                return phi @ mu + eps

            def g_(mu, idx):
                return phi.transpose()[:, idx]

            # Lower bound
            self.lower_params_ = \
                self.nesterov_optimization(m, f_, g_)

            self.mu_l_ = self.lower_params_['mu']
            self.lower_ = self.lower_params_['best_value']

            # Maximize the function
            # as the nesterov optimization gives the minimum
            self.lower_ = -1 * self.lower_

        return self.lower_

    def nesterov_optimization(self, m, f_, g_):
        '''
        Solution of the MRC convex optimization (minimization)
        using the Nesterov accelerated approach.

        .. seealso:: [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                            The Strength of Nesterov’s Extrapolation in
                            the Individual Convergence of Nonsmooth
                            Optimization. IEEE transactions on
                            neural networks and learning systems,
                            31(7), 2557-2568.
                            <https://ieeexplore.ieee.org/document/8822632>`_

        Parameters
        ----------
        m : `int`
            Length of the feature mapping vector

        f_ : a lambda function of the form - f_(mu)
            Lambda function
            calculating a part of the objective function
            depending on the type of loss function chosen
            by taking the parameters (mu) of the optimization as input.

        g_ : a lambda function of the form - g_(mu, idx)
            Lambda function
            calculating the part of the subgradient of the objective function
            depending on the type of the loss function chosen.
            It takes the as input: parameters (mu) of the optimization and
            the index corresponding to the maximum value of data matrix
            obtained from the instances.

        Returns
        -------
        new_params_ : `dict`
            Dictionary that stores the optimal points
            (`w_k`: `array-like` shape (`m`,), `w_k_prev`: `array-like`
             shape (`m`,)) where `m`is the length of the feature
            mapping vector and best value
            for the upper bound (`best_value`: `float`) of the function and
            the parameters corresponding to the optimized function value
            (`mu`: `array-like` shape (`m`,),
            `nu`: `float`).
        '''

        # Initial values for the parameters
        theta_k = 1
        theta_k_prev = 1

        # Initial values for points
        y_k = np.zeros(m, dtype=np.float)
        w_k = np.zeros(m, dtype=np.float)
        w_k_prev = np.zeros(m, dtype=np.float)

        # Setting initial values for the objective function and other results
        v = f_(y_k)
        mnu = np.max(v)
        f_best_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + mnu
        mu = y_k
        nu = -1 * mnu

        # Iteration for finding the optimal values
        # using Nesterov's extrapolation
        for k in range(1, (self.max_iters + 1)):
            y_k = w_k + theta_k * ((1 / theta_k_prev) - 1) * (w_k - w_k_prev)

            # Calculating the subgradient of the objective function at y_k
            v = f_(y_k)
            idx = np.argmax(v)
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
            mnu = v[idx]
            f_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + mnu
            if f_value < f_best_value:
                f_best_value = f_value
                mu = y_k
                nu = -1 * mnu

        # Check for possible improvement of the objective value
        # for the last generated value of w_k
        v = f_(w_k)
        mnu = np.max(v)
        f_value = self.lambda_ @ np.abs(w_k) - self.tau_ @ w_k + mnu

        if f_value < f_best_value:
            f_best_value = f_value
            mu = w_k
            nu = -1 * mnu

        # Return the optimized values in a dictionary
        new_params_ = {'w_k': w_k,
                       'w_k_prev': w_k_prev,
                       'mu': mu,
                       'nu': nu,
                       'best_value': f_best_value,
                       }

        return new_params_

    def nesterov_optimization_minimized(self, M, h):
        '''
        Solution of the MRC convex optimization (minimization)
        using an optimized version of the Nesterov accelerated approach.

        .. seealso::         [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                                The Strength of Nesterov’s Extrapolation in
                                the Individual Convergence of Nonsmooth
                                Optimization. IEEE transactions on
                                neural networks and learning systems,
                                31(7), 2557-2568.
                                <https://ieeexplore.ieee.org/document/8822632>`_

        Parameters
        ----------
        M : `array`-like of shape (:math:`m_1`, :math:`m_2`)
            Where :math:`m_1` is approximately
            :math:`(2^{\\textrm{n_classes}}-1) *
            \\textrm{min}(5000,\\textrm{len}(X))`,
            where the second factor is the number of training samples used for
            solving the optimization problem.

        h : `array`-like of shape (:math:`m_1`,)
            Where :math:`m_1` is approximately
            :math:`(2^{\\textrm{n_classes}}-1) *
            \\textrm{min}(5000,\\textrm{len}(X))`,
            where the second factor is the number of training samples used for
            solving the optimization problem.

        Returns
        ------
        new_params_ : `dict`
            Dictionary that stores the optimal points
            (`w_k`: `array-like` shape (`m`,), `w_k_prev`: `array-like`
             shape (`m`,)), where `m` is the length of the feature
            mapping vector, and best value
            for the upper bound (`best_value`: `float`) of the function and
            the parameters corresponding to the optimized function value
            (`mu`: `array-like` shape (`m`,),
            `nu`: `float`).
        '''
        h = h.reshape((len(h), 1))
        M_t = M.T
        Mc = M @ - self.tau_
        n, m = M.shape
        lambda_col = self.lambda_.reshape((m, 1))
        MD = np.multiply(M, self.lambda_)
        MD_sum = np.zeros((n, 1))

        if n * n > (1024) ** 3:
            large_dimension = True
            # print("LARGE DIMENSION")
            MMtc = []
            Mtc = np.zeros((m, n))
        else:
            large_dimension = False
            # print("SMALL DIMENSION")
            MMt = M @ M_t
            MMtc = Mc.reshape((n, 1)) + MMt
            Mtc = (-self.tau_).reshape((m, 1)) + M_t

        # Initial values for points
        y_k = np.zeros((m, 1))
        w_k = np.zeros((m, 1))

        theta_k = 1
        mnu = np.max(M @ y_k + h)  # -h en matlab
        f_value = float(-np.dot(self.tau_, y_k) +
                        np.dot(self.lambda_, abs(y_k)) + mnu)
        upper = f_value
        mu = y_k
        nu = -mnu
        My = M @ y_k + h  # -h en matlab
        Mw = M @ w_k + h  # -h en matlab

        alpha_k = 0
        signos = np.sign(y_k)
        signos_old = signos
        delta = signos - signos_old
        index1 = np.where(delta != 0)[0]
        idx = 1
        step_k = 0

        if large_dimension:
            index = (np.ones(n) * -1).astype(int)
            idx = 0
            MMtc = np.zeros((n, 1))
            for k in range(1, self.max_iters + 1):
                # calculate My-h
                MD_sum = MD_sum + MD[:, index1] @ delta[index1]
                Mg0 = MD_sum + MMtc[:, [index[idx]]]
                Mw_prev = Mw
                Mw = My - alpha_k * Mg0

                My = Mw + step_k * (Mw - Mw_prev)

                idx = np.argmax(My)
                mnu = My[idx]

                if index[idx] == -1:
                    update_MMtc = Mc + M @ (M_t[:, idx])
                    MMtc = np.concatenate([MMtc, update_MMtc.reshape(
                        (len(update_MMtc), 1))], axis=1)
                    Mtc[:, idx] = -self.tau_ + M_t[:, idx]
                    index[idx] = np.shape(MMtc)[1] - 1

                signos_old = signos
                signos = np.sign(y_k)
                delta = signos - signos_old
                index1 = np.where(delta != 0)[0]

                g0 = np.multiply(signos, lambda_col) + Mtc[:, [idx]]
                f_value = float(-np.dot(self.tau_, np.asarray(y_k)) +
                                np.dot(self.lambda_, np.asarray(abs(y_k)))
                                + mnu)

                if f_value < upper:
                    upper = f_value
                    mu = y_k
                    nu = -mnu

                theta_k_prev = theta_k
                theta_k = 2 / (k + 1)
                alpha_k = 1 / np.power((k + 1), (3 / 2))
                w_k_prev = w_k
                w_k = y_k - alpha_k * g0
                step_k = theta_k * ((1 / theta_k_prev) - 1)
                y_k = w_k + step_k * (w_k - w_k_prev)
        else:
            for k in range(1, self.max_iters + 1):
                # calculate My-h
                MD_sum = MD_sum + MD[:, index1] @ delta[index1]
                Mg0 = MD_sum + MMtc[:, [idx]]
                Mw_prev = Mw
                Mw = My - alpha_k * Mg0
                My = Mw + step_k * (Mw - Mw_prev)

                idx = np.argmax(My)
                mnu = float(My[idx])

                signos_old = signos
                signos = np.sign(y_k)
                delta = signos - signos_old
                index1 = np.where(delta != 0)[0]  # where delta!=0
                g0 = np.multiply(signos, lambda_col) + Mtc[:, [idx]]
                f_value = float(-np.dot(self.tau_, np.asarray(y_k)) +
                                np.dot(self.lambda_, np.asarray(abs(y_k)))
                                + mnu)

                if f_value < upper:
                    upper = f_value
                    mu = y_k
                    nu = -mnu

                theta_k_prev = theta_k
                theta_k = 2 / (k + 1)
                alpha_k = 1 / np.power((k + 1), (3 / 2))
                w_k_prev = w_k
                w_k = y_k - alpha_k * g0
                step_k = theta_k * ((1 / theta_k_prev) - 1)
                y_k = w_k + step_k * (w_k - w_k_prev)

        # Check for possible improvement of the objective value
        # for the last generated value of w_k
        mnu = float(max(M @ w_k + h))
        f_value = float(-np.dot(self.tau_, np.asarray(w_k)) +
                        np.dot(self.lambda_, np.asarray(abs(w_k))) + mnu)

        if f_value < upper:
            upper = f_value
            mu = w_k
            nu = -mnu

        mu = np.array(mu).flatten()

        # Return the optimized values in a dictionary
        new_params_ = {'w_k': w_k,
                       'w_k_prev': w_k_prev,
                       'mu': mu,
                       'nu': nu,
                       'best_value': upper,
                       }
        return new_params_

    def predict_proba(self, X):
        '''
        Conditional probabilities corresponding to each class
        for each unlabeled input instance

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Testing instances for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        hy_x : `ndarray` of shape (`n_samples`, `n_classes`)
            Probabilities :math:`(p(y|x))` corresponding to the predictions
            for each class.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        phi = self.phi.eval_x(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Unnormalized conditional probabilityes
            hy_x = np.clip(1 + np.dot(phi, self.mu_) + self.nu_, 0., None)

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
