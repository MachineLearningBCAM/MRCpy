"""
Marginally Constrained Minimax Risk Classification. Copyright (C) 2021 Kartheek Bondugula

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
"""

import itertools as it
import warnings

import cvxpy as cvx
import numpy as np
import scipy.special as scs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the MRC super class
from MRCpy import BaseMRC
from MRCpy.solvers.cvx import *
from MRCpy.solvers.adam import *
from MRCpy.solvers.sgd import *
from MRCpy.solvers.nesterov import *
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi


class CMRC(BaseMRC):
    '''
    Constrained Minimax Risk Classifier

    The class CMRC implements the method Minimimax Risk Classifiers
    with fixed marginal distributions proposed in :ref:`[1] <ref1>`
    using the additional marginals constraints on the instances.
    It also implements two kinds of loss functions, namely 0-1 and log loss.

    This is a subclass of the super class BaseMRC.

    See :ref:`Examples of use` for futher applications of this class and
    its methods.

    .. seealso:: For more information about CMRC, one can refer to the
        following resources:

                    [1] `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
                    Generalized Maximum
                    Entropy for Supervised Classification.
                    arXiv preprint arXiv:2007.05447.
                    <https://arxiv.org/abs/2007.05447>`_

                    [2] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021).
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

    solver : {‘cvx’, 'sgd', 'adam'}, default = ’adam’
        Method to use in solving the optimization problem. 
        Default is ‘cvx’. To choose a solver,
        you might want to consider the following aspects:

        ’cvx’
            Solves the optimization problem using the CVXPY library.
            Obtains an accurate solution while requiring more time
            than the other methods. 
            Note that the library uses the GUROBI solver in CVXpy for which
            one might need to request for a license.
            A free license can be requested `here 
            <https://www.gurobi.com/academia/academic-program-and-licenses/>`_

        ’sgd’
            Solves the optimization using stochastic gradient descent.
            The parameters `max_iters`, `stepsize` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.

        ’adam’
            Solves the optimization using
            stochastic gradient descent with adam (adam optimizer).
            The parameters `max_iters`, `alpha` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.

    alpha : `float`, default = `0.001`
        Learning rate for ’adam’ solver.

    stepsize : `float` or {‘decay’}, default = ‘decay’
        Learning rate for ’grad’ solver. The default is ‘decay’, that is,
        the learning rate decreases with the number of epochs of 
        stochastic gradient descent.

    mini_batch_size : `int`, default = `1` or `32`
        The size of the batch to be used for computing the gradient
        in case of stochastic gradient descent and adam optimizer.
        In case of stochastic gradient descent, the default is 1, and
        in case of adam optimizer, the default is 32.

    max_iters : `int`, default = `100000` or `5000` or `2000`
        The maximum number of iterations to use in case of
        ’grad’ or ’adam’ solver.
        The default value is
        100000 for ’grad’ solver and
        5000 for ’adam’ solver and 
        2000 for nesterov's gradient descent.

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
    def __init__(self,
                 loss='0-1',
                 s=0.3,
                 deterministic=True,
                 random_state=None,
                 fit_intercept=True,
                 solver='adam',
                 alpha=0.01,
                 stepsize='decay',
                 mini_batch_size=None,
                 max_iters=None,
                 phi='linear',
                 **phi_kwargs):

        if max_iters is None:
            if phi == 'relu' or phi == 'threshold':
                # In this case nesterov's gradient descent is used
                self.max_iters = 2000
            elif solver == 'adam':
                self.max_iters = 5000
            else:
                self.max_iters = 100000
        else:
            self.max_iters = max_iters

        if mini_batch_size is None:
            if solver == 'adam':
                self.mini_batch_size = 32
            else:
                self.mini_batch_size = 1
        else:
            self.mini_batch_size = mini_batch_size

        self.solver = solver
        self.alpha = alpha
        self.stepsize = stepsize
        self.cvx_solvers = ['GUROBI', 'SCS', 'ECOS']
        super().__init__(loss=loss,
                         s=s,
                         deterministic=deterministic,
                         random_state=random_state,
                         fit_intercept=fit_intercept,
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
            if self.solver == 'adam' or self.solver == 'sgd':
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
        phi = self.compute_phi(X)

        # Constants
        n = phi.shape[0]
        m = phi.shape[2]

        # Supress the depreciation warnings
        warnings.simplefilter('ignore')

        # In case of 0-1 loss, learn constraints using the phi
        # These constraints are used in the optimization instead of phi

        if self.solver == 'cvx':
            # Use CVXpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)

            if self.loss == '0-1':
                # Constraints in case of 0-1 loss function

                # Summing up the phi configurations
                # for all possible subsets of classes for each instance
                F = np.vstack(list(np.sum(phi[:, S, ], axis=1)
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

                # Calculate the psi function and
                # add it to the objective function
                # First we calculate the all possible values of psi
                # for all the points
                psi = M @ mu + h
                sum_psi = (1 / n) * cvx.sum(cvx.max( \
                                        cvx.reshape(psi, (n, int(M.shape[0] / 
                                                             n))), axis=1))

            elif self.loss == 'log':
                # Constraints in case of log loss function
                sum_psi = 0
                for i in range(n):
                    sum_psi = sum_psi + \
                        (1 / n) * cvx.log_sum_exp(phi[i, :, :] @ mu)

            # Objective function
            objective = cvx.Minimize(self.lambda_ @ cvx.abs(mu) -
                                     self.tau_ @ mu + sum_psi)

            self.mu_, self.upper_ = \
                try_solvers(objective, None, mu, self.cvx_solvers)

        elif self.solver == 'sgd' or self.solver == 'adam':

            if self.loss == '0-1':
                # Function to calculate the psi subobjective
                # to be added to the objective function.
                # In addition the function returns subgradient
                # of the expected value psi
                # to be used by nesterov optimization.
                def f_(mu):
                    # First we calculate the all possible values of psi
                    # for all the points.

                    psi = 0
                    psi_grad = np.zeros(phi.shape[2], dtype=np.float64)

                    for i in range(n):
                        # Get psi for each data point
                        # and return the max value over all subset
                        # and its corresponding index.
                        g, psi_xi = self.psi(mu, phi[i, :, :])
                        psi_grad = psi_grad + g
                        psi = psi + psi_xi

                    psi = ((1 / n) * psi)
                    psi_grad = ((1 / n) * psi_grad)
                    return psi, psi_grad

                # When using SGD for the convex optimization
                # To compute the subgradient of the subobjective at one point
                def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                    i = batch_start_sample_id
                    psi = 0
                    psi_grad = np.zeros(phi.shape[2], dtype=np.float64)
                    while i < batch_end_sample_id:
                        sample_id = i % n
                        g, psi_xi = self.psi(mu, phi[sample_id, :, :])
                        psi_grad = psi_grad + g
                        psi = psi + psi_xi
                        i = i + 1

                    batch_size = batch_end_sample_id - batch_start_sample_id
                    psi_grad = ((1 / batch_size) * psi_grad)
                    psi = ((1 / batch_size) * psi)
                    return psi_grad

            elif self.loss == 'log':
                # Define the objective function and
                # the gradient for the log loss function.

                # The psi subobjective for all the datapoints
                def f_(mu):
                    phi_mu = phi @ mu
                    psi = (1 / n) *\
                            np.sum(scs.logsumexp((phi_mu), axis=1))

                    # Only computed in case of nesterov subgradient.
                    # In case of SGD, not required.
                    psi_grad = None
                    if isinstance(self.phi, RandomReLUPhi) or \
                       isinstance(self.phi, ThresholdPhi):
                        # Use the subgradient approach for the convex optimization
                        # The subgradient of the psi subobjective
                        # for all the datapoints
                        expPhi = np.exp(phi_mu)[:, np.newaxis, :]
                        psi_grad = (1 / n) *\
                                (np.sum(((expPhi @ phi)[:, 0, :] /
                                         np.sum(expPhi, axis=2)).transpose(),
                                        axis=1))

                    return psi, psi_grad

                # Use SGD for the convex optimization in general.
                # Gradient of the subobjective (psi) at an instance.
                def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                    i = batch_start_sample_id
                    expPhi = 0
                    batch_size = batch_end_sample_id - batch_start_sample_id
                    while i < batch_end_sample_id:
                        sample_id = i % n

                        expPhi_xi = np.exp(phi[sample_id, :, :] @ mu
                                        )[np.newaxis, np.newaxis, :]

                        sumExpPhi_xi = \
                                np.sum(((expPhi_xi @ phi[sample_id, :, :])
                                        [:, 0, :] /
                                        np.sum(expPhi_xi, axis=2)).transpose(),
                                       axis=1)

                        expPhi = expPhi + sumExpPhi_xi

                        i = i + 1

                    expPhi = ((1 / batch_size) * expPhi)
                    return expPhi

            if isinstance(self.phi, RandomReLUPhi) or \
               isinstance(self.phi, ThresholdPhi):
                self.params_ = nesterov_optimization_cmrc(self.tau_,
                                                          self.lambda_,
                                                          m,
                                                          f_,
                                                          None,
                                                          self.max_iters)
            elif self.solver == 'sgd':
                self.params_ = SGD_optimization(self.tau_,
                                                self.lambda_,
                                                n,
                                                m,
                                                f_,
                                                g_,
                                                self.max_iters,
                                                self.stepsize,
                                                self.mini_batch_size)
            elif self.solver == 'adam':
                self.params_ = adam(self.tau_,
                                    self.lambda_,
                                    n,
                                    m,
                                    f_,
                                    g_,
                                    self.max_iters,
                                    self.alpha,
                                    self.mini_batch_size)

            self.mu_ = self.params_['mu']
            self.upper_ = self.params_['best_value']

        else:
            raise ValueError('Unexpected solver ... ')

        self.is_fitted_ = True

        return self

    def psi(self, mu, phi):
        '''
        Function to compute the psi function in the objective
        using the given solution mu and the feature mapping 
        corresponding to a single instance.

        Parameters:
        -----------
        mu : `array`-like of shape (n_features)
            Solution.

        phi : `array`-like of shape (n_classes, n_features)
            Feature mapping corresponding to an instance and
            each class.

        Returns:
        --------
        g : `array`-like of shape (n_features)
            Gradient of psi for a given solution and feature mapping.

        psi_value : `int`
            The value of psi for a given solution and feature mapping.
        '''

        v = phi@mu
        indices = np.argsort(v)[::-1]
        value = v[indices[0]] - 1
        g = phi[indices[0],:]

        for k in range(1, self.n_classes):
            new_value = (k * value + v[indices[k]]) / (k+1)
            if new_value >= value:
                value = new_value
                g = (k * g + phi[indices[k],:]) / (k+1)
            else:
                break

        return g, (value + 1)

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper_bound : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        return self.upper_

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

        phi = self.compute_phi(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Summing up the phi configurations
            # for all possible subsets of classes for each instance
            F = np.vstack(list(np.sum(phi[:, S, ], axis=1)
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
            hy_x = np.vstack(list(np.sum(np.exp(v - np.tile(v[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes))).transpose()
            hy_x = np.reciprocal(hy_x)

            # Set the approach for prediction to deterministic
            # if not provided by user.
            if self.deterministic is None:
                self.deterministic = True

        return hy_x
