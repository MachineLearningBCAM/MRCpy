"""
Minimax Risk Classification. Copyright (C) 2021 Kartheek Bondugula

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
import time
import scipy.special as scs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the MRC super class
from MRCpy import BaseMRC
from MRCpy.solvers.cvx import *
from MRCpy.solvers.nesterov import *
from MRCpy.solvers.cg import *

class MRC(BaseMRC):
    '''
    Minimax Risk Classifier

    The class MRC implements the method Minimimax Risk Classifiers (MRC)
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

    solver : {‘cvx’, ’subgrad’, ’cg’}, default = ’subgrad’
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

        ’subgrad’
            Solves the optimization using a subgradient approach.
            The parameter `max_iters` determines the number of iterations
            for this approach. More iteration lead to an accurate solution
            while requiring more time.

        ’cg’
            Solves the optimization using an algorithm
            based on constraint generation. This algorithm provides 
            efficient learning especially for scenarios
            with large number of features.

        .. seealso:: For more information about the constraint generation 
            algorithm for 0-1 MRC, one can refer to the following resource:
        
                    [1] `Bondugula, K., Mazuelas, S., & Pérez, A. (2023).
                    Efficient Learning of Minimax Risk Classifiers
                    in High Dimensions.
                    The 39th Conference on
                    Uncertainty in Artificial Intelligence, 206-215.
                    <https://proceedings.mlr.press/v216/bondugula23a.html>`_

    max_iters : `int`, default = `10000`
        Maximum number of iterations to use
        for finding the solution of optimization when
        using the subgradient approach.

    n_max : `int`, default = `100`
        Maximum number of features selected in each iteration
        in case of ’cg’ solver.

    k_max : `int`, default = `20`
        Maximum number of iterations in case of ’cg’ solver.

    eps : `float`, default = `1e-4`
        Dual constraints' violation threshold for ’cg’ solver. 

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

    def __init__(self,
                 loss='0-1',
                 s=0.3,
                 deterministic=True,
                 random_state=None,
                 fit_intercept=True,
                 solver='subgrad',
                 max_iters=10000,
                 n_max=100,
                 k_max=20,
                 eps=1e-4,
                 phi='linear',
                 **phi_kwargs):

        self.solver = solver
        self.n_max = n_max
        self.k_max = k_max
        self.eps = eps
        self.max_iters = max_iters
        self.cvx_solvers = ['GUROBI', 'SCS', 'ECOS']
        super().__init__(loss=loss,
                         s=s,
                         deterministic=deterministic,
                         random_state=random_state,
                         fit_intercept=fit_intercept,
                         phi=phi, **phi_kwargs)

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
        phi = self.compute_phi(X)

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

        if self.solver == 'cvx':
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

            self.mu_, self.upper_ = try_solvers(objective,
                                                None,
                                                mu,
                                                self.cvx_solvers)
            self.nu_ = (-1) * (neg_nu(self.mu_).value)

        elif self.solver == 'subgrad':
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
                self.upper_params_ = \
                        nesterov_optimization_minimized_mrc(M,
                                                            h,
                                                            self.tau_,
                                                            self.lambda_,
                                                            self.max_iters)

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
                self.upper_params_ = nesterov_optimization_mrc(self.tau_,
                                                               self.lambda_,
                                                               m,
                                                               f_,
                                                               g_,
                                                               self.max_iters)

            else:
                raise ValueError('The given loss function is not available ' +
                                 'for this classifier')

            self.mu_ = self.upper_params_['mu']
            self.nu_ = self.upper_params_['nu']
            self.upper_ = self.upper_params_['best_value']

        elif self.solver == 'cg':
            # Use methods based on constraint generation
            # to solve the optimization (corresponding to 0-1 loss only).

            if self.loss == 'log':
                raise ValueError('The \'cg\' solver is only available ' +
                                 'for 0-1 loss.')

    #-----> Initialization for constraint generation method.

        #-> Reduce the feature space by restricting the number of features
        #   based on the variance in the features, that is, picking first
        #   10*N minimum variance features.
            N = M.shape[0]
            argsort_columns = np.argsort(np.abs(self.lambda_))
            index_CG        = argsort_columns[:10*N]

        #-> Solve the optimization using the reduced training set
        #   and first order subgradient methods to get an
        #   initial low accuracy solution in minimum time.
            M_reduced = M[:, index_CG]
            M_reduced_t = M_reduced.transpose()

            # Calculate the upper bound
            upper_params_ = \
                 nesterov_optimization_minimized_mrc(M_reduced,
                                                     h,
                                                     self.tau_[index_CG],
                                                     self.lambda_[index_CG],
                                                     100)
            mu_ = upper_params_['mu']
            nu_ = upper_params_['nu']

        #-> Transform the solution obtained in the reduced space
        #   to the original space
            initial_features_limit = 100
            if np.sum(mu_!=0) > initial_features_limit:
                I = (np.argsort(np.abs(mu_))[::-1])[:initial_features_limit]
            else:
                I = np.where(mu_!=0)[0]

            warm_start = mu_[I] 
            I = np.array(index_CG)[I].tolist()

    #-----> Now apply the method of constraint generation using the 
    #       low accuracy solution.

            self.mu_, self.nu_, self.upper_, self.I = mrc_cg(M,
                                                             h,
                                                             self.tau_,
                                                             self.lambda_,
                                                             I,
                                                             self.n_max,
                                                             self.k_max,
                                                             warm_start,
                                                             nu_,
                                                             self.eps)
        else:
            raise ValueError('Unexpected solver ... ')

        self.is_fitted_ = True
        return self

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper_bound : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        if self.deterministic:
            # Number of instances in training 
            n = self.lowerPhiConfigs.shape[0]

            # Feature mapping length
            m = self.phi.len_

            phi_mu = self.lowerPhiConfigs @ self.mu_

            hy_x_det = np.zeros((n, self.n_classes))
            for i in range(phi_mu.shape[0]):
                hy_x_det[i, np.argmax(phi_mu[i,:])] = 1

            hy_x_det = np.reshape(hy_x_det, (n * self.n_classes,))

            phi = np.reshape(self.lowerPhiConfigs, (n * self.n_classes, m))

            mu = cvx.Variable(self.phi.len_)
            objective = cvx.Minimize(1 - self.tau_ @ mu + \
                            self.lambda_ @ cvx.abs(mu) + \
                            cvx.max(phi @ mu - hy_x_det))

            _, self.upper_ = try_solvers(objective,
                                         None,
                                         mu,
                                         self.cvx_solvers)
        return self.upper_

    def get_lower_bound(self):
        '''
        Obtains the lower bound on the expected loss for the fitted classifier.

        Returns
        -------
        lower_bound : `float`
            Lower bound of the error for the fitted classifier.
        '''

        # Classifier should be fitted to obtain the lower bound
        check_is_fitted(self, "is_fitted_")

        # Learned feature mappings
        phi = self.lowerPhiConfigs
        phi_mu = np.dot(phi, self.mu_)
        hy_x = np.clip(1 + phi_mu + self.nu_, 0., None)

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

        if not self.deterministic:

            if self.solver == 'cvx':
                # Use CVXpy for the convex optimization of the MRC

                low_mu = cvx.Variable(m)

                # Objective function
                objective = cvx.Minimize(self.lambda_ @ cvx.abs(low_mu) -
                                         self.tau_ @ low_mu +
                                         cvx.max(phi @ low_mu + eps))

                self.mu_l_, self.lower_ = \
                    try_solvers(objective, None, low_mu, self.cvx_solvers)

                # Maximize the function
                self.lower_ = (-1) * self.lower_

            elif self.solver == 'subgrad' or self.solver == 'cg':
                # Use the subgradient approach for the convex optimization of MRC

                # Defining the partial objective and its gradient.
                def f_(mu):
                    return phi @ mu + eps

                def g_(mu, idx):
                    return phi.transpose()[:, idx]

                # Lower bound
                self.lower_params_ = nesterov_optimization_mrc(self.tau_,
                                                               self.lambda_,
                                                               m,
                                                               f_,
                                                               g_,
                                                               self.max_iters)

                self.mu_l_ = self.lower_params_['mu']
                self.lower_ = self.lower_params_['best_value']

                # Maximize the function
                # as the nesterov optimization gives the minimum
                self.lower_ = -1 * self.lower_

            else:
                raise ValueError('Unexpected solver ... ')

        elif self.deterministic:

            hy_x_det = np.zeros((n, self.n_classes))
            for i in range(n):
                hy_x_det[i, np.argmax(phi_mu[i,:])] = 1

            hy_x_det = np.reshape(hy_x_det, (n * self.n_classes,))

            mu_l_ = cvx.Variable(m)
            objective = cvx.Maximize(1 - self.tau_ @ mu_l_ - \
                            self.lambda_ @ cvx.abs(mu_l_) + \
                            cvx.min(phi @ mu_l_ - hy_x_det))

            self.mu_l_, self.lower_ = try_solvers(objective,
                                                     None,
                                                     mu_l_,
                                                     self.cvx_solvers)

        return self.lower_

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

        phi = self.compute_phi(X)

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

        elif self.loss == 'log':
            # Constraints in case of log loss function

            v = np.dot(phi, self.mu_)

            # Normalizing conditional probabilities
            hy_x = np.vstack(list(np.sum(np.exp(v - np.tile(v[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes))).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x
