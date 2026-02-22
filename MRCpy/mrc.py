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
from itertools import combinations
import warnings

import cvxpy as cvx
import numpy as np
import time
import scipy.special as scs
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy import sparse
import scipy as sp

# Import the MRC super class
from MRCpy import BaseMRC
from MRCpy.solvers.cvx import *
from MRCpy.solvers.nesterov import *
from MRCpy.solvers.cg import *

class MRC(BaseMRC):
    r'''
    Minimax Risk Classifier

    This class implements Minimax Risk Classifiers (MRC) proposed in
    [1]_ using the default constraints. It supports two loss functions:
    0-1 loss and log-loss.

    The classifier solves the minimax risk problem:

    .. math::

        \mathrm{h}^{\mathcal{U}_1} \in \arg\min_{\mathrm{h}}
        \max_{\mathrm{p} \in \mathcal{U}_1} \ell(\mathrm{h}, \mathrm{p})

    which finds the classifier :math:`\mathrm{h}` that minimizes the
    worst-case expected loss over an uncertainty set
    :math:`\mathcal{U}_1` of distributions.

    The uncertainty set :math:`\mathcal{U}_1` is defined by bounding
    the expectations of the feature mappings :math:`\Phi(x, y)`:

    .. math::

        \mathcal{U}_1 = \left\{ \mathrm{p} :
        \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)]
        - \boldsymbol{\tau} \right| \leq \boldsymbol{\lambda} \right\}

    where :math:`\boldsymbol{\tau}` are the empirical mean estimates
    and :math:`\boldsymbol{\lambda}` controls the size of the uncertainty
    set based on the estimation accuracy.

    MRCs using :math:`\mathcal{U}_1` provide upper and lower bounds
    on the expected loss. See [1]_ and [3]_ for the detailed
    mathematical framework.

    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization. 0-1 loss
        quantifies the probability of classification error at a certain example
        for a certain rule. Log-loss quantifies the minus log-likelihood at a
        certain example for a certain rule.

    s : `float`, default = `0.3`
        Parameter that tunes the estimation of expected values
        of feature mapping function. It is used to calculate :math:`\boldsymbol{\lambda}`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \boldsymbol{\lambda} = s * \text{std}(\Phi(X,Y)) / \sqrt{\left| X \right|}

        where (X,Y) is the dataset of training samples and their
        labels respectively and
        :math:`\text{std}(\Phi(X,Y))` stands for standard deviation
        of :math:`\Phi(X,Y)` in the supervised dataset (X,Y).

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

    solver : {‘cvx’, ’subgrad’, ’cg’, 'ccg'}, default = ’subgrad’
        Method to use in solving the optimization problem. 
        Default is ‘subgrad’. To choose a solver,
        you might want to consider the following aspects:

        ’cvx’
            Solves the optimization problem using the cvxpy library.
            Obtains an accurate solution while requiring more time
            than the other methods. 
            Note that the library uses the GUROBI solver in cvxpy for which
            one might need to request for a license.
            A free license can be requested `here 
            <https://www.gurobi.com/academia/academic-program-and-licenses/>`_

        ’subgrad’
            Solves the optimization using a subgradient approach.
            The parameter `max_iters` determines the number of iterations
            for this approach. More iteration lead to an accurate solution
            while requiring more time.

        ’cg’
            Efficient learning algorithm for large number of features with
            few samples (high-dimensional data) based on column generation. 
            `eps2` is the hyperparameter that provide a 
            trade-off between time complexity and acccuracy.
            The maximum number of features 
            selected in each iteration is controlled by `m_max`. It
            also affects the computational complexity.

        .. seealso:: For more information about the constraint generation 
            algorithm for 0-1 MRC, see [4]_


        ’ccg’
            Efficient learning algorithm for large number of samples
            and features based on constraint generation. Efficiently
            handles the multi-class case (with quasi linear complexity).
            `eps1` and `eps2` are the parameters that provide a 
            trade-off between time complexity and acccuracy. 
            The maximum number of constraints selected in each iteration
            is controlled by the hyperparamters `n_max` and `m_max`. 
            These hyperparameters also affect the comptutional complexity.

        .. seealso:: For more information about the constraint generation 
            algorithm for 0-1 MRC, see [5]_

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

    tau_mat : `array`-like of shape (`n_classes`, `n_features`) or `float`
        Mean estimates
        for the expectations of feature mappings.

    lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or `float`
        Variance in the mean estimates
        for the expectations of the feature mappings.

    mu_ : `array`-like of shape (`n_classes`, `n_features`) or `float`
        Parameters learnt by the optimization.

    nu_ : `float`
        Parameter learnt by the optimization.

    mu_l_ : `array`-like of shape (`n_classes`, `n_features`) or `float`
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
    >>> clf.predict(X_test[:10, :])
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


    See Also
    --------
    MRCpy.CMRC : CMRC using uncertainty set :math:`\mathcal{U}_2` with additional marginal constraints [2]_.

    References
    ----------
    .. [1] Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax
           Classification with 0-1 Loss and Performance Guarantees.
           Advances in Neural Information Processing Systems, 33, 302-312.

    .. [2] Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized
           Maximum Entropy for Supervised Classification. IEEE Transactions
           on Information Theory, 68(4), 2530-2550.

    .. [3] Mazuelas, S., Romero, M., & Grunwald, P. (2023). Minimax Risk
           Classifiers with 0-1 Loss. Journal of Machine Learning Research,
           24(208), 1-48.

    .. [4] Bondugula, K., Mazuelas, S., & Pérez, A. (2023). Efficient
           Learning of Minimax Risk Classifiers in High Dimensions. The
           39th Conference on Uncertainty in Artificial Intelligence (UAI),
           206-215.

    .. [5] Bondugula, K., Mazuelas, S., & Pérez, A. (2025). Efficient
           Large-Scale Learning of Minimax Risk Classifiers. IEEE
           International Conference on Data Mining (ICDM).

    '''

    def __init__(self,
                 loss='0-1',
                 s=0.3,
                 deterministic=True,
                 random_state=None,
                 fit_intercept=True,
                 solver='subgrad',
                 max_iters=None,
                 n_max=None,
                 k_max=400,
                 eps1=None,
                 eps2=1e-5,
                 phi='linear',
                 **phi_kwargs):

        self.solver = solver
        self.k_max = k_max
        self.eps2 = eps2

        # Use the defaults
        if n_max is None:
            if self.solver == 'cg':
                self.n_max = 100
            elif self.solver == 'ccg':
                self.n_max = 400
        else:
            self.n_max = n_max

        # Use the defaults
        if eps1 is None:
            if self.solver == 'cg':
                self.eps1 = 1e-4
            elif self.solver == 'ccg':
                self.eps1 = 1e-2
        else:
            self.eps1 = eps1

        # Use the defaults
        if max_iters is None:
            if self.solver == 'subgrad':
                self.max_iters = 10000
            elif self.solver == 'cg':
                self.max_iters = 20
            elif self.solver == 'ccg':
                self.max_iters = 150
        else:
            self.max_iters = max_iters

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
            `n_dimensions` is the number of features.

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

        is_sparse = sparse.issparse(X)

        # 1. If the training instances matrix is sparse,
        #    then it is expected to be final features.
        #    We do not apply further feature transformation
        #    to avoid any disruption to the sparsity structure
        # 2. X_ is ignored since the current implementation only
        #    uses constraint generation approach that implicitly
        #    obtain X_ by iteratively the corresponding relevant
        #    constraints.
        if is_sparse:
            if self.solver != 'ccg':
                raise ValueError("Sparse matrices are only supported for ccg solver")
            
            if self.n_classes != 2:
                raise ValueError("Sparse matrices are not supported \
                                 for multi-class classification")
            
            if self.phi != 'linear':
                raise ValueError('Sparse matrices only \
                                 support linear features')
        

            # Call the minimax risk function that can solve the 
            # optimization for sparse matrices using ccg solver.
            # The implementation is only available for binary classification.
            n_classes = len(np.unique(Y))
            n = X.shape[0]

            # Transformed features
            if self.fit_intercept is True:
                X_transformed = sp.sparse.hstack([sp.sparse.csr_matrix([[1]]*n), X]).tocsr()
                dict_nnz = {}
                for idx, row in enumerate(X):
                    dict_nnz[idx] = row.nonzero()[1].tolist()

                for sample_idx, nnz_arr in dict_nnz.items():
                    nnz_arr_numpy = np.asarray(nnz_arr) + 1
                    nnz_arr_intercept = [0]
                    nnz_arr_intercept.extend(nnz_arr_numpy.tolist())
                    dict_nnz[sample_idx] = nnz_arr_intercept

            # Compute the mean vector estimate
            tau_0 = X_transformed[Y == 0, :].sum(axis=0)
            tau_1 = (-1) * X_transformed[Y == 1, :].sum(axis=0)
            tau_ = (tau_0 + tau_1) / n

            # Compute the standard deviation
            std_mat = X_transformed.copy()
            std_mat.data **=2
            lambda_ = np.sqrt(std_mat.sum(axis=0) / n - np.square(tau_))

            # Reshape for compatibility with upcoming computations
            tau_ = np.reshape(np.asarray(tau_), (tau_.shape[1],))
            lambda_ = self.s * np.reshape(np.asarray(lambda_), (lambda_.shape[1],))

            self.minimax_risk(X, tau_, lambda_, n_classes, dict_nnz)

        else:
            super().fit(X=X, Y=Y, X_=X_)

        return self

    def minimax_risk(self, X, tau_mat, lambda_mat, n_classes, dict_nnz = None):
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

        tau_mat : `array`-like of shape (`n_classes`, `n_features`)
            Mean estimates
            for the expectations of feature mappings.

        lambda_mat : `array`-like of shape (`n_classes`, `n_features`)
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
        self.tau_mat = check_array(tau_mat, accept_sparse=True, ensure_2d=False)
        self.lambda_mat = check_array(lambda_mat, accept_sparse=True,
                                   ensure_2d=False)
        
        # Check if the training data is sparse matrix
        is_sparse = sparse.issparse(X)
        
        if is_sparse:
            # If the input matrix is sparse, 
            # it is expected to be the feature matrix.
            # We do not apply any feature transformation.
            X_transform = X
        else:
            X_transform = self.compute_features(X)
            X_transform = np.unique(X_transform, axis=0)

        # Constants
        d = X_transform.shape[1]
        n = X_transform.shape[0]

        # Save the phi configurations for finding the lower bounds
        self.X_transform = X_transform

        if self.solver == 'cvx':
            # Use cvxpy for the convex optimization of the MRC.

            # Variables
            if self.n_classes == 2:
                mu = cvx.Variable((1, d))
            else:
                mu = cvx.Variable((self.n_classes, d))

            if self.loss == '0-1':
                # Compute the varphi function for 0-1 loss
                def neg_nu(mu):
                    if self.n_classes == 2:
                        phi_mu = X_transform @ mu.T
                        return cvx.max(
                            cvx.hstack([
                                cvx.max(phi_mu),
                                cvx.max(-phi_mu),
                                0.5
                            ])
                        )

                    # Efficient approach for multi-class case
                    # Precompute once
                    phi_mu = X_transform @ mu.T   # shape: (n_samples, n_classes)

                    exprs = []
                    for r in range(1, self.n_classes + 1):
                        for S in combinations(range(self.n_classes), r):
                            scores = cvx.sum(phi_mu[:, S], axis=1)
                            exprs.append(cvx.max(scores + 1 - 1 / r))

                    return cvx.max(cvx.hstack(exprs))


            elif self.loss == 'log':
                numConstr = X_transform.shape[0]

                def neg_nu(mu):
                    if self.n_classes == 2:
                        return cvx.max(cvx.hstack(
                                    cvx.log_sum_exp([X_transform[i,:] @ mu.T, 
                                                -X_transform[i,:] @ mu.T])
                                              for i in range(numConstr)))

                    return cvx.max(cvx.hstack(cvx.log_sum_exp(X_transform[i,:] @
                                                              mu.T)
                                              for i in range(numConstr)))

            else:
                raise ValueError('The given loss function is not available ' +
                                 'for this classifier')

            # Objective function
            objective = cvx.Minimize(cvx.sum(cvx.multiply(self.lambda_mat, cvx.abs(mu))) -
                                     cvx.sum(cvx.multiply(self.tau_mat, mu)) +
                                     neg_nu(mu))

            self.mu_, self.upper_ = try_solvers(objective,
                                                None,
                                                mu,
                                                self.cvx_solvers)
            # Flatten mu_ for binary case to match expected shape (d,)
            if self.n_classes == 2:
                self.mu_ = self.mu_.flatten()
            self.nu_ = np.atleast_1d((-1) * (neg_nu(self.mu_).value))

        elif self.solver == 'subgrad':
            # Use the subgradient approach for the convex optimization of MRC

            if self.loss == '0-1':

                if self.n_classes == 2:
                    #TODO: An efficient implementation with less matrix computation
                    # Summing up the phi configurations
                    # for all possible subsets of classes for each instance

                    # Create the feature mapping matrix
                    phi = self.phi.eval_x(X)
                    phi = np.unique(phi, axis=0)

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

                    tau_flattened = self.tau_mat.flatten()
                    lambda_flattened = self.lambda_mat.flatten()
                    # Learn the classifier
                    self.upper_params_ = \
                            nesterov_optimization_minimized_mrc(M,
                                                                h,
                                                                tau_flattened,
                                                                lambda_flattened,
                                                                self.max_iters)
                else:
                    # Define the subobjective function and
                    # its gradient for the 0-1 loss function.                    
                    def f_(mu):
                        """
                        mu: (n_classes, n_features)
                        X_transform: (n_samples, n_features)
                        """
                        phi_mu = X_transform @ mu.T   # (n_samples, n_classes)

                        # Iterate through all subsets and 
                        # find the sample and subset pair achieving maximum value
                        exprs = []
                        subset_sample_sets = []
                        for r in range(1, self.n_classes + 1):
                            for S in combinations(range(self.n_classes), r):
                                # sum over selected classes
                                scores = phi_mu[:, S].sum(axis=1)   # (n_samples,)
                                idx = np.argmax(scores)
                                subset_sample_sets.append((idx, S))
                                exprs.append(scores[idx] + 1 - 1 / r)

                        set_idx = np.argmax(exprs)

                        # Obtain the sample and subset achieving the maximum value
                        sample_idx = subset_sample_sets[set_idx][0]
                        S = subset_sample_sets[set_idx][1]

                        return exprs[set_idx], sample_idx, S
                    
                    def g_(mu, idx, subset):
                        """
                        mu: (n_classes, n_features)
                        idx: sample index
                        subset: tuple of class indices
                        """

                        # Not accounting for binary case 
                        # since this implementation is only for multiclass
                        # Construct the corresponding gradient and return
                        grad_ = np.zeros((self.n_classes, d))
                        grad_[subset, :] = X_transform[idx, :].reshape(1, -1)

                        return grad_

                    # Calculate the upper bound
                    self.upper_params_ = nesterov_optimization_mrc(self.tau_mat,
                                                                   self.lambda_mat,
                                                                   f_,
                                                                   g_,
                                                                   self.max_iters)

            elif self.loss == 'log':

                # Define the subobjective function and
                # its gradient for the log loss function.                
                def f_(mu):
                    phi_mu = X_transform @ mu.T
                    if self.n_classes == 2:
                        phi_mu = np.hstack([phi_mu, -phi_mu])

                    log_sum_exp = scs.logsumexp((phi_mu), axis=1)
                    sample_idx = np.argmax(log_sum_exp)
                    if self.n_classes == 2:
                        subset = np.arange(self.n_classes-1)
                    else:
                        subset = np.arange(self.n_classes)

                    return log_sum_exp[sample_idx], sample_idx, subset

                def g_(mu, idx, subset):
                    X_idx = X_transform[idx, :]
                    phi_mu = X_idx @ mu.T
                    if self.n_classes == 2:
                        phi_mu = np.hstack([phi_mu, -phi_mu])
                    
                    # Use log-sum-exp trick for numerical stability
                    max_phi = np.max(phi_mu)
                    expPhi_xi = np.exp(phi_mu - max_phi)

                    # Compute the softmax weights
                    weights = expPhi_xi / np.sum(expPhi_xi)

                    # Construct the corresponding gradient and return
                    if self.n_classes == 2:
                        grad_ = np.zeros((1, d))
                        grad_ = (X_idx * weights[0] - X_idx * weights[1]).reshape(1, -1)
                    else:
                        grad_ = np.zeros((self.n_classes, d))
                        grad_[subset, :] = weights[:, None] * X_idx[None, :]

                    return grad_

                # Calculate the upper bound
                self.upper_params_ = nesterov_optimization_mrc(self.tau_mat,
                                                               self.lambda_mat,
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
            # to solve the optimization with many features but few samples.

            if self.loss == 'log':
                raise ValueError('The \'cg\' solver is only available ' +
                                 'for 0-1 loss.')

    #-----> Initialization for constraint generation method.

        #-> Reduce the feature space by restricting the number of features
        #   based on the variance in the features, that is, picking first
        #   10*N minimum variance features.

            # Create the feature mapping matrix
            phi = self.phi.eval_x(X)
            phi = np.unique(phi, axis=0)

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

            tau_flattened = self.tau_mat.flatten()
            lambda_flattened = self.lambda_mat.flatten()

            M = F / (cardS[:, np.newaxis])
            h = 1 - (1 / cardS)
            N = M.shape[0]
            argsort_columns = np.argsort(np.abs(lambda_flattened))
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
                                                     tau_flattened[index_CG],
                                                     lambda_flattened[index_CG],
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
                                                             tau_flattened,
                                                             lambda_flattened,
                                                             I,
                                                             self.n_max,
                                                             self.k_max,
                                                             warm_start,
                                                             nu_,
                                                             self.eps1)

    #-----> Combined constraint-column generation with efficient constraint selections
    #       for learning with large-scale data having 
    #       large number of sample, features, and classes
        elif self.solver == 'ccg':
            # Use algorithms based on constraint generation 
            # to solve large-scale optimization with many samples, features, and classes.

            if self.loss == "log":
                raise ValueError('The \'ccg\' solver is only available ' +
                                 'for 0-1 loss.')

            # This classification is based on empirical observation.
            if d > 1000 or is_sparse:
                large_features = True
            else:
                large_features = False

            if large_features is not True:
            # Perform constraint generation approach for large number
            # of samples.
                if self.n_classes == 2:
                    from MRCpy.solvers.main_ccg_large_n.main import (
                        main_large_n
                    )

                    phi = self.phi.eval_x(X)
                    phi = np.unique(phi, axis=0)

                    (self.mu_, self.nu_, self.upper_, self.R_k) = main_large_n(
                        phi,
                        self.phi,
                        self.tau_mat,
                        self.lambda_mat,
                        self.n_max,
                        self.max_iters,
                        self.eps1
                    )

                else:
                    from MRCpy.solvers.main_ccg_large_n_multiclass.main import (
                        main_large_n_efficient_multiclass
                    )
                    (self.mu_, self.nu_, self.upper_, self.R_k, 
                    constr_dict) = main_large_n_efficient_multiclass(
                        X_transform,
                        self.tau_mat,
                        self.lambda_mat,
                        self.n_max,
                        self.max_iters,
                        self.eps1
                    )
            else:
            # Perform a combination of constraint and column generation approach
            # for large number of samples and features.
                if is_sparse and self.n_classes == 2:
                    from MRCpy.solvers.main_ccg_large_n_m_sparse_binary.main import (
                        main_large_n_m_sparse_binary
                    )

                    # Training data is considered to be sparse csr matrix
                    # The dictionary dict_nnz provides the information
                    # on the nonzero columns in each instance x of the dataset.
                    (self.mu_, self.nu_, 
                    self.upper_, self.R_k) = main_large_n_m_sparse_binary(
                        X_transform,
                        self.tau_mat,
                        self.lambda_mat,
                        self.n_max,
                        self.k_max,
                        self.eps1,
                        self.eps2,
                        dict_nnz,
                        self.max_iters
                    )
                else:
                    #TODO: Implement this solver more efficiently
                    #      Current implementation in the folder 
                    #      solvers/main_ccg_large_n_m is inefficient.
                    from MRCpy.solvers.main_ccg_large_n_m.main import (
                        main_large_n_m
                    )
                    (self.mu_, self.nu_, self.upper_, R_k,
                     solver_times_gurobi, solver_times, idx_cols,
                     mrc_cg_time, init_mrc_cg_time,
                     n_tries) = main_large_n_m(
                        X,
                        self.tau_mat,
                        self.lambda_mat,
                        self.phi,
                        self.n_max,
                        self.k_max,
                        self.eps1,
                        self.eps2,
                        self.max_iters
                    )

        else:
            raise ValueError('Unexpected solver ... ')

        self.is_fitted_ = True

        if self.n_classes == 2:
            self.mu_ = self.mu_.flatten().reshape(1, d)
        elif self.n_classes > 2 and (self.mu_.ndim != 2 or self.mu_.shape[0] == 1 or self.mu_.shape[1] == 1):
            self.mu_ = self.mu_.reshape(self.n_classes, d)

        return self

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper_bound : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        # Specific bound for deterministic classifier can be obtained
        # by solving an additional optimization problem.
        # Note that the upper bound obtained as a result of
        # minimax risk optimization corresponds to the non-deterministic classifier.
        if self.deterministic:
            # Number of instances in training 
            n = self.X_transform.shape[0]

            # Feature length
            d = self.X_transform.shape[1]

            phi_mu = self.X_transform @ self.mu_.T
            if self.n_classes == 2:
                phi_mu = np.hstack([phi_mu, -phi_mu])

            hy_x_det = np.zeros((n, self.n_classes))
            for i in range(phi_mu.shape[0]):
                hy_x_det[i, np.argmax(phi_mu[i,:])] = 1

            # Variables
            if self.n_classes == 2:
                mu = cvx.Variable((1, d))
            else:
                mu = cvx.Variable((self.n_classes, d))
            
            phi_mu_cvx = self.X_transform @ mu.T
            if self.n_classes == 2:
                phi_mu_cvx = cvx.hstack([phi_mu_cvx, -phi_mu_cvx])

            objective = cvx.Minimize(1 - cvx.sum(cvx.multiply(self.tau_mat, mu)) + \
                            cvx.sum(cvx.multiply(self.lambda_mat, cvx.abs(mu))) + \
                            cvx.max(phi_mu_cvx - hy_x_det))

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

        if sparse.issparse(self.X_transform):
            print('Lower bound computation does not support sparse matrices. \
                   It will continue the computation converting \
                   the sparse matrix of training samples to dense matrix')

        X_transform = self.X_transform
        # Learned feature mappings
        phi_mu = np.dot(X_transform, self.mu_.T)
        if self.n_classes == 2:
            phi_mu = np.hstack([phi_mu, -phi_mu])

        hy_x = np.clip(1 + phi_mu + self.nu_, 0., None)

        # Variables
        n = X_transform.shape[0]
        d = X_transform.shape[1]

        if self.loss == '0-1':

            # To define the objective function and
            # the gradient for the 0-1 loss function.
            # epsilon
            eps = np.clip(1 + phi_mu + self.nu_, 0, None)
            c = np.sum(eps, axis=1)
            zeros = np.isclose(c, 0)
            c[zeros] = 1
            eps[zeros, :] = 1 / self.n_classes
            eps = eps / (c[:, np.newaxis])
            # Using negative of epsilon
            # for the nesterov accelerated optimization
            eps = eps - 1

        elif self.loss == 'log':

            # To define the objective function and
            # the gradient for the log loss function.
            # Using negative of epsilon
            # for the nesterov accelerated optimization
            eps = phi_mu - \
                scs.logsumexp(phi_mu, axis=1)[:, np.newaxis]

        else:
            raise ValueError('The given loss function is not available ' +
                             'for this classifier')

        if not self.deterministic:

            if self.solver == 'cvx':
                # Use cvxpy for the convex optimization of the MRC

                if self.n_classes == 2:
                    low_mu = cvx.Variable((1, d))
                else:
                    low_mu = cvx.Variable((self.n_classes, d))

                phi_mu_cvx = X_transform @ low_mu.T
                if self.n_classes == 2:
                    phi_mu_cvx = cvx.hstack([phi_mu_cvx, -phi_mu_cvx])

                # Objective function
                objective = cvx.Minimize(cvx.sum(cvx.multiply(self.lambda_mat, cvx.abs(low_mu))) -
                                         cvx.sum(cvx.multiply(self.tau_mat, low_mu)) +
                                         cvx.max(phi_mu_cvx + eps))

                self.mu_l_, self.lower_ = \
                    try_solvers(objective, None, low_mu, self.cvx_solvers)

                # Maximize the function
                self.lower_ = (-1) * self.lower_

            elif self.solver == 'subgrad' or self.solver == 'cg' or self.solver == 'ccg':
                # Use the subgradient approach for the convex optimization of MRC

                # Defining the partial objective and its gradient.
                def f_(mu):
                    """
                    mu: (n_classes, n_features)
                    X_transform: (n_samples, n_features)
                    """

                    phi_mu_subgrad = X_transform @ mu.T
                    if self.n_classes == 2:
                        phi_mu_subgrad = np.hstack([phi_mu_subgrad, -phi_mu_subgrad])

                    varphi_lower = phi_mu_subgrad + eps
                    idx = np.argmax(varphi_lower)
                    sample_idx, label_idx = np.unravel_index(idx, varphi_lower.shape)
                    return varphi_lower[sample_idx, label_idx], sample_idx,  [label_idx]

                def g_(mu, idx, subset):
                    """
                    mu: (n_classes, n_features)
                    idx: sample index
                    subset: tuple of class indices
                    """

                    # Construct the corresponding gradient and return
                    if self.n_classes == 2:
                        grad_ = np.zeros((1, d))
                        if subset[0] == 1:
                            grad_ = -X_transform[idx, :].reshape(1, -1)
                        else:
                            grad_ = X_transform[idx, :].reshape(1, -1)
                    else:
                        grad_ = np.zeros((self.n_classes, d))
                        grad_[subset, :] = X_transform[idx, :].reshape(1, -1)

                    return grad_

                # Lower bound
                self.lower_params_ = nesterov_optimization_mrc(self.tau_mat,
                                                               self.lambda_mat,
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

            if self.n_classes == 2:
                mu_l_ = cvx.Variable((1, d))
            else:
                mu_l_ = cvx.Variable((self.n_classes, d))

            phi_mu_cvx = X_transform @ mu_l_.T
            if self.n_classes == 2:
                phi_mu_cvx = cvx.hstack([phi_mu_cvx, -phi_mu_cvx])
    
            objective = cvx.Maximize(1 - cvx.sum(cvx.multiply(self.tau_mat, mu_l_)) - \
                            cvx.sum(cvx.multiply(self.lambda_mat, cvx.abs(mu_l_))) + \
                            cvx.min(phi_mu_cvx - hy_x_det))

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

        X_tranform = self.compute_features(X)
        phi_mu = X_tranform @ self.mu_.T

        if self.n_classes == 2:
            phi_mu = np.hstack([phi_mu, -phi_mu])

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Unnormalized conditional probabilityes
            hy_x = np.clip(1 + phi_mu + self.nu_, 0., None)

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
            # Normalizing conditional probabilities
            hy_x = np.vstack(list(np.sum(np.exp(phi_mu - np.tile(phi_mu[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes))).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x
