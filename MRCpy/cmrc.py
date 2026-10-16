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

import warnings

import cvxpy as cvx
import time
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
    r'''
    Constrained Minimax Risk Classifier

    This class implements Minimax Risk Classifiers with fixed marginal
    distributions (CMRC) proposed in [1]_ using additional marginals
    constraints on the instances. It supports two loss functions:
    0-1 loss and log-loss.

    The classifier solves the minimax risk problem:

    .. math::

        \mathrm{h}^{\mathcal{U}_2} \in \arg\min_{\mathrm{h}}
        \max_{\mathrm{p} \in \mathcal{U}_2} \ell(\mathrm{h}, \mathrm{p})

    which finds the classifier :math:`\mathrm{h}` that minimizes the
    worst-case expected loss over an uncertainty set
    :math:`\mathcal{U}_2` of distributions.

    The uncertainty set :math:`\mathcal{U}_2` is defined by fixing the
    marginal distribution of features to the empirical marginal and
    bounding the expectations of the feature mappings
    :math:`\Phi(x, y)`:

    .. math::

        \mathcal{U}_2 = \left\{ \mathrm{p} : \mathrm{p}_x =
        \hat{\mathrm{p}}_x, \;
        \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)]
        - \boldsymbol{\tau} \right| \leq \boldsymbol{\lambda} \right\}

    where :math:`\hat{\mathrm{p}}_x` is the empirical marginal
    distribution of features, :math:`\boldsymbol{\tau}` are the empirical mean
    estimates, and :math:`\boldsymbol{\lambda}` controls the size of the
    uncertainty set based on the estimation accuracy.

    See [1]_ for further details.

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

        where (X,Y) is the dataset of training samples and their labels
        respectively and :math:`\text{std}(\Phi(X,Y))` stands for
        standard deviation of :math:`\Phi(X,Y)` in the supervised
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
            Note that the library uses the GUROBI solver in cvxpy for which
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

    tau_ : `array`-like of shape (`n_classes`, `n_features`)
        Mean estimates for the expectations of feature mappings.

    lambda_ : `array`-like of shape (`n_classes`, `n_features`)
        Variance in the mean estimates
        for the expectations of the feature mappings.

    mu_ : `array`-like of shape (`n_classes`, `n_features`)
        Parameters learnt by the optimization.

    params_ : `dict`
        Dictionary that stores the optimal points and best value of
        the function.

    See Also
    --------
    MRCpy.MRC : MRC using uncertainty set :math:`\mathcal{U}_1` without marginal constraints.

    References
    ----------
    .. [1] Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized
           Maximum Entropy for Supervised Classification. IEEE Transactions
           on Information Theory, 68(4), 2530-2550.

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
                 return_fn_vals = False,
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
                self.mini_batch_size = 32
        else:
            self.mini_batch_size = mini_batch_size

        self.solver = solver
        self.alpha = alpha
        self.stepsize = stepsize
        self.return_fn_vals = return_fn_vals
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
            Instances used in the minimax risk optimization instead of the
            training instances X, which are still used to compute the
            expectation estimates. For instance, a smaller set of instances
            to reduce the training time, or instances from a different
            distribution. If None, all the training instances X are used.

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

    def minimax_risk(self, X, tau_mat, lambda_mat, n_classes):
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
        self.tau_ = check_array(tau_mat.flatten(), accept_sparse=True, ensure_2d=False)
        self.lambda_ = check_array(lambda_mat.flatten(), accept_sparse=True,
                                   ensure_2d=False)

        # Features of the instances (n_samples, n_features), used in the
        # compact form of the feature mapping Phi(x, y) (see the method
        # scores), without the tensor of Phi(x, y) for all the labels.
        # Binary classification without one-hot encoding uses
        # Phi(x, y) = +-features(x) (parameters of length n_features).
        X_transform = self.compute_features(X)
        self.binary_compact_ = (self.n_classes == 2 and
                                not getattr(self.phi, 'one_hot', False))

        # Constants
        n = X_transform.shape[0]
        m = self.tau_.shape[0]

        if self.solver == 'cvx':
            # Use cvxpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)
            phi_mu = self.scores(X_transform, mu)
            constraints = None

            if self.loss == '0-1':
                # varphi(x_i) = 1 + max over the subsets S of labels of
                # (sum_{y in S} phi_mu[i, y] - 1) / |S| is at most nu_i if
                # and only if sum_y pos(phi_mu[i, y] - nu_i + 1) <= 1, as in
                # MRC. This encodes the maximum over the 2^n_classes - 1
                # subsets of labels with a variable nu_i for each instance.
                nu = cvx.Variable(n)
                constraints = [
                    cvx.sum(cvx.pos(phi_mu - cvx.reshape(nu, (n, 1), order='F') @
                                    np.ones((1, self.n_classes)) + 1), axis=1) <= 1
                ]
                sum_varphi = (1 / n) * cvx.sum(nu)

            elif self.loss == 'log':
                # Constraints in case of log loss function
                sum_varphi = (1 / n) * cvx.sum(cvx.log_sum_exp(phi_mu, axis=1))

            # Objective function
            objective = cvx.Minimize(self.lambda_ @ cvx.abs(mu) -
                                     self.tau_ @ mu + sum_varphi)

            self.mu_, self.upper_ = \
                try_solvers(objective, constraints, mu, self.cvx_solvers)

        elif self.solver == 'sgd' or self.solver == 'adam':

            if self.loss == '0-1':
                # Function to calculate the varphi subobjective
                # to be added to the objective function.
                # In addition the function returns subgradient
                # of the expected value varphi
                # to be used by nesterov optimization.
                def f_(mu):
                    varphi, weights = self.varphi(self.scores(X_transform, mu))
                    varphi_grad = self.weighted_features(X_transform, weights)
                    return (1 / n) * np.sum(varphi), (1 / n) * varphi_grad

                # When using SGD for the convex optimization
                # To compute the subgradient of the subobjective
                # at a batch of instances
                def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                    batch = X_transform[np.arange(batch_start_sample_id,
                                                  batch_end_sample_id) % n]
                    _, weights = self.varphi(self.scores(batch, mu))
                    batch_size = batch_end_sample_id - batch_start_sample_id
                    return (1 / batch_size) * \
                        self.weighted_features(batch, weights)

            elif self.loss == 'log':
                # Define the objective function and
                # the gradient for the log loss function.

                # The varphi subobjective for all the datapoints
                def f_(mu):
                    phi_mu = self.scores(X_transform, mu)
                    varphi = (1 / n) * np.sum(scs.logsumexp(phi_mu, axis=1))

                    # Only computed in case of nesterov subgradient.
                    # In case of SGD, not required.
                    varphi_grad = None
                    if isinstance(self.phi, RandomReLUPhi) or \
                       isinstance(self.phi, ThresholdPhi):
                        # The subgradient of the varphi subobjective
                        # for all the datapoints: expectation of the
                        # features with the softmax of the scores
                        varphi_grad = (1 / n) * self.weighted_features(
                            X_transform, scs.softmax(phi_mu, axis=1))

                    return varphi, varphi_grad

                # Use SGD for the convex optimization in general.
                # Gradient of the subobjective (varphi) at a batch of instances.
                def g_(mu, batch_start_sample_id, batch_end_sample_id, n):
                    batch = X_transform[np.arange(batch_start_sample_id,
                                                  batch_end_sample_id) % n]
                    weights = scs.softmax(self.scores(batch, mu), axis=1)
                    batch_size = batch_end_sample_id - batch_start_sample_id
                    return (1 / batch_size) * \
                        self.weighted_features(batch, weights)

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
                                                self.mini_batch_size,
                                                self.return_fn_vals)
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

    def scores(self, X_transform, mu):
        '''
        Products :math:`\Phi(x, y)^T \mu` of the feature mapping and the
        parameters for each instance and label, using the features of the
        instances, without the feature mapping of every label: with
        :math:`\psi(x)` the features of instance :math:`x` (from
        `compute_features`), in the multiclass case
        :math:`\Phi(x, y)^T \mu = \psi(x)^T \mu_y` with :math:`\mu_y` the
        parameters of label :math:`y`, and in the binary case
        :math:`\Phi(x, y)^T \mu = \pm \psi(x)^T \mu`.

        Parameters
        ----------
        X_transform : array-like of shape (n_samples, n_features)
            Features of the instances (from `compute_features`).

        mu : array-like of shape (n_classes * n_features,) or (n_features,)
            Parameters (`numpy` array or `cvxpy` variable).

        Returns
        -------
        phi_mu : array-like of shape (n_samples, n_classes)
            Product of the feature mapping and the parameters.
        '''

        if self.binary_compact_:
            phi_mu = X_transform @ mu
            if isinstance(phi_mu, cvx.Expression):
                return cvx.vstack([phi_mu, -phi_mu]).T
            return np.column_stack([phi_mu, -phi_mu])
        if isinstance(mu, cvx.Expression):
            mu_mat = cvx.reshape(mu, (self.n_classes, X_transform.shape[1]),
                                 order='C')
        else:
            mu_mat = mu.reshape(self.n_classes, -1)
        return X_transform @ mu_mat.T

    def weighted_features(self, X_transform, weights):
        '''
        Sum of the feature mappings of the instances and labels weighted by
        the given weights, :math:`\sum_i \sum_y w_{i,y} \Phi(x_i, y)`,
        using the features of the instances.

        Parameters
        ----------
        X_transform : array-like of shape (n_samples, n_features)
            Features of the instances (from `compute_features`).

        weights : array-like of shape (n_samples, n_classes)
            Weight of each instance and label.

        Returns
        -------
        features : array-like of shape (n_classes * n_features,) or (n_features,)
            Weighted sum of the feature mappings.
        '''

        if self.binary_compact_:
            return np.asarray(X_transform.T @ (weights[:, 0] - weights[:, 1])
                              ).ravel()
        return np.asarray((X_transform.T @ weights).T).ravel()

    def varphi(self, phi_mu):
        '''
        Function :math:`\varphi` of the 0-1 loss for each instance, the
        maximum over the subsets S of labels of
        :math:`1 + (\sum_{y \in S} \Phi(x, y)^T \mu - 1) / |S|`,
        attained at the labels with the largest scores, and the weights of
        the labels in its subgradient (1 / |S| for the labels in S).

        Parameters
        ----------
        phi_mu : array-like of shape (n_samples, n_classes)
            Product of the feature mapping and the parameters for each
            instance and label (see `scores`).

        Returns
        -------
        varphi_value : array-like of shape (n_samples,)
            The value of :math:`\varphi` for each instance.

        weights : array-like of shape (n_samples, n_classes)
            Weight of each label in the subgradient of :math:`\varphi`.
        '''

        n = phi_mu.shape[0]
        # For a fixed cardinality r, the subset of labels maximizing the
        # sum of scores is the r labels with the largest scores, so the
        # maximum over the subsets is a maximum over r of the cumulative
        # sums of the scores sorted in decreasing order (as in MRC).
        indices = np.argsort(-phi_mu, axis=1)
        cumsum = np.cumsum(np.take_along_axis(phi_mu, indices, axis=1), axis=1)
        r_vals = np.arange(1, self.n_classes + 1)
        values = (cumsum - 1) / r_vals + 1
        r = np.argmax(values, axis=1) + 1

        # Weights 1 / |S| of the labels in the subset S
        weights = np.zeros_like(phi_mu, dtype=np.float64)
        in_S = r_vals[np.newaxis, :] <= r[:, np.newaxis]
        np.put_along_axis(weights, indices, in_S / r[:, np.newaxis], axis=1)

        return values[np.arange(n), r - 1], weights

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

        X_transform = self.compute_features(X)
        phi_mu = self.scores(X_transform, self.mu_)

        if self.loss == '0-1':

            # Compute the conditional probabilities
            varphi, _ = self.varphi(phi_mu)
            hy_x = np.clip(1 + phi_mu + varphi[:, np.newaxis], 0., None)

            # Normalize the probabilities
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

            v = phi_mu

            # Normalizing conditional probabilities using log-sum-exp trick
            # For each class i, compute: 1 / sum_j(exp(v_j - v_i))
            # This is numerically stable as we subtract v_i before exp
            hy_x = np.vstack(list(np.sum(np.exp(v - np.tile(v[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes))).transpose()
            hy_x = np.reciprocal(hy_x)

            # Set the approach for prediction to deterministic
            # if not provided by user.
            if self.deterministic is None:
                self.deterministic = True

        return hy_x
