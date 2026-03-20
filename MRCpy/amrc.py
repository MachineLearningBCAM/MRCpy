"""
Adaptative Minimax Risk Classification. Copyright (C) 2022 Veronica Alvarez

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
"""

import itertools
import math

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the AMRC super class
from MRCpy import BaseMRC
from MRCpy.phi import BasePhi, RandomFourierPhi


class AMRC(BaseMRC):
    r'''
    Adaptive Minimax Risk Classifier

    This class implements Adaptive Minimax Risk Classifiers (AMRCs)
    proposed in [1]_ for online learning with streaming data. Training
    samples are fed sequentially and the classification rule is updated
    every time a new sample is provided.

    At each time step :math:`t`, the classifier solves the minimax risk
    problem:

    .. math::

        \mathrm{h}_t^{\mathcal{U}_t} \in \arg\min_{\mathrm{h}}
        \max_{\mathrm{p} \in \mathcal{U}_t} \ell(\mathrm{h}, \mathrm{p})

    which finds the classifier :math:`\mathrm{h}_t` that minimizes the
    worst-case expected 0-1 loss over a time-varying uncertainty set
    :math:`\mathcal{U}_t` of distributions.

    The uncertainty set :math:`\mathcal{U}_t` follows the same form as
    :math:`\mathcal{U}_1` (no marginal constraint), with time-varying
    parameters:

    .. math::

        \mathcal{U}_t = \left\{ \mathrm{p} :
        \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)]
        - \boldsymbol{\tau}_t \right| \leq \boldsymbol{\lambda}_t \right\}

    where :math:`\boldsymbol{\tau}_t` and :math:`\boldsymbol{\lambda}_t`
    are updated at each time step via Kalman filtering to track the
    time-varying underlying distribution.

    AMRC provides adaptation to concept drift (change in the underlying
    distribution of the data) by means of a multivariate and high-order
    tracking of the distribution. The mean vector estimates
    :math:`\boldsymbol{\tau}_t` and confidence vectors
    :math:`\boldsymbol{\lambda}_t` are obtained from a linear dynamical
    system model with Kalman filter recursions. The kinematic model
    order (controlled by the ``order`` parameter) determines the
    complexity of the tracking: order 0 corresponds to a zero-order
    model, order 1 to white noise acceleration, and order 2 to Wiener
    process acceleration.

    AMRC provides computable tight performance guarantees at learning.
    The instantaneous error probability satisfies
    :math:`R(\mathrm{h}_t) \leq R(\mathcal{U}_t) + \alpha_t`, and
    the accumulated mistakes are bounded using the sum of minimax risks
    and a confidence term controlled by the ``delta`` parameter.

    It implements the 0-1 loss function and can be used with linear and
    Random Fourier features.

    See [1]_ for further.

    Parameters
    ----------
    n_classes : `int`
        Number of different possible labels for an instance.

    deterministic : `bool`, default = `True`
        Whether the prediction of the labels
        should be done in a deterministic way (given a fixed `random_state`
        in the case of using random Fourier features).

    loss : `str` {'0-1'}, default = '0-1'
        Type of loss function to use for the risk minimization.
        AMRC supports 0-1 loss only.
        0-1 loss quantifies the probability of classification error at
        a certain example for a certain rule.

    unidimensional : `bool`, default = `False`
        Whether to model change in the variables unidimensionally or not.
        When ``True``, the kinematic model order is forced to 0 and
        tracking is performed independently per dimension.
        Available for comparison purposes.

    delta : `float`, default = `0.05`
        Significance level for the upper bound on accumulated mistakes.
        Lower values produce higher (more conservative) bounds.

    order : `int`, default = `1`
        Order of the kinematic model used for tracking the time-varying
        distribution. Controls the complexity of the Kalman filter:

        - ``0``: Zero-order model (constant state).
        - ``1``: White noise acceleration model.
        - ``2``: Wiener process acceleration model.

        Ignored when ``unidimensional=True`` (forced to 0).

    W : `int`, default = `200`
        Window size for estimating label probabilities. The model uses
        the last ``W`` samples for sliding-window probability estimation.

    N : `int`, default = `100`
        Maximum number of subgradients retained for the local
        approximation of :math:`\varphi(\cdot)` in the optimization.

    max_iters : `int`, default = `2000`
        Maximum number of iterations for the accelerated subgradient
        method used to solve the minimax risk optimization.

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier' and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' feature mapping,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

    random_state : `int`, RandomState instance, default = `None`
        Random seed used when using 'fourier' for feature mappings
        to produce the random weights.

    fit_intercept : `bool`, default = `False`
        Whether to calculate the intercept for MRCs.
        If set to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    **phi_kwargs : Additional parameters for feature mappings.
            Groups the multiple optional parameters
            for the corresponding feature mappings(``phi``).

            For example in case of fourier features,
            the number of features is given by ``n_components``
            parameter which can be passed as argument
            ``AMRC(n_classes=2, phi='fourier', n_components=500)``

            The list of arguments for each feature mappings class
            can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : `bool`
        Whether the classifier is fitted i.e., the parameters are learnt.

    mu : `array`-like of shape (`m`, 1)
        Classifier parameters learnt by the minimax risk optimization.

    varphi : `float`
        Value of the :math:`\varphi` function at the current solution.

    sample_counter : `int`
        Number of samples processed so far.

    params_ : `dict`
        Dictionary storing optimization state including Kalman filter
        parameters, subgradient approximation matrices, and upper bounds.

    See Also
    --------
    MRCpy.MRC : MRC using uncertainty set :math:`\mathcal{U}_1` without marginal constraints.
    MRCpy.CMRC : CMRC using uncertainty set :math:`\mathcal{U}_2` with marginal constraints.

    References
    ----------
    .. [1] Álvarez, V., Mazuelas, S., & Lozano, J.A. (2022). Minimax
           Classification under Concept Drift with Multidimensional
           Adaptation and Performance Guarantees. In Proceedings of the
           39th International Conference on Machine Learning, pp. 486-499.

    .. [2] Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax
           Classification with 0-1 Loss and Performance Guarantees.
           Advances in Neural Information Processing Systems, 33, 302-312.

    .. [3] Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized
           Maximum Entropy for Supervised Classification. IEEE Transactions
           on Information Theory, 68(4), 2530-2550.
    '''

    def __init__(self, n_classes, loss='0-1',
                 deterministic=True,
                 random_state=None,
                 phi='linear',
                 unidimensional=False,
                 delta = 0.05,
                 order=1,
                 W=200,
                 N=100,
                 fit_intercept=False,
                 max_iters=2000, **phi_kwargs):
        self.n_classes = n_classes
        self.unidimensional = unidimensional
        self.order = order
        self.delta = delta
        if self.unidimensional:
            self.order = 0
        self.W = W
        self.N = N
        self.max_iters = max_iters
        self.Y = np.zeros(self.W)
        self.p = np.zeros((self.n_classes, self.W))
        self.sample_counter = 0
        self.params_ = {}
        if 'one_hot' in phi_kwargs and not phi_kwargs['one_hot']:
            raise ValueError('AMRC does not support one_hot=False')
        else:
            phi_kwargs['one_hot'] = True
        if loss != '0-1':
            raise ValueError('AMRC only support loss=0-1')
        super().__init__(loss=loss,
                         deterministic=deterministic,
                         random_state=random_state,
                         fit_intercept=fit_intercept,
                         phi=phi, **phi_kwargs)

    def tracking(self, feature, y, p, s):
        '''
        Track uncertainty sets over time.

        This function obtains mean vector estimates and confidence vectors
        by updating the Kalman filter state with a new observation.

        Parameters
        ----------
        feature : `array`-like of shape (1, `m`)
            Feature vector for the current sample.

        y : `int`
            Label of the current sample.

        p : `array`-like of shape (`n_classes`,)
            Estimated class probabilities from the sliding window.

        s : `array`-like of shape (`n_classes`,)
            Standard deviation of the class probability estimates.

        Returns
        -------
        tau_ : `array`-like of shape (`m`, 1)
            Updated mean vector estimate.

        lambda_ : `array`-like of shape (`m`, 1)
            Updated confidence vector.

        params_ : `dict`
            Updated optimization parameters containing:

            - ``Ht`` : Transition matrix
            - ``eta`` : Updated state vectors
            - ``Sigma`` : Updated mean squared error matrices
            - ``eta0``, ``Sigma0``, ``epsilon`` : Parameters for noise variance updates
            - ``Q``, ``R`` : Variances of noise processes
        '''
        Ht = self.params_['Ht']
        eta = self.params_['eta']
        Sigma = self.params_['Sigma']
        eta0 = self.params_['eta0']
        Sigma0 = self.params_['Sigma0']
        epsilon = self.params_['epsilon']
        Q = self.params_['Q']
        R = self.params_['R']
        e1 = np.zeros((1, self.order + 1))
        e1[0, 0] = 1

        m = len(feature[0])
        n_classes = len(p)
        d = m / n_classes
        alpha = 0.3
        tau_ = np.zeros((m, 1))
        lambda_ = np.zeros((m, 1))
        if self.unidimensional:
            KK = np.zeros((m, 1))
            for i in range(m):
                innovation = feature[0, i] - eta[0, i]
                aa = alpha * R[i, 0] + (1 - alpha) * \
                    (np.dot(epsilon[i], epsilon[i]) +
                     np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1)))
                R[i] = aa[0]
                a = (np.dot(Sigma[i, :, :], np.transpose(e1)))
                b = np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1)) + \
                    R[i, :]
                KK[i] = a / b
            K = np.mean(KK)
            for i in range(m):
                eta0[:, i] = eta[:, i] + K * innovation
                Sigma0[i, :, :] = np.dot((np.identity(self.order + 1) -
                                          np.dot(K, e1)), Sigma[i, :, :])
                Q[i, :, :] = alpha * Q[i, :, :] + (1 - alpha) * \
                    np.dot(innovation * innovation * K, np.transpose(K))
                epsilon[i] = feature[0, i] - eta0[0, i]
                eta[:, i] = np.dot(Ht, eta0[:, i])
                Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]),
                                        np.transpose(Ht)) + Q[i, :, :]
                tau_[i, 0] = (1 / n_classes) * eta[0, i]
                lmb_eta = np.sqrt(Sigma[i, 0, 0])
                lambda_[i, 0] = np.mean(lmb_eta)
        elif not self.unidimensional:
            for i in range(m):
                if i > y * d - 1 and i < (y + 1) * d + 1:
                    innovation = feature[0, i] - eta[0, i]
                    aa = alpha * R[i, 0] + (1 - alpha) * \
                        (np.dot(epsilon[i], epsilon[i]) +
                         np.dot(np.dot(e1, Sigma[i, :, :]), np.transpose(e1)))
                    R[i] = aa[0]
                    a = (np.dot(Sigma[i, :, :], np.transpose(e1)))
                    b = np.dot(np.dot(e1, Sigma[i, :, :]),
                               np.transpose(e1)) + R[i, :]
                    K = a / b
                    eta0[:, i] = eta[:, i] + np.transpose(K[:] * innovation)
                    Sigma0[i, :, :] = np.dot((np.identity(self.order + 1) -
                                              np.dot(K, e1)), Sigma[i, :, :])
                    Q[i, :, :] = alpha * Q[i, :, :] + (1 - alpha) * \
                        np.dot(innovation * innovation * K, np.transpose(K))
                    epsilon[i] = feature[0, i] - eta0[0, i]
                    eta[:, i] = np.dot(Ht, eta0[:, i])
                    Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]),
                                            np.transpose(Ht)) + Q[i, :, :]
                    tau_[i, 0] = p[y] * eta[0, i]
                    lmb_eta = np.sqrt(Sigma[i, 0, 0])
                    lambda_[i, 0] = np.sqrt((lmb_eta ** 2 + eta[0, i] ** 2) *
                                            (s[y] ** 2 + p[y] ** 2) -
                                            ((eta[0, i]) ** 2) *
                                            (p[y] ** 2))
                else:
                    eta[:, i] = np.dot(Ht, eta0[:, i])
                    Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]),
                                            np.transpose(Ht)) + Q[i, :, :]
                    tau_[i, 0] = (p[int((i) / d)]) * eta[0, i]
                    lmb_eta = np.sqrt(Sigma[i, 0, 0])
                    lambda_[i, 0] = np.sqrt((lmb_eta ** 2 + eta[0, i] ** 2) *
                                            (s[int((i) / d)] ** 2 +
                                             p[int((i) / d)] ** 2) -
                                            ((eta[0, i]) ** 2) *
                                            (p[int((i) / d)] ** 2))
        else:
            raise ValueError('The given value for parameter unidimensional ' +
                             'is not a boolean')

        params_ = {'Ht': Ht,
                   'eta': eta,
                   'Sigma': Sigma,
                   'eta0': eta0,
                   'Sigma0': Sigma0,
                   'epsilon': epsilon,
                   'Q': Q,
                   'R': R}

        return tau_, lambda_, params_

    def initialize_tracking(self, m):
        '''
        Initialize the tracking stage.

        This function initializes mean vector estimates, confidence vectors,
        and defines matrices and vectors that are used to update mean vector
        estimates and confidence vectors via Kalman filtering.

        Parameters
        ----------
        m : `int`
            Length of the feature vector (dimensionality of the state).

        Returns
        -------
        params_ : `dict`
            Initialized optimization parameters containing:

            - ``Ht`` : Transition matrix of shape (`order+1`, `order+1`)
            - ``eta`` : State vectors of shape (`order+1`, `m`)
            - ``Sigma`` : Mean squared error matrices of shape (`m`, `order+1`, `order+1`)
            - ``eta0``, ``Sigma0`` : Prior state and covariance estimates
            - ``epsilon`` : Innovation residuals of shape (`m`, 1)
            - ``Q`` : Process noise covariance of shape (`m`, `order+1`, `order+1`)
            - ``R`` : Observation noise variance of shape (`m`, 1)
        '''

        e1 = np.ones((1, self.order + 1))
        for i in range(1, self.order + 1):
            e1[0, i] = 0
        deltat = 1
        variance_init = 0.001
        Ht = np.identity(self.order + 1)
        for i in range(self.order):
            for j in range(i + 1, self.order + 1):
                Ht[i, j] = pow(deltat, j - i) / math.factorial(j - i)
        eta0 = np.zeros((self.order + 1, m))
        eta = np.zeros((self.order + 1, m))
        Sigma0 = np.zeros((m, self.order + 1, self.order + 1))
        Sigma = np.zeros((m, self.order + 1, self.order + 1))
        Q = np.zeros((m, self.order + 1, self.order + 1))
        R = np.zeros((m, 1))
        epsilon = np.zeros((m, 1))

        for i in range(m):
            for j in range(self.order + 1):
                Sigma0[i, j, j] = 1
                Q[i, j, j] = variance_init
            R[i] = variance_init
            epsilon[i] = - eta0[0, i]
            eta[:, i] = np.dot(Ht, eta0[:, i])
            Sigma[i, :, :] = np.dot(np.dot(Ht, Sigma0[i, :, :]),
                                    np.transpose(Ht)) + Q[i, :, :]

            params_ = {'Ht': Ht,
                       'eta': eta,
                       'Sigma': Sigma,
                       'eta0': eta0,
                       'Sigma0': Sigma0,
                       'epsilon': epsilon,
                       'Q': Q,
                       'R': R}
        return params_

    def minimax_risk(self, x, tau_, lambda_, n_classes):
        '''
        Learn classifier parameters via minimax risk optimization.

        This function efficiently learns classifier parameters by solving
        the minimax risk optimization problem using a subgradient approach.

        Parameters
        ----------
        x : `array`-like of shape (`n_dimensions`,)
            Training instance used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like of shape (`m`, 1)
            Mean estimates for the expectations of feature mappings.

        lambda_ : `array`-like of shape (`m`, 1)
            Variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
            Number of labels in the dataset.

        Returns
        -------
        self :
            Fitted estimator with updated ``mu``, ``is_fitted_``,
            ``varphi``, and ``params_`` attributes.
        '''
        self.n_classes = n_classes
        self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)
        self.lambda_ = check_array(lambda_, accept_sparse=True,
                                   ensure_2d=False)
        F = self.params_['F']
        h = self.params_['h']
        theta = 1
        theta0 = 1
        muaux = self.mu
        R_Ut = 0
        M = np.zeros((n_classes, len(self.mu)))
        for j in range(n_classes):
            M[j, :] = self.phi.eval_xy(x.reshape((1, -1)), [j])
        for j in range(n_classes):
            aux = list(itertools.combinations([*range(n_classes)],
                                              j + 1))
            for k in range(np.size(aux, 0)):
                idx = np.zeros((1, n_classes))
                a = aux[k]
                for mm in range(len(a)):
                    idx[0, a[mm]] = 1
                a = (np.dot(idx, M)) / (j + 1)
                b = np.size(F, 0)
                F2 = np.zeros((b + 1, len(self.mu)))
                h2 = np.zeros((b + 1, 1))
                for mm in range(b):
                    for jj in range(len(self.mu)):
                        F2[mm, jj] = F[mm, jj]
                    h2[mm, :] = h[mm, :]
                F2[-1, :] = a
                b = -1 / (j + 1)
                h2[-1, 0] = b
                F = F2
                h = h2
        if self.sample_counter == 0:
            F = np.delete(F, 0, 0)
            h = np.delete(h, 0, 0)
        v = np.dot(F, muaux) + h
        varphi = max(v)[0]
        regularization = sum(self.lambda_ * abs(muaux))
        R_Ut_best_value = 1 - np.dot(np.transpose(self.tau_), muaux)[0] + \
            varphi + regularization
        F_count = np.zeros((len(F[:, 0]), 1))
        for i in range(self.max_iters):
            muaux = self.params_['w'] + theta * ((1 / theta0) - 1) * \
                (self.params_['w'] - self.params_['w0'])
            v = np.dot(F, muaux) + h
            varphi = max(v)[0]
            idx_mv = np.where(v == varphi)
            if len(idx_mv[0]) > 1:
                fi = F[[idx_mv[0][0]], :]
                F_count[[idx_mv[0][0]]] = F_count[[idx_mv[0][0]]] + 1
            else:
                fi = F[idx_mv[0], :]
                F_count[idx_mv[0]] = F_count[idx_mv[0]] + 1
            subgradient_regularization = np.multiply(self.lambda_,
                                                     np.sign(muaux))
            regularization = np.sum(np.multiply(self.lambda_, np.abs(muaux)))
            g = - self.tau_ + np.transpose(fi) + subgradient_regularization
            theta0 = theta
            theta = 2 / (i + 2)
            alpha = 1 / ((i + 2) ** (3 / 2))
            self.params_['w0'] = self.params_['w']
            self.params_['w'] = muaux - np.multiply(alpha, g)
            R_Ut = 1 - np.dot(np.transpose(self.tau_), muaux)[0] + \
                varphi + regularization
            if R_Ut < R_Ut_best_value:
                R_Ut_best_value = R_Ut
                self.mu = muaux
        v = np.dot(F, muaux) + h
        varphi = max(v)[0]
        regularization = np.sum(np.multiply(self.lambda_,
                                            np.abs(self.params_['w'])))
        R_Ut = 1 - np.dot(np.transpose(self.tau_), self.params_['w'])[0] + \
            varphi + regularization
        if R_Ut < R_Ut_best_value:
            R_Ut_best_value = R_Ut
            self.mu = self.params_['w']
        if len(F[:, 0]) > self.N:
            idx_F_count = np.where(F_count == 0)
            if len(idx_F_count) > len(F[:, 0]) - self.N:
                for j in range(len(idx_F_count[0]) - self.N):
                    t = len(idx_F_count[0]) - 1 - j
                    F = np.delete(F, idx_F_count[0][t], idx_F_count[1][t])
                    h = np.delete(h, idx_F_count[0][t], 0)
            else:
                for j in range(len(idx_F_count[0]) - len(F[:, 0]) +
                               self.N):
                    t = len(idx_F_count[0]) - 1 - j
                    F = np.delete(F, idx_F_count[0][t], idx_F_count[1][t])
                    h = np.delete(h, idx_F_count[0][t], 0)
        self.is_fitted_ = True
        self.varphi = varphi
        self.params_['R_Ut'] = R_Ut
        self.params_['sum_R_Ut'] = self.params_['sum_R_Ut'] + R_Ut
        self.params_['F'] = F
        self.params_['h'] = h
        return self

    def fit(self, x, y, X_=None):
        '''
        Fit the AMRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.
        Designed for online learning where samples are fed sequentially.

        Parameters
        ----------
        x : `array`-like of shape (`n_dimensions`,)
            Training instance used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

        y : `int`
            Label corresponding to the training instance
            used only to compute the expectation estimates.

        X_ : None
            Unused in AMRC. Kept for API compatibility with BaseMRC.

        Returns
        -------
        self :
            Fitted estimator
        '''

        x = check_array(x, accept_sparse=True, ensure_2d=False)

        # Calculate the length m of the feature vector
        if self.sample_counter == 0:
            if self.phi == 'linear':
                self.phi = BasePhi(n_classes=self.n_classes,
                                   fit_intercept=self.fit_intercept,
                                   **self.phi_kwargs)
            elif self.phi == 'fourier':
                self.phi = RandomFourierPhi(n_classes=self.n_classes,
                                            fit_intercept=self.fit_intercept,
                                            random_state=self.random_state,
                                            **self.phi_kwargs)
            elif not isinstance(self.phi, BasePhi):
                raise ValueError('Unexpected feature mapping type ... ')

            # Each time we call phi.fit the random weights in random fourier
            # featurees change
            self.phi.fit(x.reshape((1, -1)), [y])
            m = self.phi.len_

            # Initialize classifier parameter, upper bounds vector, and matrix
            # and vector that are used to obtain local approximations of
            # varphi function
            self.params_['F'] = np.zeros((1, m))
            self.params_['h'] = np.zeros((1, 1))
            self.mu = np.zeros((m, 1))
            self.params_['w'] = np.zeros((m, 1))
            self.params_['w0'] = np.zeros((m, 1))
            self.params_['R_Ut'] = 0
            self.params_['sum_R_Ut'] = 0

            params_ = self.initialize_tracking(m)
            self.params_ = {**self.params_, **params_}

        # Initialize mean vector estimate
        # Estimating probabilities
        sample_idx = self.sample_counter % self.W
        self.Y[sample_idx] = y
        end = min(self.sample_counter + 1, self.W)
        s = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            self.p[i, sample_idx] = \
                np.mean(self.Y[:end] == i)
            if end > 1:
                s[i] = np.std(self.p[i, :end]) * \
                    np.sqrt(end) / np.sqrt(end - 1)
            else:
                s[i] = 0

        # Feature vector
        feature = self.phi.eval_xy(x.reshape((1, -1)), [y])

        # Update mean vector estimate and confidence vector
        tau_, lambda_, params_ = \
            self.tracking(feature, y, self.p[:, sample_idx], s)

        self.params_ = {**self.params_, **params_}

        # Update classifier parameter and obtain upper bound
        self.minimax_risk(x, tau_, lambda_, self.n_classes)

        self.sample_counter = self.sample_counter + 1

        self.is_fitted_ = True
        return self

    def predict(self, X):
        '''
        Predict the class for a new instance using the fitted model.

        Returns the predicted class for the given instance in `X`
        using the probabilities given by the function `predict_proba`.

        Parameters
        ----------
        X : `array`-like of shape (`n_dimensions`,)
            Test instance to predict by the AMRC model.

        Returns
        -------
        y_pred : `int`
            Predicted label corresponding to the given instance.
        '''

        X = check_array(X, accept_sparse=True, ensure_2d=False)
        check_is_fitted(self, "is_fitted_")

        # Get the prediction probabilities for the classifier
        proba = self.predict_proba(X)

        if self.deterministic:
            y_pred = np.argmax(proba)
        else:
            np.random.seed(self.random_state)
            y_pred = np.random.choice(self.n_classes, p=proba)

        # Check if the labels were provided for fitting
        # (labels might be omitted if fitting is done through minimax_risk)
        # Otherwise return the default labels i.e., from 0 to n_classes-1.
        if hasattr(self, 'classes_'):
            y_pred = np.asarray([self.classes_[label] for label in y_pred])

        return y_pred

    def predict_proba(self, x):
        '''
        Compute conditional probabilities for each class.

        Parameters
        ----------
        x : `array`-like of shape (`n_dimensions`,)
            Testing instance for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        h : `ndarray` of shape (`n_classes`,)
            Conditional probabilities :math:`p(y|x)` corresponding to
            the predictions for each class.
        '''

        M = np.zeros((self.n_classes, len(self.mu)))
        c = np.zeros(self.n_classes)
        for j in range(self.n_classes):
            M[j, :] = self.phi.eval_xy(x.reshape((1, -1)), [j])
        for j in range(self.n_classes):
            c[j] = max([np.dot(M[j, :], self.mu)[0] - self.varphi, 0])
        cx = sum(c)
        h = np.zeros(self.n_classes)
        for j in range(self.n_classes):
            if cx == 0:
                h[j] = 1 / self.n_classes
            else:
                h[j] = c[j] / cx
        return h

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper_bound : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        return self.params_['R_Ut']

    def get_upper_bound_accumulated(self):
        '''
        Returns the upper bound on the accumulated mistakes 
        of the fitted classifier.

        Returns
        -------
        upper_bound_accumulated : `float`
            Upper bound of the accumulated for the fitted classifier.
        '''

        return ((self.params_['sum_R_Ut'] + \
                np.sqrt(2 * self.sample_counter * np.log(1 / self.delta))) / self.sample_counter)