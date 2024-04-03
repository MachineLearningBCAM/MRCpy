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
    '''
    Adaptative Minimax Risk Classifier

    The class AMRC implements the method Adaptative Minimimax Risk
    Classificafiers (AMRCs) proposed in :ref:`[1] <ref1>`. It is designed for
    online learning with streaming data. Training samples
    are fed sequentially and the classification rule is
    updated every time a new sample is provided.

    AMRC provides adaptation to concept drift (change in the
    underlying distribution of the data). Such concept drift is common
    in multiple applications including electricity price prediction,
    spam mail filtering, and credit card fraud detection. AMRC accounts
    for multidimensional time changes by means of a
    multivariate and high-order tracking of the time-varying
    underlying distribution. In addition, differently
    from conventional techniques, AMRCs
    can provide computable tight performance guarantees at learning.

    It implements 0-1 loss function and it can be used with linear and
    Random Fourier features.

    .. seealso:: For more information about AMRC, one can refer to the
        following paper:

                    [1] `Álvarez, V., Mazuelas, S., & Lozano, J. A. (2022).
                    Minimax Classification under Concept Drift with
                    Multidimensional Adaptation and Performance Guarantees.
                    International Conference on Machine Learning (ICML) 2022.

                    @InProceedings{AlvMazLoz22,
                    title = 	 {Minimax Classification under Concept Drift with
                                 Multidimensional Adaptation and
                                 Performance Guarantees},
                    author =     {{\'A}lvarez, Ver{\'o}nica
                                  and Mazuelas, Santiago
                                  and Lozano, Jose A},
                    booktitle = {Proceedings of the 39th
                                 International Conference on Machine Learning},
                    pages = 	 {486--499},
                    year = 	 {2022},
                    volume = 	 {162},
                    series = 	 {Proceedings of Machine Learning Research},
                    month = 	 {Jul},
                    publisher =    {PMLR},
                    }

    Parameters
    ----------
    n_classes : `int`
        Number of different possible labels for an instance.

    deterministic : `bool`, default = `True`
       Whether the prediction of the labels
       should be done in a deterministic way (given a fixed `random_state`
       in the case of using random Fourier or random ReLU features).

    loss : `str` {'0-1'}, default = '0-1'
        Type of loss function to use for the risk minimization.
        AMRC supports 0-1 loss.
        0-1 loss quantifies the probability of classification error at
        a certain example for a certain rule.

    unidimensional : `bool`, default = False
        Whether to model change in the variables unidimensionally or not.
        Available for comparison purposes.

    delta : `float`, default = 0.05
        Significance of the upper bound on the accumulated mistakes.
        Lower values will produce higher values for bounds.

    order : `int`, default = 1
        Order of the subgradients used in optimization.

    W : `int`, default = 200
        Window size. The model uses the last `W` samples for fitting the model.

    N : `int`, default = 100
        Number of subgradients used for optimization.

    max_iters : `int`, default = `2000`
        Maximum number of iterations to use
        for finding the solution of optimization in
        the subgradient approach.

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.
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

    fit_intercept : `bool`, default = `True`
        Whether to calculate the intercept for MRCs
        If set to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    **phi_kwargs : Additional parameters for feature mappings.
            Groups the multiple optional parameters
            for the corresponding feature mappings(`phi`).

            For example in case of fourier features,
            the number of features is given by `n_components`
            parameter which can be passed as argument
            `AMRC(phi='fourier', n_components=500)`

            The list of arguments for each feature mappings class
            can be found in the corresponding documentation.
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
        Tracking uncertainty sets

        This function obtains mean vector estimates and confidence vectors

        Input
        -----

        feature: feature vector

        y: new label

        p: `float`
            Probability

        s: `float`
            Standard deviation

        Output
        ------

        tau_: mean vector estimate

        lambda_: confidence vector

        params_: `dict`
            Optimization parameters

                eta: updated mean vector estimate

                Sigma: updated mean quadratic error matrix

                eta0, Sigma0, epsilon: parameters required to update variances
                of noise processes

                Q, R: variances of noise processes
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
        Initialize tracking stage

        This function initializes mean vector estimates, confidence vectors,
        and defines matrices and vectors that are used to update mean vector
        estimates and confidence vectors.

        Attributes
        ----------

        m: length of mean vector estimate

        Output
        ------
        params_: `dict`
            Optimization parameters
                Ht: transition matrix

                e1: vector with 1 in the first component and 0 in the
                remainning components

                eta: state vectors

                Sigma: mean squared error matrices

                eta0, Sigma0, epsilon: parameters required to obtain variances
                of noise processes

                Q, R: variances of noise processes
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
        Learning

        This function efficiently learns classifier parameters

        Input
        -----

        X : `array`-like
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like
            Mean estimates
            for the expectations of feature mappings.

        lambda_ : `array`-like
            Variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
            Number of labels in the dataset.

        Output
        ------

        self :
            Fitted estimator
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

        Parameters
        ----------
        X : `array`-like of shape (`n_features`)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.


        Y : `int`, default = `None`
            Label corresponding to the training instance
            used only to compute the expectation estimates.

        X_ : None
            Unused in AMRC

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

            # Initialize mean vector estimate
            params_ = self.initialize_tracking(m)
            self.params_ = {**self.params_, **params_}

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
        Predicts classes for new instances using a fitted model.

        Returns the predicted classes for the given instances in `X`
        using the probabilities given by the function `predict_proba`.

        Parameters
        ----------
        X : `array`-like of shape (`n_features`)
            Test instance for to predict by the AMRC model.

        Returns
        -------
        y_pred : `int`
            Predicted labels corresponding to the given instances.
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
        Conditional probabilities corresponding to each class
        for each unlabeled input instance

        Parameters
        ----------
        x : `array`-like of shape (`n_dimensions`)
            Testing instance for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        h : `ndarray` of shape (`n_classes`)
            Probabilities :math:`(p(y|x))` corresponding to the predictions
            for each class.

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