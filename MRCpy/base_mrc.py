"""
Super class for Minimax Risk Classifiers. Copyright (C) 2021 Kartheek Bondugula

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
"""

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
    r'''
    Base class for different minimax risk classifiers

    This class is a parent class for different MRCs
    implemented in the library.
    It defines the different parameters and the layout.

    .. seealso::

        For more information about MRC, one can refer to the
        following resources:

        - `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
          Minimax Classification with
          0-1 Loss and Performance Guarantees.
          Advances in Neural Information Processing
          Systems, 33, 302-312.
          <https://proceedings.neurips.cc/paper_files/paper/2020
          /file/02f657d55eaf1c4840ce8d66fcdaf90c-Paper.pdf>`_

        - `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
          Generalized Maximum Entropy for Supervised Classification.
          IEEE Transactions on Information Theory, 68(4), 2530-2550.
          <https://ieeexplore.ieee.org/stamp/
          stamp.jsp?arnumber=9682746>`_

        - `Mazuelas, S., Romero, M., Grunwald, P. (2023).
          Minimax Risk Classifiers with 0-1 Loss.
          Journal of Machine Learning Research, 24(208), 1-48.
          <https://jmlr.org/papers/volume24/22-0339/22-0339.pdf>`_

        - `Bondugula, K., Mazuelas, S., & Pérez,
          A. (2021). MRCpy: A Library for Minimax Risk Classifiers.
          arXiv preprint arXiv:2108.01952.
          <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    loss : str, default='0-1'
        Type of loss function to use for the risk
        minimization.
        The options are 0-1 loss and logarithmic loss.

        - ``'0-1'`` : 0-1 loss quantifies the probability of classification
          error at a certain example for a certain rule.
        - ``'log'`` : Log-loss quantifies the minus log-likelihood at a
          certain example for a certain rule.

    s : float, default=0.3
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

    deterministic : bool, default=True
        Whether the prediction of the labels
        should be done in a deterministic way (given a fixed ``random_state``
        in the case of using random Fourier or random ReLU features).

    random_state : int, RandomState instance, default=None
        Random seed used when 'fourier' and 'relu' options for feature mappings
        are used to produce the random weights.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for MRCs.
        If set to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    phi : str or BasePhi instance, default='linear'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.
        The users can also implement their own feature mapping object
        (should be a ``BasePhi`` instance) and pass it to this argument.
        Note that when using 'fourier' or 'relu' feature mappings,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class ``BasePhi``.

        'fourier'
            It uses Random Fourier Feature map. See class ``RandomFourierPhi``.

        'relu'
            It uses Rectified Linear Unit (ReLU) features.
            See class ``RandomReLUPhi``.

        'threshold'
            It uses Feature mappings obtained using a threshold.
            See class ``ThresholdPhi``.

    **phi_kwargs : Additional parameters for feature mappings.
        Groups the multiple optional parameters
        for the corresponding feature mappings (``phi``).

        For example in case of fourier features,
        the number of features is given by ``n_components``
        parameter which can be passed as argument -
        ``MRC(loss='log', phi='fourier', n_components=500)``

        The list of arguments for each feature mappings class
        can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the classifier is fitted i.e., the parameters are learnt.

    tau_mat : array-like of shape (n_classes, n_features)
        Mean estimates for the expectations of feature mappings.

    lambda_mat : array-like of shape (n_classes, n_features)
        Variance in the mean estimates for the expectations
        of the feature mappings.

    classes_ : array-like of shape (n_classes,)
        Labels in the given dataset.
        If the labels Y are not given during fit
        i.e., tau and lambda are given as input,
        then this array is None.
    '''

    def __init__(self,
                 loss='0-1',
                 s=0.3,
                 deterministic=True,
                 random_state=None,
                 fit_intercept=True,
                 phi='linear',
                 **phi_kwargs):

        self.loss = loss
        self.s = s
        self.deterministic = deterministic
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        # Feature mapping and its parameters
        self.phi = phi
        self.phi_kwargs = phi_kwargs
        # Solver list for cvxpy
        self.cvx_solvers = ['GUROBI', 'SCS', 'ECOS']

    def fit(self, X, Y, X_=None):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the ``minimax_risk`` function to solve the optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

            ``n_samples`` is the number of training samples and
            ``n_dimensions`` is the number of features.

        Y : array-like of shape (n_samples, 1), default=None
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        X_ : array-like of shape (n_samples2, n_dimensions), default=None
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
        X_transform = self.phi.transform(X)
        tau_mat = self.compute_tau(X_transform, Y)
        lambda_mat = self.compute_lambda(X_transform, Y, tau_mat)

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory 
        # while retaining competitive performance
        n_max = 5000

        # Check if separate instances are given for the optimization
        if X_ is not None:
            X_opt = check_array(X_.copy(), accept_sparse=True)
        elif X.shape[0] < n_max:
            X_opt = check_array(X.copy(), accept_sparse=True)
        else:
            # Use some of the training samples
            # for the optimization.
            n_per_class = int(n_max / n_classes)
            X_opt = X[:3, :]
            for i in range(n_classes):
                X_class = X[Y == i, :]
                X_opt = np.vstack((X_class[:n_per_class, :], X_opt))

        # Shuffle the instances
        np.random.seed(self.random_state)
        np.random.shuffle(X_opt)

        # Fit the MRC classifier
        self.minimax_risk(X_opt, tau_mat, lambda_mat, n_classes)

        return self

    def compute_features(self, X):
        '''
        Compute the features (without one-hot encoding) corresponding
        to instances given for learning the classifiers (in case of
        training) and prediction (in case of testing).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Instances to be converted to features.

        Returns
        -------
        X_transform : array-like of shape (n_samples, n_features)
            Feature corresponding with each instance.
        '''

        return self.phi.transform(X)

    def compute_tau(self, X_transform, Y):
        '''
        Compute mean estimate tau using the given training instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features corresponding with the training instances
            :math:`\psi(x)`.

        Y : array-like of shape (n_samples, 1), default=None
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        Returns
        -------
        tau_mat : array-like of shape (n_classes, n_features) or (1, n_features)
            Mean estimates :math:`\boldsymbol{\tau}` of the feature mappings.
            Shape is (n_classes, n_features) for multi-class problems,
            or (1, n_features) for binary classification.
        '''

        return self.phi.est_exp(X_transform, Y)

    def compute_lambda(self, X_transform, Y, tau_mat):
        '''
        Compute deviation in the mean estimate tau
        using the given training instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features corresponding with the training instances
            :math:`\psi(x)`.

        Y : array-like of shape (n_samples, 1), default=None
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        tau_mat : array-like of shape (n_classes, n_features)
            Mean estimates :math:`\boldsymbol{\tau}` corresponding with
            the expectations of feature mappings.

        Returns
        -------
        lambda_mat : array-like of shape (n_classes, n_features) or (1, n_features)
            Deviation estimates :math:`\boldsymbol{\lambda}` that define the uncertainty
            set constraints. Shape matches tau_mat.
        '''

        return (self.s * self.phi.est_std(X_transform, Y, tau_mat)) / np.sqrt(X_transform.shape[0])

    def minimax_risk(self, X, tau_mat, lambda_mat, n_classes):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Solves the minimax risk optimization problem
        for the corresponding variant of MRC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_mat : array-like of shape (n_classes, n_features)
            Mean estimates :math:`\boldsymbol{\tau}` corresponding with
            the expectations of feature mappings.

        lambda_mat : array-like of shape (n_classes, n_features)
            Inaccuracies :math:`\boldsymbol{\lambda}` in the mean estimates
            corresponding with the expectations of the feature mappings.

        n_classes : int
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


    def predict_proba(self, X):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Computes conditional probabilities corresponding
        to each class for the given unlabeled instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Testing instances for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        hy_x : array-like of shape (n_samples, n_classes)
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

    def compute_phi(self, X):
        '''
        Compute the feature mapping corresponding to instances given
        for learning the classifiers (in case of training) and
        prediction (in case of testing).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Instances to be converted to features.
        '''

        return self.phi.eval_x(X)

    def predict(self, X):
        '''
        Predicts classes for new instances using a fitted model.

        Returns the predicted classes for the given instances in ``X``
        using the probabilities given by the function ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted labels corresponding to the given instances.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Get the prediction probabilities for the classifier
        proba = self.predict_proba(X)

        if self.deterministic:
            y_pred = np.argmax(proba, axis=1)
        else:
            np.random.seed(self.random_state)
            y_pred = [np.random.choice(self.n_classes, size=1, p=pc)[0]
                      for pc in proba]

        # Check if the labels were provided for fitting
        # (labels might be omitted if fitting is done through minimax_risk)
        # Otherwise return the default labels i.e., from 0 to n_classes-1.
        if hasattr(self, 'classes_'):
            y_pred = np.asarray([self.classes_[label] for label in y_pred])

        return y_pred

    def error(self, X, Y):
        '''
        Return the mean error obtained for the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Y : array-like of shape (n_samples, 1), default=None
            Labels corresponding to the testing instances
            used to compute the error in the prediction.

        Returns
        -------
        error : float
            Mean error of the learned MRC classifier
        '''
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        Y_pred = self.predict(X)

        error = np.average(Y_pred != Y)
        return error
