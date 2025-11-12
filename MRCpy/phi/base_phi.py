'''Super class for the feature mapping functions.'''

import numpy as np
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class BasePhi():

    '''
    Base class for feature mappings

    The class provides a base for different feature mapping functions
    that can be used with the MRC.
    This class provides definition for some utility functions that are
    used by the MRCs in the library. It corresponds
    to the usual identity feature map referred to as Linear feature map.

    To see an example of how to extend the class `BasePhi` to implement
    yout own feature mapping see :ref:`this example <ex_phi>`.



    .. note:: This is a base class for all the feature mappings.
        To create a new feature mapping that can be used with MRC objects,
        the user can extend this class and then implement the functions -

            1) `fit` - learns the required parameters for feature
                transformation
            2) `transform` - transforms the input instances to the features

        The above functions are principal components
        for different feature transformation.
        Apart from these functions, the users can also re-define other
        functions
        in this class according to their need.

        The definition of `fit` and `transform` in this class correspond to the
        usual identity feature map referred to as **Linear feature map**.

        The `transform` function is only used by the `eval_xy` function
        of this class to get the one-hot encoded features.
        If the user defines his own `eval_xy` function
        that returns the features directly without the need
        of `transform` function, then the `transform` function can be omitted.

    .. seealso:: For more information about MRC, one can refer to the following
            resources:

                    [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
                    Minimax Classification with 0-1 Loss and Performance
                    Guarantees. Advances in Neural Information Processing
                    Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

                    [2] `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
                    Generalized Maximum Entropy for Supervised Classification.
                    arXiv preprint arXiv:2007.05447.
                    <https://arxiv.org/abs/2007.05447>`_

                    [3] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021).
                    MRCpy: A Library for Minimax Risk Classifiers.
                    arXiv preprint arXiv:2108.01952.
                    <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    n_classes : `int`
        Number of classes in the dataset

    fit_intercept : `bool`, default = `True`
        Whether to calculate the intercept.
        If set to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered)

    one_hot : `bool`, default = `False`
        Controls the method used for evaluating the features of the
        given instances in the binary case.
        Only applies in the **binary case**, namely, only when there are two
        classes. When set to true, one-hot-encoding will be used. If set to
        false a more efficient shorcut will be performed.

    Attributes
    ----------
    is_fitted_ : `bool`
        Whether the feature mapping has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    len_ : `int`
        Length of the feature mapping vector.

    '''

    def __init__(self, n_classes, fit_intercept=True, one_hot=False):

        self.n_classes = n_classes
        self.fit_intercept = fit_intercept
        self.one_hot = one_hot

    def fit(self, X, Y=None):
        '''
        Performs training stage.

        Learns the required hyperparameters
        for the feature mapping transformation
        from the training instances
        and set the length of the feature mapping (one-hot encoded)
        obtained from the `eval_xy` function.

        .. note:: If a user implements `fit` function in his own feature
              mapping, then it is recommended to call this `fit` function
              at the end of his own function to automatically set
              the length of the feature mapping.
              This `fit` function can be called in a subclass as follows -

                                `super().fit(X,Y)`

              Feature mappings implemented in this library follow this
              same approach.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances
            used to learn the feature configurations

        Y : `array`-like of shape (`n_samples`)
            Labels corresponding to the unlabeled instances.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X = check_array(X, accept_sparse=True)
        self.is_fitted_ = True

        # Defining the length of the phi
        self.len_ = self.transform(X[0:1, :]).shape[1]

        return self

    def transform(self, X):
        '''
        Transform the given instances to the features and return a copy.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances.

        Returns
        -------
        X_feat : `array`-like of shape (`n_samples`, `n_features`)
            Transformed features from the given instances.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ["is_fitted_"])

        if self.fit_intercept:
            X_feat = np.hstack(([[1]] * n, X))
        else:
            X_feat = X.copy()

        return X_feat

    def eval_xy(self, X, Y):
        '''
        Evaluates the one-hot encoded features of the given instances i.e.,
        X, :math:`\phi(x,y)`, x
        :math:`\in` X and y :math:`\in` Y.
        The encodings are calculated,
        corresponding to the given labels, which is used by the learning stage
        for estimating the expectation of :math:`\phi(x,y)`.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances for developing the feature matrix

        Y : `array`-like of shape (`n_samples`)
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        phi : `array`-like of shape
              (`n_samples`, `n_features` * `n_classes`)
            Matrix containing the one-hot encoding
            with respect to the labels given for all the instances.

        '''

        # Get the features
        X_feat = self.transform(X)
        X_feat, Y = check_X_y(X_feat, Y, accept_sparse=True)
        n = X_feat.shape[0]

        # Efficient configuration in case of binary classification.
        if self.n_classes == 2 and not self.one_hot:
            X_feat[Y == 1, :] = X_feat[Y == 1, :] * -1
            phi = X_feat

        # One-hot encoding for multi-class classification.
        else:
            phi = np.zeros((n, self.n_classes * X_feat.shape[1]))
            tweaked_eye_mat = (np.eye(self.n_classes))[Y, :]
            for i in range(n):
                phi[i, :] = np.kron(tweaked_eye_mat[i], X_feat[i, :])

        return phi

    def eval_x(self, X):
        '''
        Evaluates the one-hot encoded features of the given instances i.e.,
        X, :math:`\phi(x,y)`, x
        :math:`\in` X and all the labels. The output is 3D matrix
        that is composed of 2D matrices corresponding to each of the instance.
        These 2D matrices are the one-hot encodings of the instances' features
        corresponding to all the possible labels in the data.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances for developing the feature matrix.

        Returns
        -------
        phi : `array`-like of shape
                    (`n_samples`, `n_classes`, `n_features` * `n_classes`)
            Matrix containing the one-hot encoding for all the classes
            for each of the instances given.

        '''

        n = X.shape[0]

        # Compute the one-hot encodings
        phi = self.eval_xy(np.repeat(X, self.n_classes, axis=0),
                           np.tile(np.arange(self.n_classes), n))

        m = phi.shape[1]
        # Reshape to 3D matrix for easy multiplication in the MRC optimization
        phi = np.reshape(phi, (n, self.n_classes, m))

        return phi

    def est_exp(self, X_transform, Y):
        '''
        Computes the average value of :math:`\phi(x,y)` to estimate :math:`{\tau}`
        that defines the constraint of the uncertainty set of distribution.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_features`)
            Features corresponding with the training instances 
            :math:`\psi(x)`.

        Y : `array`-like of shape (`n_samples`,)
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        tau_ : `array`-like of shape (`n_classes`, `n_features`) for multiple classes and
                 (`1`, `n_features`) for two classes.
            Empirical mean of :math:`\phi(x,y)`.

        '''

        X_transform, Y = check_X_y(X_transform, Y, accept_sparse=True)        

        # Feature mapping length
        feat_len = X_transform.shape[1]
        # Number of samples
        n = X_transform.shape[0]

        # Compute tau efficiently without creating one-hot encoded phi matrix
        if self.n_classes > 2 or self.one_hot is True:
            tau_mat = np.zeros((n_classes, feat_len))
            for y_i in range(self.n_classes):
                tau_mat[y_i, :] = np.sum(X_transform[Y == y_i, :], axis=0) * (1 / n)

        # This corresponds with binary classification
        elif self.one_hot is False:
            tau_mat = np.zeros((1, feat_len))
            tau_mat[0, :] = (np.sum(X_transform[Y == 0, :], axis=0) + \
                             np.sum(-1 * X_transform[Y == 1, :], axis=0)) / X_transform.shape[0]

        return tau_mat

    def est_std(self, X_transform, Y, tau_mat):
        '''
        Standard deviation of :math:`\phi(x,y)` 
        that accounts for inaccuracies in the mean estimate :math:`{\tau}`.
        It is used to estimate :math:`\lambda`
        defining the uncertainty set constraints.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_features`)
            Features corresponding with the training instances 
            :math:`\psi(x)`.

        Y : `array`-like of shape (`n_samples`,)
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        lambda_ : `array`-like of shape (`n_classes`, `n_features`) for multiple classes and
                 (`1`, `n_features`) for two classes.
            Standard deviation of :math:`\phi(x,y)`.

        '''

        X_transform, Y = check_X_y(X_transform, Y, accept_sparse=True)

        # Feature mapping length
        feat_len = X_transform.shape[1]
        # Number of samples
        n = X_transform.shape[0]

        # Compute the standard deviation of features corresponding with the mean tau_mat
        if self.n_classes > 2 or self.one_hot is True:
            lambda_mat = np.zeros((n_classes, feat_len))
            for y_i in range(self.n_classes):
                not_y_i = np.sum(Y != y_i)
                lambda_mat[y_i, :] = np.sqrt((np.sum(np.square(X_transform[Y == y_i, :] - tau_mat[y_i, :]), axis=0) + \
                                                   (np.square(tau_mat[y_i, :]) * not_y_i)) / n)

        # This corresponds with binary classification.
        elif self.one_hot is False:
            lambda_mat = np.zeros((1, feat_len))
            lambda_mat[0, :] = np.sqrt((np.sum(np.square(X_transform[Y == 0, :] - tau_mat[0, :]), axis = 0) + \
                                        np.sum(np.square((-1 * X_transform[Y == 1, :]) - tau_mat[0, :]), axis = 0)) / n)

        return lambda_mat
