'''Feature mappings obtained using a threshold (Half planes).'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping base class
from MRCpy.phi import BasePhi


class ThresholdPhi(BasePhi):
    '''
    Threshold features

    A threshold feature is a function, :math:`f(x_d,t)=1`
    when :math:`x_d<t` and 0 otherwise,
    for a given x in dimension d and threshold t in that dimension.
    A product of threshold features is an indicator of a region
    and its expectation is closely related to cumulative distributions.
    This class obtains the thresholds fitting multiple one-dimensional
    decision stumps on the training data.

    .. seealso:: For more information about MRC, one can refer to the
    following resources:

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
        (i.e. data is expected to be already centered).

    one_hot : `bool`, default = `False`
        Controls the method used for evaluating the features of the
        given instances in the binary case.
        Only applies in the binary case, namely, only when there are two
        classes. If set to true, one-hot-encoding will be used. If set to
        false a more efficient shorcut will be performed.

    n_thresholds : `int`, default = `200`
        Maximum number of allowed threshold values
        for each dimension.

    Attributes
    ----------
    self.thrsVal : `array`-like of shape (`n_thresholds`)
        Threshold values learned from the training data.

    self.thrsDim : `array`-like of shape (`n_thresholds`)
        Dimensions
        corresponding to the learned threshold value in `self.thrsVal`.

    is_fitted_ : `bool`
        Whether the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    len_ : `int`
        Length of the feature mapping vector.

    '''

    def __init__(self, n_classes, fit_intercept=True,
                 n_thresholds=200, one_hot=False):

        # Call the base class init function.
        super().__init__(n_classes=n_classes, fit_intercept=fit_intercept,
                         one_hot=one_hot)

        self.n_thresholds = n_thresholds

    def fit(self, X, Y=None):
        '''
        Learns the set of thresholds using one-dimensional decision stumps
        obtained from the dataset.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances
            used to learn the feature configurations.

        Y : `array`-like of shape (`n_samples`,), default = `None`
            Labels corresponding to the unlabeled instances X,
            used for finding the thresholds from the dataset.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X, Y = check_X_y(X, Y, accept_sparse=True)

        # Obtain the thresholds from the instances-label pairs.
        self.thrsDim_, self.thrsVal_ = self.d_tree_split(X, Y,
                                                         self.n_thresholds)

        # Sets the length of the feature mapping
        super().fit(X, Y)

        return self

    def d_tree_split(self, X, Y, n_thresholds=None):
        '''
        Learn the univariate thresholds
        by using the split points of decision trees
        for each dimension of data.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled instances.

        Y : `array`-like of shape (`n_samples`,)
            Labels corresponding to the instances.

        n_thresholds : `int`, default = `None`
            Maximum limit on the number of thresholds obtained

        Returns
        -------
        prodThrsDim : `array`-like of shape (`n_thresholds`)
            Dimension in which the thresholds are defined.

        prodThrsVal : `array`-like of shape (`n_thresholds`)
            Threshold value in the corresponding dimension.

        '''

        (n, d) = X.shape

        prodThrsVal = []
        prodThrsDim = []

        # One order thresholds: all the univariate thresholds
        for dim in range(d):
            if n_thresholds is None:
                dt = DecisionTreeClassifier()
            else:
                dt = DecisionTreeClassifier(max_leaf_nodes=n_thresholds + 1)

            dt.fit(np.reshape(X[:, dim], (n, 1)), Y)

            dimThrsVal = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2])

            for t in dimThrsVal:
                prodThrsVal.append([t])
                prodThrsDim.append([dim])

        return prodThrsDim, prodThrsVal

    def transform(self, X):
        '''
        Compute the threshold features (0/1)
        by comparing with the thresholds obtained in each dimension.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances.

        Returns
        -------
        X_feat : `array`-like of shape (`n_samples`, `n_features`)
            Transformed features from the given instances.

        '''

        n = X.shape[0]

        check_is_fitted(self, ["thrsDim_", "thrsVal_", "is_fitted_"])
        X = check_array(X, accept_sparse=True)

        # Store the features based on the thresholds obtained
        X_feat = np.zeros((n, len(self.thrsDim_)), dtype=int)

        # Calculate the threshold features
        for thrsInd in range(len(self.thrsDim_)):
            X_feat[:, thrsInd] = \
                np.all(X[:, self.thrsDim_[thrsInd]] <= self.thrsVal_[thrsInd],
                       axis=1).astype(int)

        return X_feat
