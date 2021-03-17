"""
    Super class for the feature mapping functions
"""

from sklearn.utils import check_X_y, check_array
import numpy as np
import itertools as it
import warnings
import scipy.special as scs


class Phi():

    """
    Phi (Feature mapping function)

    The class implements some feature functions
    that can be used by the MRC to map the input instances to these features.
    It also provides the user to define his own custom feature functions.

    Implementation for following types of feature functions are provided -

    1) Linear kernel - The features are the instances itself i.e.,
        there is no mapping function except
        the one-hot encoding of the instances.

    2) Thresholding kernel - The end features are defined as binary numbers
        which are obtained by comparing
        with the threshold values in each dimension.
        These threshold values are obtained from the training dataset.
        In this case, the number of features depend upon
        the number of thresholds obtained
        and each of the threshold is defined for one of the dimensions.

    3) Gaussian kernel - The features are defined using the gaussian kernel.
        The gaussian kernel is defined as -
                        K(x1,x2) = exp( -1 * gamma * |x1-x2|^2 )

    4) Custom kernel - These feature functions are defined by the user.
        The user needs to define a function which takes instances as input
        and returns the mapped features for those instances.
        This class then uses that function
        to map the instances to the features
        and develop a one-hot encoding of those features.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset

    Attributes
    ----------
    is_fitted_ : bool
        True if the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    len_ : int
        Defines the length of the feature mapping vector.

    Note:
    -----
    This is a base class for all the feature mappings.
    To create a new feature mapping,
    it is expected to extend this class and
    then define the functions transform and
    fit if it is required to learn any hyper parameters
    for the feature mappings

    """

    def __init__(self, n_classes):

        self.n_classes = n_classes

    def fit(self, X, Y=None):
        """
        Learn the hyperparameters
        required for the feature mapping function from the training instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances
            used to learn the feature configurations

        Y : array-like of shape (n_samples)
            Labels corresponding to the unlabeled instances.

        Returns
        -------
        self :
            Fitted estimator

        """

        self.is_fitted_ = True
        return self

    def learnConfig(self, X, duplicate=False):
        """
        Learn all the configurations of x in X for every value of y.
        Duplicate configurations are obtained
        if the duplicate parameter is True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances
            used to learn the feature configurations.

        duplicate : bool, default = False
            Learn the configurations of phi with duplicate instances if exists
            if the value is true.

        Returns
        -------
        F : array-like of shape (n_samples, n_classes,
                            length of features (n_features * n_classes))
            The learned configurations from the given instances.

        """

        phi = self.eval(X)

        # Used in the definition of the constraints/objective of the MRC
        # F is a tuple of floats with dimension n_intances X n_classes X m
        if duplicate:
            # Used in case of CMRC
            F = phi
        else:
            # Used in case of MRC
            # Disctinct configurations for phi_x,y for x in X and y=1,...,r.
            F = np.unique(phi, axis=0)

        return F

    def transform(self, X):
        """
        Transform the given instances to the features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances.

        Returns
        -------
        X_feat : array-like of shape (n_samples, n_features)
            Transformed features from the given instances.

        """

        return X

    def eval(self, X, Y=None):
        """
        Evaluate the one-hot encoded features of the given instances i.e.,
        X, phi(x,y) for all x in X and y=0,...,r-1. In case labels are given,
        the encodings are calculated corresponding to those labels
        and this form of eval function is used in the learning stage
        for estimating the expected values of phi i.e., tau and lambda

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances for developing the feature matrix

        Y : array-like of shape (n_samples,), default=None
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        phi : array-like of shape
              (n_samples, n_classes, n_features * n_classes) or
              (n_samples, n_features * n_classes) if Y is given
            Matrix containing the one-hot encoded features
            of all possible classes of each instance.
            In case Y is given, the encoding for each instance are calculated
            corresponding to those labels.

        """

        n = X.shape[0]

        # Get the features
        X_feat = self.transform(X)

        X_feat = check_array(X_feat, accept_sparse=True)

        # Number of features + 1 (for the intercept being added for each class)
        m = X_feat.shape[1] + 1

        if Y is None:

            # For binary case,
            # the problem can be solved using a smaller of configurations.
            # Without one-hot encoding.
            if self.n_classes == 2:
                phi = np.repeat(np.hstack((np.ones((X_feat.shape[0], 1)),
                                           X_feat))
                                [:, np.newaxis, :], 2, axis=1)
                phi[:, 1, :] *= -1

            # One-hot encoding for multi-class classification.
            else:
                phi = np.zeros((n, self.n_classes, self.len_), dtype=float)

                # adding the intercept
                phi[:, np.arange(self.n_classes),
                    np.arange(self.n_classes) * m] = \
                    np.tile(np.ones(n), (self.n_classes, 1)).transpose()

                # Compute the phi function
                for dimInd in range(1, m):
                    phi[:, np.arange(self.n_classes),
                        np.arange(self.n_classes) * m + dimInd] = \
                        np.tile(X_feat[:, dimInd - 1],
                                (self.n_classes, 1)).transpose()

        else:

            # Efficient configuration in case of binary classification.
            if self.n_classes == 2:
                phi = np.hstack((np.ones((X_feat.shape[0], 1)), X_feat))
                phi[Y == 1, :] *= -1

            # One-hot encoding for multi-class classification.
            else:
                phi = np.zeros((n, self.len_), dtype=float)

                # adding the intercept
                phi[np.arange(n), Y * m] = np.ones(n)

                # Compute the phi function
                for dimInd in range(1, m):
                    phi[np.arange(n), dimInd + Y * m] = X_feat[:, dimInd - 1]

        return phi

    def estExp(self, X, Y):
        """
        Average value of phi in the supervised dataset (X,Y)
        Used in the learning stage as an estimate
        of the expected value of phi, tau

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances.

        Y : array-like of shape (n_samples,)
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        tau : array-like of shape (n_features * n_classes)
            Average value of phi

        """

        X, Y = check_X_y(X, Y, accept_sparse=True)
        return np.average(self.eval(X, Y), axis=0)

    def estStd(self, X, Y):
        """
        Standard deviation of phi in the supervised dataset (X,Y)
        Used in the learning stage
        to estimate the bounds of the expected value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances.

        Y : array-like of shape (n_samples,)
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        lambda : array-like of shape (n_features * n_classes)
            Standard deviation of phi

        """

        X, Y = check_X_y(X, Y, accept_sparse=True)
        return np.std(self.eval(X, Y), axis=0)

    def getAllSubsetConfig(self, F):
        """
        Calculate the sum of the feature configuration vectors
        for all possible subsets
        that can be obtained from n_classes for each of the instances i.e.,
        sum{ phi(x,yi) } for yi in C subset of Y and for x in X

        Parameters
        ----------
        F : array-like of shape (n_samples, n_classes, n_features * n_classes)
            One-hot encoded feature matrix
            for each instance and each class in n_classes.

        Returns
        -------
        M : array-like of shape (n_samples, n_features * n_classes)
            Matrix containing the subset sum of feature configures
            for each instances
            and for all possible of n_classes.
            The last column of the matrix defines
            the number of classes in that subset
        """

        n = F.shape[0]

        # Supress the depreciation warnings
        warnings.simplefilter('ignore')

        # Summing up the phi configurations
        # for all possible subsets of classes for each instance
        avgF = np.vstack((np.sum(F[:, S, ], axis=1)
                          for numVals in range(1, self.n_classes + 1)
                          for S in it.combinations(np.arange(self.n_classes),
                                                   numVals)))

        # Compute the corresponding length of the subset of classes
        # for which sums computed for each instance
        cardS = np.arange(1, self.n_classes + 1).\
            repeat([n * scs.comb(self.n_classes, numVals)
                    for numVals in np.arange(1,
                    self.n_classes + 1)])[:, np.newaxis]

        M = np.hstack((avgF, cardS))

        return M
