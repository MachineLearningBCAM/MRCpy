# Import the feature mapping base class
from minimax_risk_classifiers.phi.phi import Phi

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
import numpy as np

class PhiThreshold(Phi):
    """
    Phi (feature function) composed by products of (univariate) threshold features.
    A threshold feature is a funtion, f(x;t,d)=1 when x_d<t and 0 otherwise.
    A product of threshold features is an indicator of a region 
    and its expectancy is closely related to cumulative distributions.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset

    n_thresholds : int, default=200
        It defines the maximum number of allowed threshold values for each dimension.

    Attributes
    ----------
    self.thrsVal : array-like of shape (n_thresholds)
        Array of all threshold values learned from the training data.

    self.thrsDim : array-like of shape (n_thresholds)
        Array of dimensions corresponding to the learned threshold value in self.thrsVal.

    is_fitted_ : bool
        True if the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    """

    def __init__(self, n_classes, n_thresholds=200):

        # Call the base class init function.
        super().__init__(n_classes=n_classes)
        self.n_thresholds = n_thresholds

    def fit(self, X, Y=None):
        """
        Learn the set of features using the thresholds obtained from the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances used to learn the feature configurations.

        Y : array-like of shape (n_samples,), default=None
            Labels corresponding to the unlabeled instances X,
            used for finding the thresholds from the dataset.

        Returns
        -------
        self : 
            Fitted estimator

        """

        X, Y = check_X_y(X, Y, accept_sparse=True)

        # Obtain the thresholds from the instances-label pairs.
        self.thrsDim_, self.thrsVal_ = self.d_tree_split(X, Y, self.n_thresholds)

        # Defining the length of the phi
        self.m = len(self.thrsDim_) + 1
        self.len = self.m*self.n_classes
        self.is_fitted_ = True

        return self

    def d_tree_split(self, X, Y, n_thresholds=None):
        """
        Learn the univariate thresholds 
        by using the split points of decision trees for each dimension of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled instances.

        Y : array-like of shape (n_samples,)
            Labels corresponding to the instances.

        n_thresholds : int, default = None
            Maximum limit on the number of thresholds obtained

        Returns
        -------
        prodThrsDim : array-like of shape (n_thresholds)
            The dimension in which the thresholds are defined.

        prodThrsVal : array-like of shape (n_thresholds)
            The threshold value in the corresponding dimension.

        """

        (n, d) = X.shape

        prodThrsVal = []
        prodThrsDim = []

        # One order thresholds: all the univariate thresholds
        for dim in range(d):
            if n_thresholds== None:
                dt = DecisionTreeClassifier()
            else:
                dt= DecisionTreeClassifier(max_leaf_nodes=n_thresholds+1)

            dt.fit(np.reshape(X[:,dim],(n,1)),Y)

            dimThrsVal= np.sort(dt.tree_.threshold[dt.tree_.threshold!= -2])

            for t in dimThrsVal:
                prodThrsVal.append([t])
                prodThrsDim.append([dim])

        return prodThrsDim, prodThrsVal

    def transform(self, X):
        """
        Compute the random fourier features from the given instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances.

        Returns
        -------
        X_feat : array-like of shape (n_samples, n_features)
            Transformed features from the given instances i.e., 
            the instances itself.

        """

        n = X.shape[0]

        check_is_fitted(self, ["thrsDim_", "thrsVal_"])
        X = check_array(X, accept_sparse=True)
        # Store the features based on the thresholds obtained
        X_feat = np.zeros((n, len(self.thrsDim_)), dtype=int)

        # Calculate the threshold features
        for thrsInd in range(len(self.thrsDim_)):
            X_feat[:, thrsInd] = np.all(X[:, self.thrsDim_[thrsInd]] <= self.thrsVal_[thrsInd],
                                axis=1).astype(np.int)

        return X_feat

    