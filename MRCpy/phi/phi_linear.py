"""Linear Kernel."""

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping base class
from MRCpy.phi.phi import Phi


class PhiLinear(Phi):
    """
    Phi (feature function) obtained using the linear kernel i.e.,
    the features are the instances itself with some intercept added.

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

    """

    def fit(self, X, Y=None):
        """
        Learn the set of Phi features from the dataset by one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances
            used to learn the feature configurations

        Y : array-like of shape (n_samples,), default=None
            This argument will never be used in this case.
            It is present in the signature for consistency
            in the signature of the function among different feature mappings.

        Returns
        -------
        self :
            Fitted estimator

        """

        X = check_array(X, accept_sparse=True)

        d = X.shape[1]

        # Defining the length of the phi
        self.len_ = (d + 1)
        # For one-hot encoding in case of multi-class classification.
        if self.n_classes != 2:
            self.len_ *= self.n_classes

        self.is_fitted_ = True

        return self

    def transform(self, X):
        """
        Transform the given instances to the features.
        The features in case of a linear kernel are the instances itself.

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

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, ["len_", "is_fitted_"])
        X_feat = X

        return X_feat
