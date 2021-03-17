"""
An example of creating you own custom feature mappings

In this example, I am extending the Phi parent class
according to the needs of the mappings.
You can choose the best feature mapping class for extension
according to your requirements.
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils import check_array

from MRCpy import CMRC
from MRCpy.phi import *

# Custom phi example: Generating the linear kernel
# modified by multiplying a constant


class myPhi(Phi):

    """
    This constructor is by default present in the parent Phi class.
    So, no need to redefine this constructor
    unless you need any extra parameters from the user.
    In our example here, we don't actually need this
    as we are not using any extra parameters here
    but it is defined here as an example.
    Removing this constructor doesn't have any affect on the performance.
    """
    def __init__(self, n_classes):
        # Calling the parent constructor.
        # It is always better convention to call the parent constructor
        # for primary variables initialization.
        super().__init__(n_classes)

        # Define any extra parameters for your own features
        # Example : self.add_intercept = True/False

    def fit(self, X, Y=None):
        """
        Fit any extra parameter for your feature mappings
        and set the length of the feature mapping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances
            used to learn the feature configurations

        Y : array-like of shape (n_samples,), default=None
            Labels corresponding to the unlabeled instances.
        """

        # Check if the array is 2D numpy matrix or not.
        # X is expected to be a numpy 2D matrix.
        X = check_array(X, accept_sparse=True)

        d = X.shape[1]

        # Defining the length of the phi

        # Here we define the total length of the phi feature vector
        # given by the class variable self.len_
        # In this case, the kernel is linear,
        # so the length of the kernel is d (Number of dimension of X)
        # added by 1 (for the intercept).
        # We one-hot encoded feature mapping
        # so we multiply by number of classes for total length
        self.len_ = (d + 1) * self.n_classes

        # A class variable. You need set it true
        # after you fit your feature mapping functions
        self.is_fitted_ = True

        # Return the fitted feature mapping instance
        return self

    def transform(self, X):

        """
        Transform the given instances to the principal features if any.
        No need to give definition for this function
        if you are not calling it in the eval function.

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

        # We want to use the linear kernel feature mapping (i.e., X itself)
        # and transform it by multiplying by a factor 2
        # Note: This is just an example of building custom feature mappings,
        #       so the results after using this feature mappings
        #       might not be satisfactory
        X_feat = X * 2

        # Return the features
        return X_feat

    def eval(self, X, Y=None):

        """
        Computes the complete feature mapping vector
        corresponding to instance X.
        X can be a matrix in which case
        the function returns a matrix in which
        the rows represent the complete feature mapping vector
        corresponding to each instance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances for developing the feature matrix

        Y : array-like of shape (n_samples,), default=None
            Labels corresponding to the unlabeled training instances

        Returns
        -------
        phi : array-like of shape (n_samples, n_classes, n_features*n_classes)
            Matrix containing the complete feature vector as rows
            corresponding to each of the instance.
            In case of one-hot encoding, the feature mappings are given by
            calling the transform function to get the principal features and
            then appending zeros for the one-hot encoding.
            In case Y is given, the encoding for each instance are calculated
            corresponding to those labels.
        """
        # Here in this example,
        # we want to use the one-hot encoded feature mappings.
        # So, we call the parent class eval function
        # which does the one-hot encoding by default
        # and also adds the intercept corresponding to each class
        return super().eval(X, Y)

        # In case you don't want the one-hot encoding,
        # you have to define you own eval function
        # without calling the parent class eval function.


if __name__ == '__main__':

    # Loading the dataset
    X, Y = load_iris(return_X_y=True)
    r = len(np.unique(Y))

    # Creating the custom phi object
    myphi = myPhi(n_classes=r)

    # Fit the MRC model with the custom phi
    clf = CMRC(n_classes=r, phi=myphi).fit(X, Y)

    # Prediction
    print('\n\nThe predicted values for the first 3 instances are : ')
    print(clf.predict(X[:3, :]))

    # Predicted probabilities
    print('\n\nThe predicted probabilities for the first 3 instances are : ')
    print(clf.predict_proba(X[:3, :]))

    # Accuracy/Score of the model
    print('\n\nThe score is : ')
    print(clf.score(X, Y))
