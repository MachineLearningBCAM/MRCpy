# Import the feature mapping base class
from minimax_risk_classifiers.phi.phi import Phi

import numpy as np
import statistics
import random
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_random_state

class PhiGaussian(Phi):
    """
    Phi (feature function) obtained by approximating the rbf kernel by 
    Random Fourier Feature map.

    Parameters
    ----------
    n_classes : int
        The number of classes in the dataset.

    gamma : {'scale', 'avg_ann', 'avg_ann_50', float} default = 'avg_ann_50'
        It defines the type of heuristic to be used 
        to calculate the scaling parameter for the gaussian kernel.

    n_components : int, default=300
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    Attributes
    ----------
    random_weights_ : array-like of shape (n_features, n_components/2)
        The sampled basis.

    is_fitted_ : bool
        True if the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    References
    ----------
    [1] Random Features for Large-Scale Kernel Machines.
    Ali Rahimi and Ben Recht.
    In NIPS 2007.
    (https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

    """

    def __init__(self, n_classes, gamma='avg_ann_50', n_components=300):

        # Call the base class init function.
        super().__init__(n_classes=n_classes)

        self.gamma = gamma
        self.n_components = n_components

    def fit(self, X, Y=None):
        """
        Learn the set of features for the given type using the given instances.
        Also, compute the value of gamma hyperparameter if the value is not given.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled training instances used to learn the feature configurations.

        Y : array-like of shape (n_samples,), default=None
            This argument will never be used in this case. 
            It is present in the signature for consistency 
            in the signature among different feature mappings.

        Returns
        -------
        self : 
            Fitted estimator

        """

        X = check_array(X, accept_sparse=True)

        d= X.shape[1]
        # Evaluate the gamma according to the gamma type given in self.gamma
        if self.gamma == 'scale':
            self.gamma_val = 1 / (d*X.var())

        elif self.gamma == 'avg_ann_50':
            self.gamma_val = self.rff_gamma(X)

        elif type(self.gamma) != str:
            self.gamma_val = self.gamma

        else:
            raise ValueError('Unexpected value for gamma ...')

        # Approximating the gaussian kernel using random features 
        # that are obtained from a normal distribution.
        random_state = check_random_state(None)
        self.random_weights_ = random_state.normal(0, np.sqrt(2*self.gamma_val), \
                                            size=(d, int(self.n_components/2)))

        # Defining the length of the phi
        self.m = self.n_components+1
        self.len = self.m*self.n_classes
        self.is_fitted_ = True

        return self

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

        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)

        X_trans = X@self.random_weights_
        X_feat = (1/np.sqrt(int(self.n_components/2))) * \
                        np.hstack((np.cos(X_trans),np.sin(X_trans)))

        return X_feat

    def heuristic_gamma(self, X, Y):

        """
        Compute the scale parameter for gaussian kernels using the heuristic - 

                sigma = median{ {min ||x_i-x_j|| for j|y_j != +1 } for i|y_i != -1 } 

        for two classes {+1, -1}. For multi-class, we use the same strategy 
        by finding median of min norm value for each class against all other classes 
        and then taking the average value of the median of all the classes.

        The heuristic calculates the value of sigma for gaussian kernels. 
        So to find the gamma value for the kernel defined in our implementation, 
        the following formula is used - 
                gamma = 1/ (2 * sigma^2)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled instances.

        Y : array-like of shape (n_samples,)
            Labels corresponding to the instances.

        Returns
        -------
        gamma : float value
            Gamma value computed using the heuristic from the instances.

        """

        # List to store the median of min norm value for each class
        dist_x_i = list()

        n_classes = np.max(Y) + 1

        for i in range(n_classes):
            x_i = X[Y == i,:]
            x_not_i = X[Y != i,:]

            # Find the distance of each point of this class 
            # with every other point of other class
            norm_vec = np.linalg.norm(np.tile(x_not_i,(x_i.shape[0], 1)) - \
                np.repeat(x_i, x_not_i.shape[0], axis=0), axis=1)
            dist_mat = np.reshape(norm_vec, (x_not_i.shape[0], x_i.shape[0]))

            # Find the min distance for each point and take the median distance
            minDist_x_i = np.min(dist_mat, axis=1)
            dist_x_i.append(statistics.median(minDist_x_i))

        sigma = np.average(dist_x_i)

        # Evaluate gamma
        gamma = 1/(2 * sigma * sigma)

        return gamma

    def rff_gamma(self, X):

        """
        Function to find the scale parameter for random fourier features obtained from 
        gaussian kernels using the heuristic given in - 
                    
                "Compact Nonlinear Maps and Circulant Extensions"

        The heuristic to calculate the sigma states that it is a value that is obtained from
        the average distance to the 50th nearest neighbour estimated from 1000 samples of the dataset.

        Gamma value is given by - 
                gamma = 1/ (2 * sigma^2)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Unlabeled instances.

        Returns
        -------
        gamma : float value
            Gamma value computed using the heuristic from the instances.

        References
        ----------
        [1] Compact Nonlinear Maps and Circulant Extensions
        Felix X. Yu, Sanjiv Kumar, Henry Rowley and Shih-Fu Chang
        (https://arxiv.org/pdf/1503.03893.pdf)

        """

        # Number of training samples
        n = X.shape[0]

        neighbour_ind = 50

        # Find the nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=(neighbour_ind+1), algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Compute the average distance to the 50th nearest neighbour
        sigma = np.average(distances[:, neighbour_ind])

        return 1/ (2 * sigma * sigma)

