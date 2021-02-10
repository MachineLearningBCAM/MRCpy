# Import the MRC super class
from minimax_risk_classifiers._MRC_ import _MRC_

import numpy as np
import cvxpy as cvx
import itertools as it
import scipy.special as scs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

class CMRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the additional marginals constraints on the instances.
    It also implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.
    """

    def _minimaxRisk(self, X):
        """
        Solves the marginally constrained minimax risk problem 
        for different types of loss (0-1 and log loss).

        Parameters 
        ----------
        X : array-like of shape (n_samples1, n_dimensions)
            Training instances used in the optimization.

        """

        # Get the learn configurations of phi (feature mapping)
        phi = self.phi.learnConfig(X, self.learn_duplicates)

        # Constants
        n= phi.shape[0]
        m= self.phi.len

        # Variables
        mu = cvx.Variable(m)
        zhi = cvx.Variable(m)

        # Objective function
        objective = cvx.Minimize((1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # # We compute the learn constraints without omitting the duplicate elements
            M= self.phi.getAllSubsetConfig(phi)
            # F is the sum of phi for different subset of Y for each data point 
            F = M[:, :m]
            cardS= M[:, -1]    

            # Number of classes in each set
            cardS= np.arange(1, self.n_classes+1).repeat([n*scs.comb(self.n_classes, numVals)
                                            for numVals in np.arange(1, self.n_classes+1)])

            # Calculate the psi function and add it to the objective function
            # First we calculate the all possible values of psi for all the points
            psi_arr = (np.ones(cardS.shape[0])-(F@mu + cardS))/cardS
            for i in range(n):
                # Get psi for each data point and add the min value to objective
                psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
                objective = objective + cvx.Minimize((-1/n)*cvx.min((psi_arr_xi)))

        elif self.loss == 'log':
            # Constraints in case of log loss function

            for i in range(n):
                objective = objective + \
                            cvx.Minimize((1/n)*cvx.log_sum_exp(phi[i,:,:]@mu))

        # Constraints
        constraints= [zhi + mu >= 0, zhi - mu >= 0]

        self.mu, self.zhi = self.trySolvers(objective, constraints, mu, zhi)

    def predict_proba(self, X):
        """
        Conditional probabilities corresponding to each class for each unlabeled instance

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            Testing instances for which 
            the prediction probabilities are calculated for each class.
        
        Returns
        -------
        hy_x : ndarray of shape (n_samples, n_classes)
            The probabilities (p(y|x)) corresponding to the predictions
            for each class.

        """
        X = check_array(X, accept_sparse=True)
        n = X.shape[0]

        # n_instances X n_classes X phi.len
        phi= self.phi.eval(X)
        m = self.phi.len

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            M= self.phi.getAllSubsetConfig(phi)
            # # F is the sum of phi for different subset of Y for each data point 
            F = M[:, :m]
            cardS= M[:, -1]

            # Compute psi
            psi = np.zeros(n)

            # First we calculate the all possible values of psi for all the points
            psi_arr = (np.ones(cardS.shape[0])-(F@self.mu + cardS))/cardS

            for i in range(n):
                # Get psi values for each data point and find the min value
                psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
                psi[i] = np.min(psi_arr_xi)

            # Conditional probabilities
            hy_x = np.clip(np.ones((n,self.n_classes)) + np.dot(phi, self.mu) + \
                np.tile(psi, (self.n_classes,1)).transpose(), 0., None)


            # normalization constraint
            c = np.sum(hy_x, axis=1)
            # check when the sum is zero
            zeros = np.isclose(c, 0)
            c[zeros] = 1
            hy_x[zeros, :] = 1 / self.n_classes
            c = np.tile(c, (self.n_classes, 1)).transpose()
            hy_x = hy_x / c

        elif self.loss == 'log':
            # Constraints in case of log loss function

            v = np.dot(phi, self.mu)

            # Unnormalized conditional probabilities
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:,i], (self.n_classes, 1)).transpose()), axis=1) \
                        for i in range(self.n_classes)).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x

    def setLearnConfigType(self):
        """
        Learn the duplicate configuration to be used in the objective function 
        in case of this constrained MRC. Duplicate configuration are observed 
        when the dataset contains duplication entries.
        """
        self.learn_duplicates = True
