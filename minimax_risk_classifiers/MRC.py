# Import the MRC super class
from minimax_risk_classifiers._MRC_ import _MRC_

import numpy as np
import cvxpy as cvx
import scipy.special as scs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

class MRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the default constraints and 
    implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.
    """

    def _minimaxRisk(self, X):
        """
        Solves the minimax risk problem 
        for different types of loss (0-1 and log loss).
        The solution of the default MRC optimization gives the upper bound of the error.

        Parameters 
        ----------
        X : array-like of shape (n_samples1, n_dimensions)
            Training instances used in the optimization.

        """

        # Constants
        m= self.phi.len

        # Variables
        mu = cvx.Variable(m)
        zhi = cvx.Variable(m)
        nu = cvx.Variable()

        # Cost function
        cost = (1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu - nu

        # Objective function
        objective = cvx.Minimize(cost)

        # Constraints
        constraints= [zhi + mu >= 0, zhi - mu >= 0]

        # Get the learn configurations of phi (feature mapping)
        phi = self.phi.learnConfig(X, self.learn_duplicates)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Exponential number in num_class of linear constraints
            M = self.phi.getAllSubsetConfig(phi)
            
            # F is the sum of phi
            # for different subset of Y
            # for each data point
            F = M[:, :m]
            cardS= M[:, -1]
            numConstr= M.shape[0]
            constraints.extend([F[i, :]@mu + cardS[i]*nu + cardS[i]*1 <= 1 \
                                    for i in range(numConstr)])

        elif self.loss == 'log':
            # Constraints in case of log loss function

            numConstr = phi.shape[0]
            constraints.extend([cvx.log_sum_exp(phi[i, :, :]@mu + np.ones(self.r) * nu) <= 0 \
                                    for i in range(numConstr)])

        self.mu, self.zhi, self.nu = self.trySolvers(objective, constraints, mu, zhi, nu)

        # Save the phi configurations for finding the lower bounds
        self.lowerPhiConfigs = phi

        # Upper bound
        self.upper= (1/2)*(self.b - self.a).T@self.zhi - (1/2)*(self.b + self.a).T@self.mu - self.nu

    def getLowerBound(self):
        """
        Obtains the lower bound of the fitted classifier: unbounded...

        Returns
        -------
        lower : float value
            The lower bound of the error for the fitted classifier.
        """

        # Variables
        m= self.phi.len
        low_mu = cvx.Variable(m)
        low_zhi = cvx.Variable(m)
        low_nu = cvx.Variable()

        # Cost function
        cost = (1/2)*(self.b + self.a).T@low_mu - (1/2)*(self.b - self.a).T@low_zhi + low_nu

        # Objective function
        objective = cvx.Maximize(cost)

        # Constraints
        constraints= [low_zhi + low_mu >= 0, low_zhi - low_mu >= 0]

        phi = self.lowerPhiConfigs
        numConstr= phi.shape[0]

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # epsilon
            eps = np.clip(1 + phi@self.mu + self.nu, 0, None)
            c= np.sum(eps, axis=1)
            zeros= np.isclose(c, 0)
            c[zeros]= 1
            eps[zeros, :]= 1/self.r
            c= np.tile(c, (self.r, 1)).transpose()
            eps/= c
            eps = 1 - eps

            constraints.extend(
                [phi[j, y, :]@low_mu + low_nu <= eps[j, y]
                    for j in range(numConstr) for y in range(self.r)])

        elif self.loss == 'log':
            # Constraints in case of log loss function

            # epsilon
            eps = phi@self.mu
            eps = np.tile(scs.logsumexp(eps, axis=1), (self.r, 1)).transpose() - eps

            constraints.extend(
                [phi[i, :, :]@low_mu + low_nu <= eps[i, :] \
                    for i in range(numConstr)])

        self.mu_l, self.zhi_l, self.nu_l = self.trySolvers(objective, constraints, low_mu, low_zhi, low_nu)

        # Get the lower bound
        self.lower= (1/2)*(self.b + self.a).T@self.mu_l - (1/2)*(self.b - self.a).T@self.zhi_l + self.nu_l

        return self.lower

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
        # n_instances X n_classes X phi.len
        phi = self.phi.eval(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Unnormalized conditional probabilityes
            hy_x = np.clip(1 + np.dot(phi, self.mu) + self.nu, 0., None)


            # normalization constraint
            c = np.sum(hy_x, axis=1)
            # check when the sum is zero
            zeros = np.isclose(c, 0)
            c[zeros] = 1
            hy_x[zeros, :] = 1 / self.r
            c = np.tile(c, (self.r, 1)).transpose()
            hy_x = hy_x / c

        elif self.loss == 'log':
            # Constraints in case of log loss function

            v = np.dot(phi, self.mu)

            # Unnormalized conditional probabilityes
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:,i], (self.r, 1)).transpose()), axis=1) \
                        for i in range(self.r)).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x

    def setLearnConfigType(self):
        """
        Avoid the duplicate configuration while learning the constraints
        in case of this default MRC. Duplicate configuration are observed 
        when the dataset contains duplication entries which are not of any use 
        and increase the computation time
        """
        self.learn_duplicates = False
