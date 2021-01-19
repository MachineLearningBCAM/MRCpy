# Import the MRC super class
from _MRC_ import _MRC_

import numpy as np
import cvxpy as cvx
from sklearn.base import BaseEstimator, ClassifierMixin

class MRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the default constraints and 
    implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.
    """

    def _minimaxRisk(self, X, Y):
        """
        Solves the minimax risk problem for different types of loss (0-1 and log loss)

        Parameters 
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Y : array-like of shape (n_samples,)
            Target labels.

        """

        # Constants
        n, d= X.shape
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
        
        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            if self.linConstr:
                #Exponential number in num_class of linear constraints
                M = self.phi.getLearnConstr(self.linConstr)
                #F is the sum of phi for different subset of Y for each data point in case of linear constr 
                F = M[:, :m]
                cardS= M[:, -1]
                numConstr= M.shape[0]
                constraints.extend([F[i, :]@mu + cardS[i]*nu + cardS[i]*1 <= 1 for i in range(numConstr)])
            else:
                #Constant number in num_class of non-linear constraints
                F = self.phi.getLearnConstr(self.linConstr)
                numConstr = F.shape[0]
                constraints.extend([cvx.sum(cvx.pos((np.ones(self.r) + F[i, :, :]@mu + np.ones(self.r) * nu))) <= 1 for i in range(numConstr)])

        elif self.loss == 'log':
            # Constraints in case of log loss function

            F = self.phi.getLowerConstr()
            numConstr = F.shape[0]
            constraints.extend([cvx.log_sum_exp(F[i, :, :]@mu + np.ones(self.r) * nu) <= 0 for i in range(numConstr)])




        self.mu, self.zhi, self.nu = self.trySolvers(objective, constraints, mu, zhi, nu)

        # Upper bound
        self.upper= (1/2)*(self.b - self.a).T@zhi.value - (1/2)*(self.b + self.a).T@mu.value - nu.value

    def getLowerBound(self):
        '''
        Obtains the lower bound of the fitted classifier: unbounded...

        
        Parameters 
        ----------
        Self : Fitted classifier

        Return
        ------
        lower : float
            The lower bound of the error for the fitted classifier.
        '''

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

        F = self.phi.getLowerConstr()
        numConstr= F.shape[0]

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # epsilon
            eps = np.clip(1 + F@self.mu + self.nu, 0, None)
            c= np.sum(eps, axis=1)
            zeros= np.isclose(c, 0)
            c[zeros]= 1
            eps[zeros, :]= 1/self.r
            c= np.tile(c, (self.r, 1)).transpose()
            eps/= c
            eps = 1 - eps

            constraints.extend(
                [F[j, y, :]@low_mu + low_nu <= eps[j, y]
                    for j in range(numConstr) for y in range(self.r)])

        elif self.loss == 'log':
            # Constraints in case of log loss function

            # epsilon
            eps = F@self.mu
            eps = np.tile(scp.logsumexp(eps, axis=1), (self.r, 1)).transpose() - eps

            constraints.extend(
                [F[i, :, :]@low_mu + low_nu <= eps[i, :] \
                    for i in range(numConstr)])

        self.mu_l, self.zhi_l, self.nu_l = self.trySolvers(objective, constraints, low_mu, low_zhi, low_nu)

        # Get the lower bound
        self.lower= (1/2)*(self.b + self.a).T@self.mu_l - (1/2)*(self.b - self.a).T@self.zhi_l + self.nu_l

        return self.lower

    def predict_proba(self, X):
        '''
        Conditional probabilities corresponding to each class for each unlabeled instance

        @param X: the unlabeled instances, np.array(double) n_instances X dimensions
        @return: p(Y|X), np.array(float) n_instances X n_classes
        '''

        # n_instances X n_classes X phi.len
        Phi = self.phi.eval(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Unnormalized conditional probabilityes
            hy_x = np.clip(1 + np.dot(Phi, self.mu) + self.nu, 0., None)


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

            v = np.dot(Phi, self.mu)

            # Unnormalized conditional probabilityes
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:,i], (self.r, 1)).transpose()), axis=1) \
                        for i in range(self.r)).transpose()
            hy_x = np.reciprocal(hy_x)

        return hy_x
