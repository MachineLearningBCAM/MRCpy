# Import the MRC super class
from minimax_risk_classifiers._MRC_ import _MRC_

import random
import numpy as np
import cvxpy as cvx
import itertools as it
import scipy.special as scs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

class CMRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the additional marginals constraints on the instances.
    It also implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.

    Attributes
    ----------
    is_fitted_ : bool
        True if the classifier is fitted i.e., the parameters are learnt.

    tau_ : array-like of shape (n_features) or float
        The mean estimates for the expectations of feature mappings.

    lambda_ : array-like of shape (n_features) or float
        The variance in the mean estimates for the expectations of the feature mappings.

    mu_ : array-like of shape (n_features) or float
        Parameters learnt by the optimization.

    zhi_ : array-like of shape (n_features) or float
        Parameters learnt by the optimization when solved using CVXpy.
        This paramter is not required in the prediction stage of MRC

    params_ : a dictionary
        Stores the optimal points and best value of the function 
        when the warm_start=True.

    """

    # Redefining the init function to reduce the default number for maximum iterations.
    # In case of CMRC, the convergence is observed to be fast 
    # and hence less iterations should be sufficient
    def __init__(self, n_classes, equality=False, s=0.3, \
                deterministic=False, random_state=None, loss='0-1', \
                warm_start=False, use_cvx=False, solver='SCS', \
                max_iters = 2000, phi='gaussian', **phi_kwargs):
        super().__init__(n_classes, equality, deterministic, \
                        random_state, loss, warm_start, \
                        use_cvx, solver, max_iters, \
                        phi, **phi_kwargs)

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
        m= self.phi.len_

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)
            zhi = cvx.Variable(m)

            # Objective function
            objective = cvx.Minimize((1/2)*(self.b - self.a).T@zhi - (1/2)*(self.b + self.a).T@mu)

            if self.loss == '0-1':
                # Constraints in case of 0-1 loss function

                # We compute the learn constraints without omitting the duplicate elements
                M= self.phi.getAllSubsetConfig(phi)
                # F is the sum of phi for different subset of Y for each data point 
                F = M[:, :m]
                cardS= M[:, -1]

                # Calculate the psi function and add it to the objective function
                # First we calculate the all possible values of psi for all the points
                psi = (np.ones(cardS.shape[0])-(F@mu + cardS))/cardS
                for i in range(n):
                    # Get psi for each data point and add the min value to objective
                    psi_xi = psi[np.arange(i, psi.shape[0], n)]
                    objective = objective + cvx.Minimize((-1/n)*cvx.min((psi_xi)))

            elif self.loss == 'log':
                # Constraints in case of log loss function

                for i in range(n):
                    objective = objective + \
                            cvx.Minimize((1/n)*cvx.log_sum_exp(phi[i,:,:]@mu))

            # Constraints
            constraints= [zhi + mu >= 0, zhi - mu >= 0]

            self.mu_, self.zhi_ = self.trySolvers(objective, constraints, mu, zhi)

        elif not self.use_cvx:
            # Use the subgradient approach for the convex optimization of MRC

            if self.loss == '0-1':
                # Define the objective function and the gradient for the 0-1 loss function.
                
                M_ = self.phi.getAllSubsetConfig(phi)
                # M is the sum of phi for different subset of Y for each data point 
                M = M_[:, :m]
                h = M_[:, -1]
                M = M/(h[:, np.newaxis])
                h = 1 - (1/h)

                # Function to calculate the psi subobjective to be added to the objective function.
                def f_(mu):
                    # First we calculate the all possible values of psi for all the points.
                    psi = M@mu + h
                    idx = []

                    for i in range(n):
                        # Get psi for each data point 
                        # and return the max value over all subset 
                        # and its corresponding index

                        xi_subsetInd = np.arange(i, psi.shape[0], n)
                        idx.append(xi_subsetInd[np.argmax(psi[xi_subsetInd])])

                    return (1/n)*np.sum(psi[idx]), idx
                     
                # Subgradient of the subobjective
                g_ = lambda mu, idx : (1/n)*np.sum(M.transpose()[:, idx], axis=1)

            elif self.loss == 'log':
                # Define the objective function and the gradient for the log loss function.

                # The psi subobjective for all the datapoints
                f_ = lambda mu : ((1/n)*np.sum(scs.logsumexp((phi@mu), axis=1)), None)

                # The subgradient of the psi subobjective for all the datapoints 
                def g_(mu, idx):
                    expPhi = np.exp(phi@mu)[:, np.newaxis, :]
                    return (1/n)*(np.sum(((expPhi@phi)[:, 0, :] / \
                                np.sum(expPhi, axis=2)).transpose(), axis=1))

            # Check if the warm start is true to reuse the solution from previous call to fit.
            if self.warm_start:
                # Start from a previous solution.
                try:
                    self.params_= self.nesterovOptimization(m, self.params_, f_, g_)
                except AttributeError:
                    self.params_ = self.nesterovOptimization(m, None, f_, g_)
            else:
                self.params_ = \
                                self.nesterovOptimization(m, None, f_, g_)   

            self.mu_ = self.params_['mu']

    def nesterovOptimization(self, m, params_, f_, g_):
        """
        Solution of the CMRC convex optimization(minimization) using the Nesterov accelerated approach.

        Parameters
        ----------
        m : int
            Length of the feature mapping vector

        params_ : a dictionary
            A dictionary of parameters values
            obtained from the previous call to fit
            used as the initial values for the current optimization 
            when warm_start is True.

        f_ : a lambda function/ function of the form - f_(mu)
            It is expected to be a lambda function or a function
            calculating a part of the objective function
            depending on the type of loss function chosen
            by taking the parameters(mu) of the optimization as input.

        g_ : a lambda function of the form - g_(mu, idx)
            It is expected to be a lambda function
            calculating the part of the subgradient of the objective function
            depending on the type of the loss function chosen.
            It takes the as input - 
            parameters (mu) of the optimization and 
            the indices corresponding to the maximum value of subobjective 
            for a given subset of Y (set of labels).

        Return
        ------
        mu : array-like, shape (m,)
            The parameters corresponding to the optimized function value

        f_best_value : float
            The optimized value of the function in consideration.      

        """

        # Initial values for the parameters
        theta_k = 1
        theta_k_prev = 1

        # Initial values for points
        if params_ is not None:
            y_k = params_['mu']
            w_k = params_['w_k']
            w_k_prev = params_['w_k_prev']

            # Length of the points array might change depending on the new dataset
            # as the length of feature mapping might change with the new dataset.
            old_m = y_k.shape[0]
            if old_m != m:

                # Length of each class in the feature mapping depending on old dataset
                old_class_len = int(old_m/self.n_classes)
                # Length of each class in the feature mapping depending on new dataset
                new_class_len = int(m/self.n_classes)

                # New points array with increased size while restoring the old values of points.
                new_y_k = np.zeros(m, dtype = np.float)
                new_w_k = np.zeros(m, dtype = np.float)
                new_w_k_prev = np.zeros(m, dtype = np.float)

                # Restoring the old values of the points obtained from previous call to fit.
                for i in range(self.n_classes):
                    new_class_start = new_class_len * i
                    old_class_start = old_class_len * i

                    if old_m < m:
                        # Increase the size by appending zeros at the end of each class segment.
                        new_y_k[new_class_start : new_class_start + old_class_len] = \
                                        y_k[old_class_start : old_class_start + old_class_len]

                        new_w_k[new_class_start : new_class_start + old_class_len] = \
                                        w_k[old_class_start : old_class_start + old_class_len]

                        new_w_k_prev[new_class_start : new_class_start + old_class_len] = \
                                        w_k_prev[old_class_start : old_class_start + old_class_len]
                    else:
                        # Decrease the size by taking the starting values of each class segment.
                        new_y_k[new_class_start : new_class_start + new_class_len] = \
                                        y_k[old_class_start : old_class_start + new_class_len]

                        new_w_k[new_class_start : new_class_start + new_class_len] = \
                                        w_k[old_class_start : old_class_start + new_class_len]

                        new_w_k_prev[new_class_start : new_class_start + new_class_len] = \
                                        w_k_prev[old_class_start : old_class_start + new_class_len]

                # Updating values.
                y_k = new_y_k
                w_k = new_w_k
                w_k_prev = new_w_k_prev

        else:
            y_k = np.zeros(m, dtype=np.float)
            w_k = np.zeros(m, dtype=np.float)
            w_k_prev = np.zeros(m, dtype=np.float)

        # Setting initial values for the objective function and other results 
        psi, idx = f_(y_k)
        f_best_value = self.lambda_@np.abs(y_k) - self.tau_@y_k + psi
        mu = y_k

        # Iteration for finding the optimal values using Nesterov's extrapolation
        for k in range(1, (self.max_iters + 1)):
            y_k = w_k + theta_k * ((1/theta_k_prev)-1) * (w_k - w_k_prev)

            # Calculating the subgradient of the objective function at y_k
            psi, idx = f_(y_k)
            g_0 = self.lambda_*np.sign(y_k) - self.tau_ + g_(y_k, idx)

            # Update the parameters
            theta_k_prev = theta_k
            theta_k = 2/(k+1)
            alpha_k = 1/(np.power((k+1),(3/2)))

            # Calculate the new points
            w_k_prev = w_k
            w_k = y_k - alpha_k*g_0

            # Check if there is an improvement in the value of the objective function
            f_value = self.lambda_@np.abs(y_k) - self.tau_@y_k + psi
            if f_value < f_best_value:
                f_best_value = f_value
                mu = y_k

        # Check for possible improvement of the objective value for the last generated value of w_k
        psi, idx = f_(w_k)
        f_value = self.lambda_@np.abs(w_k) - self.tau_@w_k + psi

        if f_value < f_best_value:
            f_best_value = f_value
            mu = w_k

        # Return the optimized values in a dictionary
        new_params_ = {'w_k' : w_k, 
                       'w_k_prev' : w_k_prev, 
                       'mu' : mu, 
                       'best_value' : f_best_value, 
                      }

        return new_params_

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
        check_is_fitted(self, "is_fitted_")

        n = X.shape[0]

        # n_instances X n_classes X phi.len
        phi= self.phi.eval(X)
        m = self.phi.len_

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            M= self.phi.getAllSubsetConfig(phi)
            # # F is the sum of phi for different subset of Y for each data point 
            F = M[:, :m]
            cardS= M[:, -1]

            # Compute psi
            psi = np.zeros(n)

            # First we calculate the all possible values of psi for all the points
            psi_arr = (np.ones(cardS.shape[0])-(F@self.mu_ + cardS))/cardS

            for i in range(n):
                # Get psi values for each data point and find the min value
                psi_arr_xi = psi_arr[np.arange(i, psi_arr.shape[0], n)]
                psi[i] = np.min(psi_arr_xi)

            # Conditional probabilities
            hy_x = np.clip(np.ones((n,self.n_classes)) + np.dot(phi, self.mu_) + \
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

            v = np.dot(phi, self.mu_)

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
