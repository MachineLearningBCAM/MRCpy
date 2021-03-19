"""Minimax Risk Classification."""

import cvxpy as cvx
import numpy as np
import scipy.special as scs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import the MRC super class
from MRCpy import _MRC_


class MRC(BaseEstimator, ClassifierMixin, _MRC_):
    """
    Minimax risk classifier using the default constraints and
    implements two kinds of loss functions, namely 0-1 and log loss.
    This is a subclass of the super class _MRC_.

    Attributes
    ----------
    is_fitted_ : bool
        True if the classifier is fitted i.e., the parameters are learnt.

    tau_ : array-like of shape (n_features) or float
        The mean estimates
        for the expectations of feature mappings.

    lambda_ : array-like of shape (n_features) or float
        The variance in the mean estimates
        for the expectations of the feature mappings.

    mu_ : array-like of shape (n_features) or float
        Parameters learnt by the optimization.

    nu_ : float
        Parameter learnt by the optimization.

    zhi_ : array-like of shape (n_features) or float
        Parameters learnt by the optimization when solved using CVXpy.
        This paramter is not required in the prediction stage of MRC

    mu_l_ : array-like of shape (n_features) or float
        Parameters learnt by solving the lower bound optimization of MRC.

    nu_l_ : float
        Parameter learnt by solving the lower bound optimization of MRC.

    zhi_l_ : array-like of shape (n_features) or float
        Parameter learnt by solving the lower bound optimization of MRC
        when solved using CVXpy.

    upper_ : float
        Optimized upper bound of the MRC classifier.

    lower_ : float
        Optimized lower bound of the MRC classifier.

    upper_params_ : a dictionary
        Stores the optimal points and best value
        for the upper bound of the function
        when the warm_start=True.

    params_ : a dictionary
        Stores the optimal points and best value
        for the lower bound of the function
        when the warm_start=True.
    """

    def _minimaxRisk(self, X):
        """
        Solves the minimax risk problem
        for different types of loss (0-1 and log loss).
        The solution of the default MRC optimization
        gives the upper bound of the error.

        Parameters
        ----------
        X : array-like of shape (n_samples1, n_dimensions)
            Training instances used in the optimization.

        """

        # Constants
        m = self.phi.len_

        # Get the learn configurations of phi (feature mapping)
        phi = self.phi.learnConfig(X, self.learn_duplicates)

        # Save the phi configurations for finding the lower bounds
        self.lowerPhiConfigs = phi

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC.

            # Variables
            mu = cvx.Variable(m)
            zhi = cvx.Variable(m)
            nu = cvx.Variable()

            # Cost function
            cost = (1 / 2) * (self.b - self.a).T @ zhi - \
                   (1 / 2) * (self.b + self.a).T @ mu - nu

            # Objective function
            objective = cvx.Minimize(cost)

            # Constraints
            constraints = [zhi + mu >= 0, zhi - mu >= 0]

            if self.loss == '0-1':
                # Constraints in case of 0-1 loss function

                # Exponential number in num_class of linear constraints
                M = self.phi.getAllSubsetConfig(phi)

                # F is the sum of phi
                # for different subset of Y
                # for each data point
                F = M[:, :m]
                cardS = M[:, -1]
                numConstr = M.shape[0]
                constraints.extend([F[i, :] @ mu +
                                    cardS[i] * nu +
                                    cardS[i] * 1 <= 1
                                    for i in range(numConstr)])

            elif self.loss == 'log':
                # Constraints in case of log loss function

                numConstr = phi.shape[0]
                constraints.extend([cvx.log_sum_exp(phi[i, :, :] @ mu +
                                                    np.ones(self.n_classes) *
                                                    nu) <= 0
                                    for i in range(numConstr)])

            self.mu_, self.zhi, self.nu_ = \
                self.trySolvers(objective, constraints, mu, zhi, nu)

            # Upper bound
            self.upper_ = (1 / 2) * (self.b - self.a).T @ self.zhi - \
                          (1 / 2) * (self.b + self.a).T @ self.mu_ - self.nu_

        elif not self.use_cvx:
            # Use the subgradient approach for the convex optimization of MRC

            if self.loss == '0-1':
                # Get the constraint matrix M and
                # the corresponding norm of the class subset h
                M_ = self.phi.getAllSubsetConfig(phi)
                M = M_[:, :m]
                h = M_[:, -1]
                M = M / (h[:, np.newaxis])
                h = 1 - (1 / h)

                # Define the subobjective function and
                # its gradient for the 0-1 loss function.
                def f_(mu):
                    return M @ mu + h

                def g_(mu, idx):
                    return M.transpose()[:, idx]

            elif self.loss == 'log':

                # Define the subobjective function and
                # its gradient for the log loss function.
                def f_(mu):
                    return scs.logsumexp((phi @ mu), axis=1)

                def g_(mu, idx):
                    phi_xi = phi[idx, :, :]
                    expPhi_xi = np.exp(phi_xi @ mu)
                    return (expPhi_xi @ phi_xi).transpose() / np.sum(expPhi_xi)

            # Calculate the upper bound
            # Check if the warm start is true
            # to reuse the solution from previous call to fit.
            if self.warm_start:
                # Start from a previous solution.
                try:
                    self.upper_params_ = \
                        self.nesterovOptimization(m, self.upper_params_,
                                                  f_, g_)
                except AttributeError:
                    self.upper_params_ = self.nesterovOptimization(m, None,
                                                                   f_, g_)
            else:
                self.upper_params_ = self.nesterovOptimization(m, None, f_, g_)

            self.mu_ = self.upper_params_['mu']
            self.nu_ = self.upper_params_['nu']
            self.upper_ = self.upper_params_['best_value']

    def getLowerBound(self):
        """
        Obtains the lower bound of the fitted classifier: unbounded...

        Returns
        -------
        lower : float value
            The lower bound of the error for the fitted classifier.
        """

        # Variables
        m = self.phi.len_

        # Learned feature mappings
        phi = self.lowerPhiConfigs

        # Variables
        n, r, m = phi.shape

        if self.use_cvx:
            # Use CVXpy for the convex optimization of the MRC

            low_mu = cvx.Variable(m)
            low_zhi = cvx.Variable(m)
            low_nu = cvx.Variable()

            # Cost function
            cost = (1 / 2) * (self.b + self.a).T @ low_mu - \
                   (1 / 2) * (self.b - self.a).T @ low_zhi + low_nu

            # Objective function
            objective = cvx.Maximize(cost)

            # Constraints
            constraints = [low_zhi + low_mu >= 0, low_zhi - low_mu >= 0]

            # Number of constraints.
            numConstr = phi.shape[0]

            if self.loss == '0-1':
                # Constraints in case of 0-1 loss function

                # epsilon
                eps = np.clip(1 + phi @ self.mu_ + self.nu_, 0, None)
                c = np.sum(eps, axis=1)
                zeros = np.isclose(c, 0)
                c[zeros] = 1
                eps[zeros, :] = 1 / self.n_classes
                c = np.tile(c, (self.n_classes, 1)).transpose()
                eps /= c
                eps = 1 - eps

                constraints.extend(
                    [phi[j, y, :] @ low_mu + low_nu <= eps[j, y]
                     for j in range(numConstr)
                     for y in range(self.n_classes)])

            elif self.loss == 'log':
                # Constraints in case of log loss function

                # epsilon
                eps = phi @ self.mu_
                eps = np.tile(scs.logsumexp(eps, axis=1),
                              (self.n_classes, 1)).transpose() - eps

                constraints.extend([phi[i, :, :] @ low_mu + low_nu
                                    <= eps[i, :]
                                    for i in range(numConstr)])

            self.mu_l_, self.zhi_l_, self.nu_l_ = \
                self.trySolvers(objective, constraints,
                                low_mu, low_zhi, low_nu)

            # Compute the lower bound
            self.lower_ = (1 / 2) * (self.b + self.a).T @ self.mu_l_ - \
                (1 / 2) * (self.b - self.a).T @ self.zhi_l_ + \
                self.nu_l_

        elif not self.use_cvx:
            # Use the subgradient approach for the convex optimization of MRC

            if self.loss == '0-1':

                # To define the objective function and
                # the gradient for the 0-1 loss function.
                # epsilon
                eps = np.clip(1 + phi @ self.mu_ + self.nu_, 0, None)
                c = np.sum(eps, axis=1)
                zeros = np.isclose(c, 0)
                c[zeros] = 1
                eps[zeros, :] = 1 / self.n_classes
                eps = eps / (c[:, np.newaxis])
                # Using negative of epsilon
                # for the nesterov accelerated optimization
                eps = eps - 1

                # Reshape it for the optimization function
                eps = eps.reshape((n * r,))

            elif self.loss == 'log':

                # To define the objective function and
                # the gradient for the log loss function.
                # Using negative of epsilon
                # for the nesterov accelerated optimization
                eps = phi @ self.mu_ - \
                    scs.logsumexp(phi @ self.mu_, axis=1)[:, np.newaxis]
                eps = eps.reshape((n * r,))

            phi = phi.reshape((n * r, m))

            # Defining the partial objective and its gradient.
            def f_(mu):
                return phi @ mu + eps

            def g_(mu, idx):
                return phi.transpose()[:, idx]

            # Lower bound
            # Check if the warm start is true
            # to reuse the solution from previous call to fit.
            if self.warm_start:
                # Start from a previous solution.
                try:
                    self.lower_params_ = \
                        self.nesterovOptimization(m, self.lower_params_,
                                                  f_, g_)
                except AttributeError:
                    self.lower_params_ = \
                        self.nesterovOptimization(m, None, f_, g_)
            else:
                self.lower_params_ = \
                    self.nesterovOptimization(m, None, f_, g_)

            self.mu_l_ = self.lower_params_['mu']
            self.nu_l_ = self.lower_params_['nu']
            self.lower_ = self.lower_params_['best_value']

            # Maximize the function
            # as the nesterov optimization gives the minimum
            self.lower_ = -1 * self.lower_

        return self.lower_

    def nesterovOptimization(self, m, params_, f_, g_):
        """
        Solution of the MRC convex optimization(minimization)
        using the Nesterov accelerated approach.

        Parameters
        ----------
        m : int
            Length of the feature mapping vector

        params_ : a dictionary
            A dictionary of parameters values
            obtained from the previous call to fit
            used as the initial values for the current optimization
            when warm_start is True.

        f_ : a lambda function of the form - f_(mu)
            It is expected to be a lambda function
            calculating a part of the objective function
            depending on the type of loss function chosen
            by taking the parameters(mu) of the optimization as input.

        g_ : a lambda function of the form - g_(mu, idx)
            It is expected to be a lambda function
            calculating the part of the subgradient of the objective function
            depending on the type of the loss function chosen.
            It takes the as input -
            parameters (mu) of the optimization and
            the index corresponding to the maximum value of data matrix
            obtained from the instances.

        Return
        ------
        mu : array-like, shape (m,)
            The parameters corresponding to the optimized function value

        nu : float
            The parameter corresponding to the optimized function value

        f_best_value : float
            The optimized value of the function in consideration i.e.,
            the upper bound of the minimax risk classification.

        """

        # Initial values for the parameters
        theta_k = 1
        theta_k_prev = 1

        # Initial values for points
        if params_ is not None:
            y_k = params_['mu']
            w_k = params_['w_k']
            w_k_prev = params_['w_k_prev']

            # Length of the points array might change
            # depending on the new dataset
            # as the length of feature mapping might
            # change with the new dataset.
            old_m = y_k.shape[0]
            if old_m != m:

                # Length of each class
                # in the feature mapping depending on old dataset
                old_len = int(old_m / self.n_classes)

                # Length of each class
                # in the feature mapping depending on new dataset
                new_len = int(m / self.n_classes)

                # New points array with increased size
                # while restoring the old values of points.
                new_y_k = np.zeros(m, dtype=np.float)
                new_w_k = np.zeros(m, dtype=np.float)
                new_w_k_prev = np.zeros(m, dtype=np.float)

                # Restoring the old values of the points
                # obtained from previous call to fit.
                for i in range(self.n_classes):
                    new_start = new_len * i
                    old_start = old_len * i

                    if old_m < m:
                        # Increase the size by appending zeros
                        # at the end of each class segment.
                        new_y_k[new_start:new_start + old_len] = \
                            y_k[old_start:old_start + old_len]

                        new_w_k[new_start:new_start + old_len] = \
                            w_k[old_start:old_start + old_len]

                        new_w_k_prev[new_start:new_start + old_len] = \
                            w_k_prev[old_start:old_start + old_len]
                    else:
                        # Decrease the size
                        # by taking the starting values of each class segment.
                        new_y_k[new_start:new_start + new_len] = \
                            y_k[old_start:old_start + new_len]

                        new_w_k[new_start:new_start + new_len] = \
                            w_k[old_start:old_start + new_len]

                        new_w_k_prev[new_start:new_start + new_len] = \
                            w_k_prev[old_start:old_start + new_len]

                # Updating values.
                y_k = new_y_k
                w_k = new_w_k
                w_k_prev = new_w_k_prev

        else:
            y_k = np.zeros(m, dtype=np.float)
            w_k = np.zeros(m, dtype=np.float)
            w_k_prev = np.zeros(m, dtype=np.float)

        # Setting initial values for the objective function and other results
        v = f_(y_k)
        mnu = np.max(v)
        f_best_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + mnu
        mu = y_k
        nu = -1 * mnu

        # Iteration for finding the optimal values
        # using Nesterov's extrapolation
        for k in range(1, (self.max_iters + 1)):
            y_k = w_k + theta_k * ((1 / theta_k_prev) - 1) * (w_k - w_k_prev)

            # Calculating the subgradient of the objective function at y_k
            v = f_(y_k)
            idx = np.argmax(v)
            g_0 = self.lambda_ * np.sign(y_k) - self.tau_ + g_(y_k, idx)

            # Update the parameters
            theta_k_prev = theta_k
            theta_k = 2 / (k + 1)
            alpha_k = 1 / (np.power((k + 1), (3 / 2)))

            # Calculate the new points
            w_k_prev = w_k
            w_k = y_k - alpha_k * g_0

            # Check if there is an improvement
            # in the value of the objective function
            mnu = v[idx]
            f_value = self.lambda_ @ np.abs(y_k) - self.tau_ @ y_k + mnu
            if f_value < f_best_value:
                f_best_value = f_value
                mu = y_k
                nu = -1 * mnu

        # Check for possible improvement of the objective value
        # for the last generated value of w_k
        v = f_(w_k)
        mnu = np.max(v)
        f_value = self.lambda_ @ np.abs(w_k) - self.tau_ @ w_k + mnu

        if f_value < f_best_value:
            f_best_value = f_value
            mu = w_k
            nu = -1 * mnu

        # Return the optimized values in a dictionary
        new_params_ = {'w_k': w_k,
                       'w_k_prev': w_k_prev,
                       'mu': mu,
                       'nu': nu,
                       'best_value': f_best_value,
                       }

        return new_params_

    def predict_proba(self, X):
        """
        Conditional probabilities corresponding to each class
        for each unlabeled instance

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

        # n_instances X n_classes X phi.len
        phi = self.phi.eval(X)

        if self.loss == '0-1':
            # Constraints in case of 0-1 loss function

            # Unnormalized conditional probabilityes
            hy_x = np.clip(1 + np.dot(phi, self.mu_) + self.nu_, 0., None)

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

            # Unnormalized conditional probabilityes
            hy_x = np.vstack(np.sum(np.exp(v - np.tile(v[:, i],
                             (self.n_classes, 1)).transpose()), axis=1)
                             for i in range(self.n_classes)).transpose()
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
