"""
Marginally Constrained Cost Sensitive Minimax Risk Classification.
Copyright (C) 2025 Jorge Angulo Rodriguez

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
"""
import sys
import cdd
from cvxpy.utilities.key_utils import to_str
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay, distance

from sklearn.utils import check_array
from libsvm.svmutil import *
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
# Import the MRC super class
from MRCpy import BaseMRC
from MRCpy.solvers.cvx import *
from MRCpy.solvers.cg import *

class LCMRC(BaseMRC):
    '''
    Marginally Constrained Cost Sensitive Minimax Risk Classification

    The class LCMRC implements the method
    Marginally Contrained Cost Sensitive Minimimax Risk Classifiers (MRC)
    An MRC that can use any loss defined by a loss matrix.
    It implements two kinds of prediction methods.


    Parameters
    ----------
    n_classes : `int`
        Number of different possible labels for an instance.

    s : `float`, default = `0.3`
        Parameter that tunes the estimation of expected values
        of feature mapping function. It is used to calculate :math:`\lambda`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \\lambda = s * \\text{std}(\\phi(X,Y)) / \\sqrt{\\left| X \\right|}

        where (X,Y) is the dataset of training samples and their
        labels respectively and
        :math:`\\text{std}(\\phi(X,Y))` stands for standard deviation
        of :math:`\\phi(X,Y)` in the supervised dataset (X,Y).

    loss : `str` (0-1, ordinal, abstention, random, ordinal-squared) or
            `array`-like of shape (`n_classes_classification`, `n_classes`)
            If it's a string, it generates the loss matrix corresponding to the name.
            If it's an array, it will use as the transposed loss maxtrix

    alpha: `float`: penalty for abstention in that type of loss. Defaults to
            abstention penalty (optional parameter). Defaults to 1/(n_classes+1)

    max_iters : `int`, default = `2000`
        Maximum number of iterations to use
        for finding the solution of optimization in
        the subgradient approach.

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier' and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' feature mapping,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

    eps : `float`, default = `1e-4`
        threshold comparing values. for instance, testing if two values are
        different will be done as |value1-value2| < eps
        instead of |value1-value2| == 0

    use_closed_form: `bool`, default `False`
        If true, and if the loss matrix is squared, then it will use the clossed
        for method for classification.

    **phi_kwargs : Additional parameters for feature mappings.
            Groups the multiple optional parameters
            for the corresponding feature mappings(`phi`).

            For example in case of fourier features,
            the number of features is given by `n_components`
            parameter which can be passed as argument
            `AMRC(phi='fourier', n_components=500)`

            The list of arguments for each feature mappings class
            can be found in the corresponding documentation.
    '''

    def __generate_loss_matrix(self, loss, n, alpha=None):

        '''
        This function generates some common loss matrices: abstention,
         0-1, ordinal, ordinal-squared and random.

        Parameters
        ----------
        loss : `str` (0-1, ordinal, abstention, random, ordinal-squared) or
            `array`-like of shape (`n_classes_classification`, `n_classes`)
            If it's a string, it generates the loss matrix.
            If it's an array, it will use as the transposed loss maxtrix

        n : `int`
            number of classes

        alpha : `float`
            abstention penalty (optional parameter). Defaults to 1/(n_classes+1)

        Returns
        -------
        self :
            transposed loss matrix or validated loss matrix.
        '''
        #Check if it is an string and generate the matrix
        # if it is one of the defined types
        if isinstance(loss, str):
            if "abstention" in loss:
                if alpha is None:
                    alpha = 1 / (n + 1)
                LT = np.ones((n, n)) - np.identity(n)
                LT = np.hstack((LT, alpha * np.ones((n, 1))))
                return LT
            if loss == "0-1":
                LT = np.ones((n, n)) - np.identity(n)
                return LT
            elif loss == "ordinal":
                LT = [[np.abs(i - j) for i in range(n)] for j in range(n)]
                LT = np.array(LT)
                return LT
            elif loss == "ordinal-squared":
                LT = [[(i - j) * (i - j) for i in range(n)] for j in range(n)]
                LT = np.array(LT)
                return LT
            elif loss == "random":
                LT = np.random.uniform(0, 1, (n, n))
                # Set the diagonal elements to 0
                np.fill_diagonal(LT, 0)
                return LT
            return 1
        else: #If it is not a string, we validate it as an array
            if not np.issubdtype(loss.dtype, np.number):
                raise TypeError("Array must be numeric")

            if loss.ndim != 2:
                raise ValueError(f"Expected {2} dimensions")

            return loss

    def __init__(self,
                 s=0.3,
                 loss = "0-1",
                 n_classes=2,
                 alpha = None,
                 eps=1e-4,
                 phi='linear',
                 use_closed_form = False,
                 **phi_kwargs):
        self.eps = eps
        self.use_closed_form = use_closed_form
        self.cvx_solvers = ['GUROBI', 'MOSEK', 'SCS', 'CLARABEL', 'CVXOPT',
                            'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP',
                           "COPT", "XPRESS", "PROXQP", "CPLEX", 'SCIPY']
        self.loss_matrix = self.__generate_loss_matrix(loss, n_classes, alpha)
        self.n_classes = n_classes #|Y|
        self.n_classification_classes_ = self.loss_matrix.shape[1]  # |T|
        super().__init__(s=s,
                         loss = loss,
                         phi=phi, **phi_kwargs)

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Solves the minimax risk problem
        for different the given loss matrix
        The solution of the default L-MRC optimization
        gives the upper bound of the error.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            Mean estimates
            for the expectations of feature mappings.

        lambda_ : `array`-like of shape (`n_features` * `n_classes`)
            Variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
            Number of labels in the dataset.

        Returns
        -------
        self :
            Fitted estimator

        '''
        self.n_classes = n_classes
        self.tau_ = check_array(tau_, accept_sparse=True, ensure_2d=False)
        self.lambda_ = check_array(lambda_, accept_sparse=True,
                                  ensure_2d=False)
        phi = self.compute_phi(X)

        phi = np.unique(phi, axis=0)

        # Constants
        m = phi.shape[2]
        n = phi.shape[0]

        # Save the phi configurations for finding the lower bounds
        self.lowerPhiConfigs = phi

        # Get H-form of the polyhedron of restrictions defined by the Convex learning problem given by the loss matrix
        self.b, self.A = self.__get_A_b(self.loss_matrix)

        # Let's compute homogenization matrix N

        n_0 = np.hstack([np.ones(self.n_classification_classes_), np.zeros(self.n_classes)])
        N = np.hstack([self.loss_matrix, np.identity(self.n_classes)])
        N = np.vstack([n_0, N])
        self.N = N

        # We compute the intersection between the rays generating cone(N) and Hyperplane H(alpha): alpha*z_0+z_1+...+z_{|Y|}
        points = self.__compute_intersection_points(self.loss_matrix, self.n_classes, self.n_classification_classes_)
        #Triangulate the result using Dalaunay triangulation
        try:
            tri = Delaunay(points, furthest_site=True)
        except:
            tri = Delaunay(points)

        #Learing problem

        #Define CVXpy problem
        mu = cvx.Variable(m)
        eta = cvx.Variable(m)
        nu = cvx.Variable(1)

        an = self.tau_ - self.lambda_ / np.sqrt(n)

        bn = self.tau_ + self.lambda_ / np.sqrt(n)

        # Compute C matrix and positive indices
        C = np.array(self.A @ np.ones(self.A.shape[1]))
        pos_index = np.where(C > 0)[0]

        constraints = []
        constraints.append(eta + mu >= 0)
        constraints.append(eta - mu >= 0)

        def var_phi_2(phi_xdot):
            return cvx.min(
                cvx.hstack(
                    [(self.b[index] - (C @ phi_xdot @ mu))[index] / C[index] for index in pos_index]))

        term2 = (1 / n) * cvx.sum([var_phi_2(phi[j, :, :]) for j in range(n)])
        objective = cvx.Minimize(0.5 * (bn - an) @ eta - 0.5 * (bn + an) @ mu - term2)
        problem = cvx.Problem(objective, constraints)

        for s in self.cvx_solvers:
            try:
                problem.solve(solver=s)  # , max_iters=m_iter)
                self.mu_ = mu.value
                self.nu_ = nu.value
            except:
                pass
            if mu.value is not None and nu.value is not None:
                break
        if mu.value is None or nu.value is None:
            raise RuntimeError("Learning problem is not feasible")

        # Get submatrices of N and compute the inverse matrices
        flag = True
        for T in tri.simplices:
            for j in T:
                if (np.abs(np.sum(tri.points[j] - points[j])) > 0.0001):
                    flag = False
        if not flag:
            "Error in array ordering during triangulation"
            exit()
        # We store the inverted simplex matrices for later, as they are needed for
        # classification
        TN = [[N[:, np.sort(T)], np.sort(T)] for T in tri.simplices]
        self.Inverted_Matrices = [[np.linalg.inv(TT[0]), TT[1]] for TT in TN]


        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        '''
                Conditional probabilities corresponding to each class
                for each unlabeled input instance

                Parameters
                ----------
                X : `array`-like of shape (`n_samples`, `n_dimensions`)
                    Testing instances for which
                    the prediction probabilities are calculated for each class.

                Returns
                -------
                hy_x : `ndarray` of shape (`n_samples`, `n_classes`)
                    Probabilities :math:`(p(y|x))` corresponding to the predictions
                    for each class.

        '''
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        phi = self.compute_phi(X)
        hy_x = []
        for i, x in enumerate(X):
            cx = self.nu_ * np.ones(self.n_classes) + phi[i, :, :] @ self.mu_
            cx_1 = np.hstack((1, -cx))
            max_index = None
            # If the matrix is squared and the user selects the option, we use the
            # closed classificiation rule
            if self.use_closed_form and self.n_classes == self.n_classification_classes_:
                h_t = self.__find_closed_form_classification_rule(cx,
                                                                    self.loss_matrix, self.n_classes)
            else: # Otherwise we use the general classifcation rule
                h_t = self.__find_classification_rule(self.Inverted_Matrices,
                                                      cx_1,
                                                      (self.n_classes +
                                                       self.n_classification_classes_),
                                                      self.eps)
            # If all elements are positive then we have found a valid classification rule
            # Otherwise the point falls outside the polyhedron of restrictions,
            # and we find and alternative classification rule based on projceting the point
            if not np.all(h_t >= -self.eps) and max_index is None:
                h_t = self.__find_alternative_classification_rule_2(self.Inverted_Matrices,
                                                                    cx_1, self.N)
            if h_t is not None and np.all(h_t >= -self.eps):
                # Remove auxiliary dimensions corresponding to Slack variables
                h = h_t[:self.n_classification_classes_]
            else:
                # If all fails, we just return the "uniform" with respect to the loss
                h = self.__uniform_distribution()
            hy_x.append(h)
        return np.array(hy_x)

    def get_upper_bound(self):
        '''
        Returns the upper bound on the expected loss for the fitted classifier.

        Returns
        -------
        upper_bound : `float`
            Upper bound of the expected loss for the fitted classifier.
        '''

        if self.deterministic:
            # Number of instances in training
            n = self.lowerPhiConfigs.shape[0]

            # Feature mapping length
            m = self.phi.len_
            mu = cvx.Variable(m)
            eta = cvx.Variable(m)
            nu = cvx.Variable(1)

            an = self.tau_ - self.lambda_ / np.sqrt(n)

            bn = self.tau_ + self.lambda_ / np.sqrt(n)

            objective = cvx.Minimize(0.5 * (bn - an) @ eta - 0.5 * (bn + an) @ mu - nu)

            constraints = [
                self.A @ (cvx.matmul(self.lowerPhiConfigs[i, :, :], mu) +
                          cvx.matmul(nu, np.ones((1, self.n_classes)))) <= self.b
                for i in range(n)
            ]
            constraints.append(eta + mu >= 0)
            constraints.append(eta - mu >= 0)
            problem = cvx.Problem(objective, constraints)
            for s in self.cvx_solvers:
                try:
                    problem.solve(solver=s)
                except:
                    pass
            self.upper_ = (0.5 * (bn - an) @ eta.value -
                           0.5 * (bn + an) @ mu.value - nu.value)

        return self.upper_

    def get_lower_bound(self, X, Y):
        '''
            Obtains the lower bound on the expected loss for the fitted classifier.

            Parameters
            ----------
            X : `array`-like of shape (`n_samples`, `n_dimensions`)
                Test instances for which the labels are to be predicted
                by the MRC model.

            Y : `array`-like of shape (`n_samples`, 1), default = `None`
                Labels corresponding to the testing instances
                used to compute the error in the prediction.

            Returns
            -------
            lower_bound : `float`
                Lower bound of the error for the fitted classifier.
        '''
        # Classifier should be fitted to obtain the lower bound
        check_is_fitted(self, "is_fitted_")

        # Learned feature mappings
        phi = self.lowerPhiConfigs
        # Variables
        n = phi.shape[0]
        m = phi.shape[2]
        #Get all errors
        errors = self.error(X,Y,False)

        #Define the cvx problem that determines the lower bound
        mu = cvx.Variable(m)
        eta = cvx.Variable(m)
        nu = cvx.Variable(1)

        an = self.tau_ - self.lambda_ / np.sqrt(n)

        bn = self.tau_ + self.lambda_ / np.sqrt(n)

        objective = cvx.Maximize(- 0.5 * (bn - an) @ eta + 0.5 * (bn + an) @ mu + nu)

        constraints = [
            cvx.matmul(self.lowerPhiConfigs[i, :, :], mu) +
            cvx.matmul(nu, np.ones((1, self.n_classes))) <= errors[i]
            for i in range(n)
        ]
        constraints.append(eta + mu >= 0)
        constraints.append(eta - mu >= 0)
        problem = cvx.Problem(objective, constraints)

        for s in self.cvx_solvers:
            try:
                problem.solve(solver=s)
            except:
                pass
        self.lower_ = (- 0.5 * (bn - an) @ eta.value +
                       0.5 * (bn + an) @ mu.value + nu.value)

        return self.lower_

    def error(self, X, Y, mean=True):
        '''
        Return the error obtained for the given test
         data and labels using the loss function given

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Y : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the testing instances
            used to compute the error in the prediction.

        mean: `Boolean`, defaults to `True`. If true, the method returns the mean error.
            If false, the method returns an array with all the errors.

        Returns
        -------
        error : `float` (if mean is `True`)
            Mean error of the learned MRC classifier

            OR

        error: `array`-like of shape (`n_samples`) (if mean is `False`)
            Array with the errors commited when classifying each instance
        '''
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        Y_pred = self.predict(X)
        error = []
        for i, y in enumerate(Y):
            if isinstance(Y_pred[i], (int, np.int64)):
                q = np.zeros(self.n_classification_classes_)
                q[Y_pred[i]] = 1
            else:
                q = Y_pred[i]
            p = np.zeros(self.n_classes)
            p[y] = 1
            if q is not None and np.all(q >= -self.eps):
                q_t = q[:self.n_classification_classes_]
                q_t = normalize([q_t], norm="l1")[0]
                max_index = np.argmax(q_t)
                q_max = np.zeros_like(q_t)
                q_max[max_index] = 1
                e = p @ self.loss_matrix @ q_max
                error.append(e)
        error = np.array(error)
        if mean:
            return np.average(error)
        else:
            return error
    # This version of the predict(X) function
    # is almost the same as the parent's class method
    # However, small adjustments need to be done in order to prevent
    # issues with non-squared loss matrices
    def predict(self, X):
        '''
        Predicts classes for new instances using a fitted model.

        Returns the predicted classes for the given instances in `X`
        using the probabilities given by the function `predict_proba`.
        This method overwrites the parent's class method so that
        non-square loss matrices are allowed.
        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : `array`-like of shape (`d_2`)
            Predicted labels corresponding to the given instances.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Get the prediction probabilities for the classifier
        proba = self.predict_proba(X)

        if self.deterministic:
            y_pred = np.argmax(proba, axis=1)
        else:
            np.random.seed(self.random_state)
            y_pred = [np.random.choice(self.n_classification_classes_, size=1, p=pc)[0]
                      for pc in proba]

        return y_pred

    ##################################################################################
    ######### AUXILIARY FUNCTIONS ####################################################
    ##################################################################################

    # Transforms the problem restrictions into Aw_x<=b form given LT . Returns A, b
    def __get_A_b(self, LT):
        '''
            Computes the H-form of the polyhedron of restrictions.
            That is polyhedron L is given as ${\omega \in R^m | A\omega < b}$

            Parameters:
            -----------
            LT: `array`-like of shape (`n_dimensions`,`n_classification_classes_`)
                Transposed loss matrix

            Returns:
            --------
            A: `array`-like of shape (`restrictions`,`m`),
             where `restrictions` is not known a-priori
            b `array`-like of shape (`restrictions`)
        '''
        # CP matrix encodes the restrictions. It transforms them into equality restrictions by adding Slack variables.
        # It also adds new restriction to encode that ||h||=1
        # CP = |unos(|T|) zeros(|Y|)|
        # |-L^T      -Id(|Y|) |
        d_1 = LT.shape[0]  # |Y|
        d_2 = LT.shape[1]  # |T|
        CP = np.hstack((-LT, -np.identity(d_1)))
        v = np.hstack((np.ones(d_2), np.zeros(d_1)))
        CP = np.vstack((v, CP))
        # Transform CP matrix into shape that cdd can use
        V = np.vstack(
            [
                np.hstack([np.zeros((CP.shape[1], 1)), CP.T]),
                np.hstack([1, np.zeros(CP.shape[0])]),
            ]
        )
        # Define CDD polyhedron from the given restriction matrix
        mat = cdd.matrix_from_array(V)
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.polyhedron_from_matrix(mat)

        ineq = cdd.copy_inequalities(P)
        H = np.array(ineq.array)
        A = []
        for i in range(H.shape[0]):
            A.append(-H[i, 1:])
        A = np.array(A)
        b = np.zeros(A.shape[0])
        m = A.shape[1]
        #Add intersecction with hiperplane x = 1
        v = np.zeros((2, m))
        v[0][0] = 1
        v[1][0] = -1
        b = np.concatenate((b, np.array([1, -1])))
        A = np.vstack([A, v])
        b = b.reshape((b.shape[0], 1))
        #print(A)
        mat = cdd.matrix_from_array(np.hstack([b, -A]))
        mat.rep_type = cdd.RepType.INEQUALITY
        cdd.matrix_canonicalize(mat)
        g = cdd.copy_generators(cdd.polyhedron_from_matrix(mat))
        gg = np.array(g.array)
        gg = np.hstack((np.array(gg[:, 0][:, np.newaxis]),np.array(gg[:, 2:])))
        mat = cdd.matrix_from_array(gg)
        mat.rep_type = cdd.RepType.GENERATOR
        g = cdd.polyhedron_from_matrix(mat)
        bA = np.array(cdd.copy_inequalities(g).array)
        b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])
        return b, A

    # Computes the intersection points between the cone generated by matrix N and a hyperplane that intersects all rays of the cone.
    # Coordinates are given in reference frame sush that the hyperplane has coordinates z_0=1. Points are projected to one dimension less,
    # and the first coordinate is removed (see proof details in the paper)
    def __compute_intersection_points(self, LT, d_1, d_2):
        L_1_norms = [np.dot(LT[:, i], np.ones(d_1)) for i in range(d_2)]
        # Compute alpha
        alpha = np.max(L_1_norms)
        alpha += 0.1 * np.abs(alpha)
        # print(alpha)
        # Compute the gamma points for the fist |T| rays as 1/(alpha+<l_(i,·),1>)
        gama = [1 / (alpha + L_1_norms[i]) for i in range(d_2)]
        # Compute intersection points for those rays
        int_points_1 = [gama[i] * np.hstack((1, LT[:, i])) for i in range(d_2)]
        # Compute the intersection points for the las |Y| rays as (e,e_i)^T of the form gamma_i*(0,e_i)^T where gamma_i=1
        int_points_2 = [np.zeros(d_1 + 1) for i in range(d_1)]
        for i in range(d_1):
            int_points_2[i][i + 1] = 1

        int_points = np.concatenate((int_points_1, int_points_2))
        # print(int_points)
        # Perform change of reference to a new basis, where Hyperplane H is given by z_0=1
        #   normal vector (alpha,1,1,...,1):
        nv = np.ones(d_1 + 1)
        nv[0] = alpha
        # B has the normal vector as thje first column. The other columns are the generators of H, (1,-alpha*e_i)^T 1<=i<=|Y|
        B = -alpha * np.identity(d_1 + 1)
        B[0, :] = 1
        B[:, 0] = nv
        # print(B)
        # K matrix necessary to calculate change of reference matrix
        K = np.zeros((d_1 + 1, d_1 + 1))
        K[0, 1:] = np.ones(d_1)
        # Change of reference matrix D
        # D = (np.identity(d_1+1)+K) @ np.linalg.inv(B)
        D = np.linalg.inv(B)
        # print(D)
        op = np.zeros(d_1 + 1)
        op[0] = 1
        o = (1 / alpha) * op
        # Points in the new refference frame
        new_int_points = [(D @ (v - o) + op) for v in int_points]
        # Take all but the first coordinates
        projected_points = [p[1:] for p in new_int_points]
        # Points are ordered in the same order as the columns of N
        points = np.array(projected_points)
        return points

    def __find_classification_rule(self, Inverted_Matrices, cx_1, n, eps):
        for NI in Inverted_Matrices:
            h_t = np.zeros(n)  # Create an array of zeros at the start of each iteration
            # For each simplex, we calculate h_t as the coordinates of cx_1 in that simplex
            # So, cx_1 = N_s @ h_t, where N is the matrix that generates the simplex.
            # NI = N_x^-1 of that matrix, so h_t = NI @ cx_1
            aux = NI[0] @ cx_1
            # We just need to adjust the coordinates, since NI squared matrix of size n_clases^2
            # And we need the coordinates for h_t in the general coordinate framework with
            # dimensions = n_classes + n_classification_classes_
            h_t[NI[1]] = aux
            # If we find a non-negative rule, then is valid and we are done. Otherwise keep looking
            if np.all(h_t >= -eps):
                return h_t

        return h_t  # Return the final h_t if no break condition is met

    # Clossed for classification. See article for explanation.
    # In this case we need squared and non-singular loss matrix
    def __find_closed_form_classification_rule(self, cx, LL, n):
        h_3 = - np.linalg.inv(LL) @ (cx + (self.lambda_ @ np.abs(self.mu_)) / np.sqrt(n))
        h_3 = np.where(h_3 > 0, h_3, 0)
        h_3 = h_3 / np.sum(h_3)
        return h_3

    # This is the same as __find_classification_rule, but we just return all
    # possible classification rules (one per simplex).
    # Many might not be valid classification rules
    # See __find_classification_rule for comments on how it works
    def __find_all_classification_rules(self, Inverted_Matrices, cx_1, n):
        h_list = []
        for NI in Inverted_Matrices:
            h_t = np.zeros(n)
            aux = NI[0] @ cx_1
            h_t[NI[1]] = aux
            h_list.append(h_t)

        return h_list

    # This function is used when no valid classification rule is found by standard methods.
    # This means that cx is not in the learned polyhedron of restrictions, so we need an approximation.
    # This is done by projecting all the possible rules and finding the closest one
    def __find_alternative_classification_rule_2(self, Inverted_Matrices, cx_1, N):
        h_t = None
        d = sys.maxsize
        # We find all potential classification rules, that is, the coordinates of cx_1 for each simplex
        # If this function is used, it means that they are all non-valid solutions:
        # they will have negative components.
        h_list = self.__find_all_classification_rules(Inverted_Matrices, cx_1,
                                                      (self.n_classes + self.n_classification_classes_))
        for hh in h_list:
            # Substitute all negative components with 0
            h_proj = np.maximum(0, hh)
            # Normalize, so that the resulting array is still a provability distribution
            h_proj /= np.sum(hh)
            # Now we use the new coordinates to "project" cx_1 into the simplex corresponding to this
            # particular h_t
            cx_1_proj = N @ h_proj
            # We compute the distance to the ordiginal, and we will choose the projected point that is
            # closest to the original.
            d_aux = distance.euclidean(cx_1_proj, cx_1)
            if d_aux < d:
                h_t = h_proj
                d = d_aux
        return h_t

    def __uniform_distribution(self):
        q = cvx.Variable(self.n_classification_classes_)
        p = np.ones(self.n_classes)
        p = (1 / self.n_classes) * p
        objective = cvx.Minimize((p @ self.loss_matrix) @ q)
        constraints = [(np.ones(self.n_classification_classes_) @ q) == 1, q >= 0]
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=self.cvx_solvers[0])
        uniform = q.value
        return uniform

