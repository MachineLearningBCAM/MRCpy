'''Constraint generation over the features (column generation) for the
0-1 loss MRC linear program with a large number of features, as described
in [1].

The 0-1 MRC is given by the linear program (10) in [1]

    P :  min   -(tau - lambda)^T mu_1 + (tau + lambda)^T mu_2 + nu
       mu_1, mu_2, nu
         s.t.  F (mu_1 - mu_2) - nu 1 <= b,  mu_1, mu_2 >= 0,

with a constraint for each pair (x, C) in the set S of instances x and
non-empty subsets C of the labels Y, given by the row g_i^T of F and the
component b_i of b in (9) in [1], g_i = sum_{y in C} Phi(x, y) / |C| and
b_i = 1 / |C| - 1. Then, mu = mu_1 - mu_2. Its dual is the linear program
(11) in [1]

    D :  max   -b^T alpha
        alpha
         s.t.  tau - lambda <= F^T alpha <= tau + lambda,
               1^T alpha = 1,  alpha >= 0.

The feature mapping Phi(x, y) is not built. The parameters mu, the mean
estimates tau and their deviations lambda are matrices of shape
(|Y|, n_features), or (1, n_features) for binary classification, and
Phi(x, y)^T mu = s(y) x^T mu_{r(y)}, where each label y uses the row r(y)
of mu with the sign s(y):

    multiclass: r(y) = y, s(y) = 1,
    binary:     r(y) = 0, s(0) = 1, s(1) = -1  (Phi(x, 1) = -Phi(x, 0)).

All the constraints of P are in the model. Instead of the
|S| = n (2^|Y| - 1) constraints of P, the constraints of each instance x_i
are encoded exactly by the scores z_iy = Phi(x_i, y)^T mu (this encoding is
an implementation detail, not described in [1]):

    binary:     nu >= z_i0,  nu >= z_i1  (for all i),  nu >= 1/2,
    multiclass: s_iy >= z_iy - nu + 1,  s_iy >= 0,  sum_y s_iy <= 1,

with auxiliary variables s_iy for multiclass classification, since the
constraints of P for all the subsets C are equivalent to
nu >= 1 + max_C (sum_{y in C} z_iy - 1) / |C|, and
max_C (sum_{y in C} z_iy - 1) / |C| <= t if and only if
sum_y (z_iy - t)_+ <= 1.

The features, that is, the entries (y, j) of mu with their variables in
mu_1 and mu_2, are selected with Algorithm 1 in [1]. At each iteration k,
the subproblem P_J (14) in [1] of P over a subset J of features is solved
(LPSOLVE), giving the worst-case error probability R^k, and the subset of
features is updated from the dual solution alpha^k by the function SELECT
(Algorithm 2 in [1]), with the violations of the dual constraints of the
features

    v = |F^T alpha - tau| - lambda,

where F^T alpha = beta^T X with beta the dual variables of the constraints
with the scores. The features with v_j > eps_2 (epsilon in [1]) are added,
at most m_max (n_max in [1]), the ones with the largest violations if there
are more. The features with mu_j = 0 that are not in the basis are removed
(Algorithm 2 in [1] keeps the features with v_j = 0, which include all the
features with mu_j != 0). The algorithm ends when SELECT adds no feature
or after k_max iterations.

.. [1] Bondugula, K., Mazuelas, S., & Pérez, A. (2023). Efficient
       Learning of Minimax Risk Classifiers in High Dimensions. The 39th
       Conference on Uncertainty in Artificial Intelligence (UAI), 206-215.
       arXiv preprint arXiv:2306.06649.
'''

# Kept for the modules that import this module with `*`.
import itertools as it
import scipy.special as scs
import time
import random

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def mrc_cg(X, tau_mat, lambda_mat, J, m_max, k_max, warm_start, eps_2):
	"""
	Constraint generation algorithm over the features for 0-1 loss Minimax
	Risk Classifiers with a large number of features (Algorithm 1 in [1]).

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	J : `list` of (`int`, `int`)
		Initial subset of features (I in [1]), as entries (y, j) of mu.

	m_max : `int`
		Maximum number of features added in each iteration (n_max in [1]).

	k_max : `int`
		Maximum number of iterations after solving the initial model.

	warm_start : `array`-like of the shape of `tau_mat` or `None`
		Parameters mu as a warm start for the initial model, zero outside
		the features in J.

	eps_2 : `float`
		Violation threshold of the dual constraints (epsilon in [1]).

	Return:
	-------
	mu : `array`-like of the shape of `tau_mat`
		Parameters mu = mu_1 - mu_2 learnt by the algorithm.

	nu : `float`
		Parameter nu of the classifier learnt by the algorithm, the negative
		of the variable nu of the linear program P.

	R : `float`
		Worst-case error probability R^k given by the last model.

	J : `list` of (`int`, `int`)
		Subset of features in the last model (I* in [1]).
	"""

	tau_mat = np.atleast_2d(np.asarray(tau_mat, dtype=float))
	lambda_mat = np.atleast_2d(np.asarray(lambda_mat, dtype=float))
	# Plain int pairs, as the features added in select, so that the features
	# compare consistently in J, J_new and the keys of mu_vars.
	J = [(int(y), int(j)) for y, j in J]

	# Row r(y) of mu and sign s(y) of each label
	if tau_mat.shape[0] == 1:
		rows = np.array([0, 0])
		signs = np.array([1., -1.])
	else:
		rows = np.arange(tau_mat.shape[0])
		signs = np.ones(tau_mat.shape[0])

	# Initial model over the subset J of features
	model = PrimalModel(X, tau_mat, lambda_mat, rows, signs)
	model.add_features(J)
	if warm_start is not None and len(J) > 0:
		model.set_warm_start(warm_start, J)
	mu, nu, R = model.solve()

	# Add and remove features
	J_new = select(model, J, eps_2, m_max)

	k = 0
	while((len(set(J_new).difference(set(J))) != 0) and (k < k_max)):

		# Solve the model over the new subset of features
		mu, nu, R = model.solve()

		# Add and remove features
		J = J_new.copy()
		J_new = select(model, J, eps_2, m_max)

		k = k + 1

	# Features of the last solved model. The features removed by the last
	# selection are zero, and the features added by it are not solved yet.
	J = [col for col in J if col in model.mu_vars]

	# The parameter nu of the classifier is the negative of the variable nu
	# of the linear program.
	return mu, -nu, R, J

class PrimalModel:
	"""
	Linear model P_J (14) in [1] of MRC 0-1 loss over the subset J of
	features, in which each feature (y, j) in J is a pair of variables
	mu_1, mu_2, and the constraints of each instance are encoded by the
	scores z_iy = Phi(x_i, y)^T mu.

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	rows : `array`-like of shape (`n_classes`,)
		Row r(y) of mu of each label.

	signs : `array`-like of shape (`n_classes`,)
		Sign s(y) of each label.

	Attributes:
	-----------
	constrs : `array`-like of shape (`n_samples`, `n_classes`)
		Constraints with the scores z_iy of each instance and label, whose
		dual variables beta give the dual constraints of the features. The
		terms of the variables mu_1 and mu_2 are added with each feature.

	The features in J are given by the following, which are updated as the
	features are added (`add_features`) and removed (`remove_features`)
	along the iterations:

	in_J : `array`-like of the shape of `tau_mat`
		Boolean array with the features (y, j) in J.

	mu_vars : `dict`
		Variables (mu_1, mu_2) of each feature (y, j) in J.
	"""

	def __init__(self, X, tau_mat, lambda_mat, rows, signs):
		self.X = X
		self.tau_mat = tau_mat
		self.lambda_mat = lambda_mat
		self.rows = rows
		self.signs = signs

		self.in_J = np.zeros(tau_mat.shape, dtype=bool)
		self.mu_vars = {}

		n = X.shape[0]
		n_labels = rows.shape[0]
		binary = (tau_mat.shape[0] == 1)

		# Define the MRC 0-1 linear model (primal).
		self.MRC_model = gp.Model("MRC_0_1_primal")
		self.MRC_model.Params.LogToConsole = 0
		self.MRC_model.Params.OutputFlag = 0
		self.MRC_model.setParam('Method', 0)
		self.MRC_model.setParam('LPWarmStart', 2)
		self.MRC_model.setParam('DualReductions', 0)

		self.nu = self.MRC_model.addVar(lb=-GRB.INFINITY, obj=1, name="nu")
		if binary:
			# nu - s(y) x_i^T mu >= 0
			rhs = 0
			self.s = None
		else:
			# s_iy - x_i^T mu_y + nu >= 1, s_iy >= 0, sum_y s_iy <= 1
			rhs = 1
			self.s = self.MRC_model.addVars(n, n_labels, lb=0, name="s")
		self.MRC_model.update()

		# Constraints with the scores, created without the terms in mu,
		# that is, no feature selected at this point.
		self.constrs = np.empty((n, n_labels), dtype=object)
		for i in range(n):
			for y in range(n_labels):
				expr = self.nu if self.s is None else self.nu + self.s[i, y]
				self.constrs[i, y] = self.MRC_model.addConstr(expr >= rhs)

		if binary:
			self.MRC_model.addConstr(self.nu >= 0.5)
		else:
			self.MRC_model.addConstrs(self.s.sum(i, '*') <= 1 for i in range(n))

	def add_features(self, features):
		"""
		Adds the features (y, j) to J, as the variables mu_1 and mu_2 of the
		entry (y, j) of mu, with their coefficients -s(y') x_ij and
		s(y') x_ij in the constraints with the scores of the labels y' that
		use the row y of mu.
		"""

		for col in features:
			r, j = col
			x_j = self.X[:, j]
			nnz = np.flatnonzero(x_j)

			# Coefficients of mu_1 in the constraints with the scores
			labels = np.flatnonzero(self.rows == r)
			coeffs = np.concatenate([-self.signs[y] * x_j[nnz] for y in labels])
			constrs = np.concatenate([self.constrs[nnz, y] for y in labels]).tolist()

			# Objective -(tau - lambda) for mu_1 and (tau + lambda) for mu_2.
			mu_1 = self.MRC_model.addVar(lb=0, obj=self.lambda_mat[col] - self.tau_mat[col],
										 column=gp.Column(coeffs.tolist(), constrs))
			mu_2 = self.MRC_model.addVar(lb=0, obj=self.lambda_mat[col] + self.tau_mat[col],
										 column=gp.Column((-coeffs).tolist(), constrs))
			self.mu_vars[col] = (mu_1, mu_2)
			self.in_J[col] = True
		self.MRC_model.update()

	def remove_features(self, features):
		"""
		Removes the features (y, j) from J, with their variables mu_1 and
		mu_2.
		"""

		for col in features:
			mu_1, mu_2 = self.mu_vars.pop(col)
			self.MRC_model.remove(mu_1)
			self.MRC_model.remove(mu_2)
			self.in_J[col] = False

	def set_warm_start(self, warm_start, J):
		"""
		Sets the warm start of the model at the parameters mu given by
		warm_start over the features J, with the feasible nu and scores s
		for these parameters.
		"""

		mu_start = np.zeros(self.tau_mat.shape)
		for col in J:
			mu_start[col] = warm_start[col]
			self.mu_vars[col][0].PStart = max(mu_start[col], 0)
			self.mu_vars[col][1].PStart = max(-mu_start[col], 0)

		z = self.X @ mu_start.T

		if self.s is None:
			self.nu.PStart = max(np.max(np.abs(z)), 0.5)
		else:
			# nu = 1 + max_i max_C (sum_{y in C} z_iy - 1) / |C|
			n_labels = self.rows.shape[0]
			cumsum = np.cumsum(-np.sort(-z, axis=1), axis=1)
			nu_start = np.max((cumsum - 1) / np.arange(1, n_labels + 1)) + 1
			self.nu.PStart = nu_start
			s_start = np.maximum(z - nu_start + 1, 0)
			for i in range(z.shape[0]):
				for y in range(n_labels):
					self.s[i, y].PStart = s_start[i, y]

	def solve(self):
		"""
		Solves the model (LPSOLVE in [1]), giving the solution
		mu = mu_1 - mu_2 and nu of P_J, with mu zero for the features not in
		J, and the optimal value R^k.
		"""

		self.MRC_model.optimize()
		mu = np.zeros(self.tau_mat.shape)
		for col, (mu_1, mu_2) in self.mu_vars.items():
			mu[col] = mu_1.X - mu_2.X

		return mu, self.nu.X, self.MRC_model.objVal

	def dual_constraints(self):
		"""
		F^T alpha of the dual solution of the solved model, given by
		F^T alpha = beta^T X with beta the dual variables of the constraints
		with the scores. The row r is the sum of s(y) beta_y^T X over the
		labels y that use the row r of mu.
		"""

		beta = np.asarray(self.MRC_model.getAttr("Pi", self.constrs.flatten().tolist())).reshape(self.constrs.shape)
		if self.tau_mat.shape[0] == 1:
			# Binary: Phi(x, 0) = x and Phi(x, 1) = -x
			return ((beta[:, 0] - beta[:, 1]) @ self.X)[np.newaxis, :]
		return beta.T @ self.X

def select(model, J, eps_2, m_max):
	"""
	Updates the subset J of features in the model (function SELECT,
	Algorithm 2 in [1]): removes the features with mu_j = 0 that are not in
	the basis, and adds the features whose dual constraint is violated by
	more than eps_2 (epsilon in [1]), at most m_max (n_max in [1]), the most
	violated ones if there are more.

	Parameters:
	-----------
	model : `PrimalModel`
		The solved model.

	J : `list` of (`int`, `int`)
		Subset of features in the model.

	eps_2 : `float`
		Violation threshold of the dual constraints.

	m_max : `int`
		Maximum number of features added.

	Returns:
	--------
	J_new : `list` of (`int`, `int`)
		Subset of features in the updated model.
	"""

	F_alpha = model.dual_constraints()

	# Violations of the dual constraints of the features not in J,
	# tau - lambda <= F^T alpha <= tau + lambda, that is, the positive part
	# of v = |F^T alpha - tau| - lambda in Algorithm 2 in [1].
	v = np.maximum((F_alpha - model.tau_mat - model.lambda_mat), 0.) + np.maximum((model.tau_mat - model.lambda_mat - F_alpha), 0.)
	J_c = np.argwhere(~model.in_J)
	v = v[~model.in_J]

	# Remove the features with mu_j = 0 that are not in the basis
	removed = []
	for col in J:
		mu_1, mu_2 = model.mu_vars[col]
		mu = mu_1.X - mu_2.X
		basic_status = not ((mu_1.VBasis == -1) and (mu_2.VBasis == -1))

		if (mu == 0) and (basic_status == False):
			removed.append(col)
	model.remove_features(removed)
	J_new = [col for col in J if col not in removed]

	# Add the features
	n_violations = np.sum(v > eps_2)
	if n_violations <= m_max:
		cols_to_add = J_c[v > eps_2]
	else:
		cols_to_add = J_c[np.argsort(v)[::-1][:m_max]]

	cols_to_add = [(int(y), int(j)) for y, j in cols_to_add]
	J_new.extend(cols_to_add)
	model.add_features(cols_to_add)

	return J_new
