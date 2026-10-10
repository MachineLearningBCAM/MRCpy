'''Column generation for the 0-1 loss MRC linear program with a large
number of features, as described in [1].

The 0-1 MRC is given by the linear program (4) in [1]

    P :  min   -(tau - lambda)^T mu_1 + (tau + lambda)^T mu_2 + nu
       mu_1, mu_2, nu
         s.t.  F (mu_1 - mu_2) - nu 1 <= b,  mu_1, mu_2 >= 0,

with a constraint for each instance x and non-empty subset C of the labels
Y, given by the row F_c = sum_{y in C} Phi(x, y)^T / |C| of F and the
component b_c = 1 / |C| - 1 of b. Then, mu = mu_1 - mu_2.

The feature mapping Phi(x, y) is not built. The parameters mu, the mean
estimates tau and their deviations lambda are matrices of shape
(|Y|, n_features), or (1, n_features) for binary classification, and
Phi(x, y)^T mu = s(y) x^T mu_{r(y)}, where each label y uses the row r(y)
of mu with the sign s(y):

    multiclass: r(y) = y, s(y) = 1,
    binary:     r(y) = 0, s(0) = 1, s(1) = -1  (Phi(x, 1) = -Phi(x, 0)).

All the constraints of P are in the model. Instead of the n (2^|Y| - 1)
constraints of P, the constraints of each instance x_i are encoded exactly
by the scores z_iy = Phi(x_i, y)^T mu:

    binary:     nu >= z_i0,  nu >= z_i1  (for all i),  nu >= 1/2,
    multiclass: s_iy >= z_iy - nu + 1,  s_iy >= 0,  sum_y s_iy <= 1,

with auxiliary variables s_iy for multiclass classification, since the
constraints of P for all the subsets C are equivalent to
nu >= 1 + max_C (sum_{y in C} z_iy - 1) / |C|, and
max_C (sum_{y in C} z_iy - 1) / |C| <= t if and only if
sum_y (z_iy - t)_+ <= 1.

The features, that is, the entries (y, j) of mu with their variables in
mu_1 and mu_2, are generated (function FEAT of Algorithm 1 in [1]): a
subset J of features is in the model and P is solved over J at each
iteration. A feature j not in J is added if its dual constraint (6) in [1],
tau_j - lambda_j <= (F^T alpha)_j <= tau_j + lambda_j, is violated by more
than eps_2, where F^T alpha = beta^T X with beta the dual variables of the
constraints with the scores. The features in J with mu_j = 0 that are not
in the basis are removed.

.. [1] Bondugula, K., Mazuelas, S., & Pérez, A. (2025).
       Efficient Large-Scale Learning of Minimax Risk Classifiers.
       arXiv preprint arXiv:2511.17626.
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
	Column generation algorithm for 0-1 loss Minimax Risk Classifiers with a
	large number of features (Algorithm 1 in [1], without the generation of
	constraints).

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	J : `list` of (`int`, `int`)
		Initial subset of features, as entries (y, j) of mu.

	m_max : `int`
		Maximum number of features added in each iteration.

	k_max : `int`
		Maximum number of iterations after solving the initial model.

	warm_start : `array`-like of the shape of `tau_mat` or `None`
		Parameters mu as a warm start for the initial model, zero outside
		the features in J.

	eps_2 : `float`
		Violation threshold of the dual constraints.

	Return:
	-------
	mu : `array`-like of the shape of `tau_mat`
		Parameters mu = mu_1 - mu_2 learnt by the algorithm.

	nu : `float`
		Parameter nu of the classifier learnt by the algorithm, the negative
		of the variable nu of the linear program P.

	R : `float`
		Worst-case error probability given by the last model.

	J : `list` of (`int`, `int`)
		Subset of features in the last model.
	"""

	tau_mat = np.atleast_2d(np.asarray(tau_mat, dtype=float))
	lambda_mat = np.atleast_2d(np.asarray(lambda_mat, dtype=float))
	# Plain int pairs, as the features added in select, so that the features
	# compare consistently in J, J_new and the keys of mu_vars.
	J = [(int(y), int(j)) for y, j in J]

	# Initial model
	MRC_model, mu_vars, constrs = mrc_cg_init_model(X,
													tau_mat,
													lambda_mat,
													J,
													warm_start)

	# Add and remove features
	J_new = select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, J, eps_2, m_max)

	k = 0
	while((len(set(J_new).difference(set(J))) != 0) and (k < k_max)):

		# Solve the model over the new subset of features
		MRC_model.optimize()

		# Add and remove features
		J = J_new.copy()
		J_new = select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, J, eps_2, m_max)

		k = k + 1

	# Solution of the last solved model. The features removed by the last
	# selection are zero, and the features added by it are not solved yet.
	J = [col for col in J if col in mu_vars]
	mu = np.zeros(tau_mat.shape)
	for col in J:
		mu[col] = mu_vars[col][0].X - mu_vars[col][1].X

	# The parameter nu of the classifier is the negative of the variable nu
	# of the linear program.
	nu = (-1) * MRC_model.getVarByName("nu").X
	R = MRC_model.objVal

	return mu, nu, R, J

def mrc_cg_init_model(X, tau_mat, lambda_mat, J, warm_start=None):
	"""
	Builds and solves the linear model P of MRC 0-1 loss over the subset J
	of features.

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	J : `list` of (`int`, `int`)
		Subset of features, as entries (y, j) of mu.

	warm_start : `array`-like of the shape of `tau_mat`, default=`None`
		Parameters mu as a warm start, zero outside the features in J.

	Return:
	-------
	MRC_model : A MRC object in GUROBI
		The solved linear model.

	mu_vars : `dict`
		Variables (mu_1, mu_2) of each feature (y, j) in the model.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores z_iy of each instance and label, whose
		dual variables beta give the dual constraints of the features.
	"""

	n, d = X.shape
	binary = (tau_mat.shape[0] == 1)
	n_score_cols = 2 if binary else tau_mat.shape[0]

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_primal")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	MRC_model.setParam('Method', 0)
	MRC_model.setParam('LPWarmStart', 2)
	MRC_model.setParam('DualReductions', 0)

	nu = MRC_model.addVar(lb=-GRB.INFINITY, obj=1, name="nu")
	if binary:
		# nu - s(y) x_i^T mu >= 0, with s = (1, -1)
		rhs = 0
		s = None
	else:
		# s_iy - x_i^T mu_y + nu >= 1, s_iy >= 0, sum_y s_iy <= 1
		rhs = 1
		s = MRC_model.addMVar((n, n_score_cols), lb=0, name="s")
	MRC_model.update()

	# Constraints with the scores, created without the terms in mu,
	# that is, no variable selected at this point.
	# The terms -x_i^T mu_y are added feature by feature in add_var, as the
	# features enter the model, and leave it with them.
	constrs = np.empty((n, n_score_cols), dtype=object)
	for i in range(n):
		for y in range(n_score_cols):
			expr = nu if s is None else nu + s[i, y].item()
			constrs[i, y] = MRC_model.addConstr(expr >= rhs)

	if binary:
		MRC_model.addConstr(nu >= 0.5)
	else:
		MRC_model.addConstr(s.sum(axis=1) <= 1)

	# Add the initial features.
	mu_vars = {}
	for col in J:
		add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col)
	MRC_model.update()

	# Warm start with a feasible point for the given coefficients.
	if warm_start is not None and len(J) > 0:
		mu_start = np.zeros(tau_mat.shape)
		for col in J:
			mu_start[col] = warm_start[col]
			mu_vars[col][0].PStart = max(mu_start[col], 0)
			mu_vars[col][1].PStart = max(-mu_start[col], 0)

		z = X @ mu_start.T

		if binary:
			nu.PStart = max(np.max(np.abs(z)), 0.5)
		else:
			# nu = 1 + max_i max_C (sum_{y in C} z_iy - 1) / |C|
			cumsum = np.cumsum(-np.sort(-z, axis=1), axis=1)
			nu_start = np.max((cumsum - 1) / np.arange(1, n_score_cols + 1)) + 1
			nu.PStart = nu_start
			s.PStart = np.maximum(z - nu_start + 1, 0)

	# Solve the model
	MRC_model.optimize()

	return MRC_model, mu_vars, constrs

def add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col):
	"""
	Adds the feature col = (y, j) to the model, as the variables
	mu_1 and mu_2 of the entry (y, j) of mu.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The model to be updated.

	mu_vars : `dict`
		Variables (mu_1, mu_2) of each feature (y, j) in the model.
		It is updated with the new feature.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores z_iy of each instance and label.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	col : (`int`, `int`)
		Feature (y, j) to be added.
	"""

	y, j = col
	x_j = X[:, j]
	nnz = np.flatnonzero(x_j)

	# Coefficients of mu_1 in the constraints with the scores.
	if tau_mat.shape[0] == 1:
		# Binary: -x_ij for label 0 and +x_ij for label 1.
		coeffs = np.concatenate([-x_j[nnz], x_j[nnz]])
		rows = np.concatenate([constrs[nnz, 0], constrs[nnz, 1]])
	else:
		coeffs = -x_j[nnz]
		rows = constrs[nnz, y]

	# Objective -(tau - lambda) for mu_1 and (tau + lambda) for mu_2.
	rows = rows.tolist()
	mu_1 = MRC_model.addVar(lb=0, obj=lambda_mat[col] - tau_mat[col],
							column=gp.Column(coeffs.tolist(), rows))
	mu_2 = MRC_model.addVar(lb=0, obj=lambda_mat[col] + tau_mat[col],
							column=gp.Column((-coeffs).tolist(), rows))
	mu_vars[col] = (mu_1, mu_2)

def select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, J, eps_2, m_max):
	"""
	Updates the subset J of features in the model (function FEAT of
	Algorithm 1 in [1]): removes the features with mu_j = 0 that are not in
	the basis, and adds the features whose dual constraint is violated by
	more than eps_2, at most m_max, the most violated ones if there are more.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The solved linear model.

	mu_vars : `dict`
		Variables (mu_1, mu_2) of each feature (y, j) in the model.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores z_iy of each instance and label.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

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

	# Dual variables beta of the constraints with the scores, and
	# F^T alpha = beta^T X in the dual constraints of the features.
	beta = np.asarray(MRC_model.getAttr("Pi", constrs.flatten().tolist())).reshape(constrs.shape)
	if tau_mat.shape[0] == 1:
		# Binary: Phi(x, 0) = x and Phi(x, 1) = -x
		F_alpha = ((beta[:, 0] - beta[:, 1]) @ X)[np.newaxis, :]
	else:
		F_alpha = beta.T @ X

	# Violations of the dual constraints of the features not in J,
	# tau - lambda <= F^T alpha <= tau + lambda.
	v = np.maximum((F_alpha - tau_mat - lambda_mat), 0.) + np.maximum((tau_mat - lambda_mat - F_alpha), 0.)
	not_in_model = np.ones(tau_mat.shape, dtype=bool)
	for col in J:
		not_in_model[col] = False
	J_c = np.argwhere(not_in_model)
	v = v[not_in_model]

	J_new = J.copy()

	# Remove the features with mu_j = 0 that are not in the basis
	for col in J:
		mu_1, mu_2 = mu_vars[col]
		mu = mu_1.X - mu_2.X
		basic_status = not ((mu_1.VBasis == -1) and (mu_2.VBasis == -1))

		if (mu == 0) and (basic_status == False):
			J_new.remove(col)

			# Remove from the gurobi model
			MRC_model.remove(mu_1)
			MRC_model.remove(mu_2)
			del mu_vars[col]

	# Add the features
	n_violations = np.sum(v > eps_2)
	if n_violations <= m_max:
		cols_to_add = J_c[v > eps_2]
	else:
		cols_to_add = J_c[np.argsort(v)[::-1][:m_max]]

	for y, j in cols_to_add:
		col = (int(y), int(j))
		J_new.append(col)
		add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col)

	return J_new
