'''Column generation for the 0-1 loss MRC linear program.

The linear program is built from the features without one-hot encoding.
For binary classification, with phi(x, 0) = x, phi(x, 1) = -x and the
parameters mu of shape (1, n_features), it is

    min  -tau . mu + lambda . |mu| + nu
    s.t. nu >= x_i . mu,  nu >= -x_i . mu,  nu >= 1/2      for all i.

For multiclass classification, with the parameters mu of shape
(n_classes, n_features) and the scores z_iy = x_i . mu_y, the constraint
of each instance over all the subsets S of classes,

    nu >= 1 + max_S (sum_{y in S} z_iy - 1) / |S|,

is encoded exactly by K + 1 constraints and K auxiliary variables s_iy:

    s_iy >= z_iy - nu + 1,  s_iy >= 0,  sum_y s_iy <= 1.

The columns of the linear program are the entries (y, j) of mu, that is,
the constraints of the dual linear program

    |(beta^T X)_yj - tau_yj| <= lambda_yj,

where beta are the dual variables of the constraints with the scores. The
column generation adds the columns whose dual constraint is violated.
'''

# Kept for the modules that import this module with `*`.
import itertools as it
import scipy.special as scs
import time
import random

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def mrc_cg(X, tau_mat, lambda_mat, I, n_max, k_max, warm_start, eps):
	"""
	Column generation algorithm for 0-1 loss Minimax Risk Classifiers.

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	I : `list` of (`int`, `int`)
		Columns (y, j) of mu in the initial model.

	n_max : `int`
		Maximum number of columns added in each iteration of the algorithm.

	k_max : `int`
		Maximum number of iterations of the algorithm.

	warm_start : `array`-like of the shape of `tau_mat` or `None`
		Parameters mu as a warm start for the initial problem,
		zero outside the columns in I.

	eps : `float`
		Dual constraints' violation threshold.

	Return:
	-------
	mu_ : `array`-like of the shape of `tau_mat`
		Parameters learnt by the algorithm.

	nu_ : `float`
		Parameter learnt by the algorithm.

	mrc_upper : `float`
		Optimized upper bound of the MRC classifier.

	I : `list` of (`int`, `int`)
		Columns (y, j) of mu in the final model.
	"""

	tau_mat = np.atleast_2d(np.asarray(tau_mat, dtype=float))
	lambda_mat = np.atleast_2d(np.asarray(lambda_mat, dtype=float))
	# Plain int pairs, as the columns added in select, so that the columns
	# compare consistently in I, J and the keys of mu_vars.
	I = [(int(y), int(j)) for y, j in I]

#---> Initial optimization
	MRC_model, mu_vars, constrs = mrc_cg_init_model(X,
													tau_mat,
													lambda_mat,
													I,
													warm_start)

#---> ADD THE COLUMNS TO THE MODEL
	J = select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, I, eps, n_max)

	k = 0
	while((len(set(J).difference(set(I))) != 0) and (k < k_max)):

	#---> Solve the new optimization
		MRC_model.optimize()

	#---> ADD THE COLUMNS
		I = J.copy()
		J = select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, I, eps, n_max)

		k = k + 1

#---> GET THE PRIMAL SOLUTION
	# Columns of the last solved model. The columns removed by the last
	# selection are zero, and the columns added by it are not solved yet.
	I = [col for col in I if col in mu_vars]
	mu = np.zeros(tau_mat.shape)
	for col in I:
		mu[col] = mu_vars[col][0].X - mu_vars[col][1].X

	# The variable of the linear program is the negative of the
	# parameter nu of the classifier.
	nu = (-1) * MRC_model.getVarByName("nu").X
	mrc_upper = MRC_model.objVal

	return mu, nu, mrc_upper, I

def mrc_cg_init_model(X, tau_mat, lambda_mat, I, warm_start=None):
	"""
	Builds and solves the linear model of MRC 0-1 loss
	restricted to the columns in I.

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	I : `list` of (`int`, `int`)
		Columns (y, j) of mu in the model.

	warm_start : `array`-like of the shape of `tau_mat`, default=`None`
		Parameters mu as a warm start, zero outside the columns in I.

	Return:
	-------
	MRC_model : A MRC object in GUROBI
		A solved GUROBI model of the MRC 0-1 LP.

	mu_vars : `dict`
		Pair of variables (mu_+, mu_-) of each column (y, j) in the model.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores of each instance and class, whose
		dual variables define the dual constraints of the columns.
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
		# nu - sigma_y * (x_i . mu) >= 0, with sigma = (1, -1)
		rhs = 0
		s = None
	else:
		# s_iy - x_i . mu_y + nu >= 1, s_iy >= 0, sum_y s_iy <= 1
		rhs = 1
		s = MRC_model.addMVar((n, n_score_cols), lb=0, name="s")
	MRC_model.update()

	# Constraints with the scores, created without the terms in mu,
	# that is, no variable selected at this point.
	# The terms -x_i . mu_y are added column by column in add_var, as the
	# columns enter the model, and leave it with them.
	constrs = np.empty((n, n_score_cols), dtype=object)
	for i in range(n):
		for y in range(n_score_cols):
			expr = nu if s is None else nu + s[i, y].item()
			constrs[i, y] = MRC_model.addConstr(expr >= rhs)

	if binary:
		MRC_model.addConstr(nu >= 0.5)
	else:
		MRC_model.addConstr(s.sum(axis=1) <= 1)

	# Add the initial columns.
	mu_vars = {}
	for col in I:
		add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col)
	MRC_model.update()

	# Warm start with a feasible point for the given coefficients.
	if warm_start is not None and len(I) > 0:
		mu_start = np.zeros(tau_mat.shape)
		for col in I:
			mu_start[col] = warm_start[col]
			mu_vars[col][0].PStart = max(mu_start[col], 0)
			mu_vars[col][1].PStart = max(-mu_start[col], 0)

		z = X @ mu_start.T

		if binary:
			nu.PStart = max(np.max(np.abs(z)), 0.5)
		else:
			# nu = 1 + max_i max_S (sum_{y in S} z_iy - 1) / |S|
			cumsum = np.cumsum(-np.sort(-z, axis=1), axis=1)
			nu_start = np.max((cumsum - 1) / np.arange(1, n_score_cols + 1)) + 1
			nu.PStart = nu_start
			s.PStart = np.maximum(z - nu_start + 1, 0)

	# Solve the model
	MRC_model.optimize()

	return MRC_model, mu_vars, constrs

def add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col):
	"""
	Adds the column/feature col = (y, j) to the given GUROBI model of MRC.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The model to be updated.

	mu_vars : `dict`
		Pair of variables (mu_+, mu_-) of each column (y, j) in the model.
		It is updated with the new column.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores of each instance and class.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	col : (`int`, `int`)
		Column (y, j) of mu to be added.
	"""

	y, j = col
	x_j = X[:, j]
	nnz = np.flatnonzero(x_j)

	# Coefficients of mu_+ in the constraints with the scores.
	if tau_mat.shape[0] == 1:
		# Binary: -x_ij for class 0 and +x_ij for class 1.
		coeffs = np.concatenate([-x_j[nnz], x_j[nnz]])
		rows = np.concatenate([constrs[nnz, 0], constrs[nnz, 1]])
	else:
		coeffs = -x_j[nnz]
		rows = constrs[nnz, y]

	rows = rows.tolist()
	mu_plus_i = MRC_model.addVar(lb=0, obj=lambda_mat[col] - tau_mat[col],
								 column=gp.Column(coeffs.tolist(), rows))
	mu_minus_i = MRC_model.addVar(lb=0, obj=lambda_mat[col] + tau_mat[col],
								  column=gp.Column((-coeffs).tolist(), rows))
	mu_vars[col] = (mu_plus_i, mu_minus_i)

def select(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, I, eps, n_max):
	"""
	Updates the MRC model by removing the redundant columns and adding the
	columns whose dual constraint is violated.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		A solved GUROBI model of the MRC 0-1 LP.

	mu_vars : `dict`
		Pair of variables (mu_+, mu_-) of each column (y, j) in the model.

	constrs : `array`-like of shape (`n_samples`, `n_classes`) or (`n_samples`, 2)
		Constraints with the scores of each instance and class.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	I : `list` of (`int`, `int`)
		Columns (y, j) of mu in the model.

	eps : `float`
		Dual constraints' violation threshold.

	n_max : `int`
		Maximum number of columns added.

	Returns:
	--------
	J : `list` of (`int`, `int`)
		Columns (y, j) of mu in the updated model.
	"""

	# Dual solution of the constraints with the scores
	beta = np.asarray(MRC_model.getAttr("Pi", constrs.flatten().tolist())).reshape(constrs.shape)
	if tau_mat.shape[0] == 1:
		# Binary: phi(x, 0) = x and phi(x, 1) = -x
		m = ((beta[:, 0] - beta[:, 1]) @ X)[np.newaxis, :]
	else:
		m = beta.T @ X

	# Violations in the dual constraints of the columns not in the model.
	v = np.maximum((m - tau_mat - lambda_mat), 0.) + np.maximum((tau_mat - lambda_mat - m), 0.)
	not_in_model = np.ones(tau_mat.shape, dtype=bool)
	for col in I:
		not_in_model[col] = False
	I_c = np.argwhere(not_in_model)
	v = v[not_in_model]

	J = I.copy()

	# Remove the redundant columns
	for col in I:
		mu_plus_i, mu_minus_i = mu_vars[col]
		mu = mu_plus_i.X - mu_minus_i.X
		basic_status = not ((mu_plus_i.VBasis == -1) and (mu_minus_i.VBasis == -1))

		if (mu == 0) and (basic_status == False):
			J.remove(col)

			# Remove from the gurobi model
			MRC_model.remove(mu_plus_i)
			MRC_model.remove(mu_minus_i)
			del mu_vars[col]

	# Add the columns
	n_violations = np.sum(v > eps)
	if n_violations <= n_max:
		cols_to_add = I_c[v > eps]
	else:
		cols_to_add = I_c[np.argsort(v)[::-1][:n_max]]

	for y, j in cols_to_add:
		col = (int(y), int(j))
		J.append(col)
		add_var(MRC_model, mu_vars, constrs, X, tau_mat, lambda_mat, col)

	return J
