"""
Gurobi LP model construction for sparse binary MRC.

This module provides functions to build and solve the linear programming
formulation of the Minimax Risk Classifier for sparse binary features using
the Gurobi optimizer.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def mrc_lp_large_n_m_model_gurobi(X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, idx_cols, nu_init=None, warm_start=None, dict_nnz={}):
	"""
	Build and solve the linear programming model for MRC with 0-1 loss using Gurobi.

	This function constructs the primal formulation of the Minimax Risk Classifier
	as a linear program and solves it using the Gurobi optimizer. The formulation
	uses variable splitting (mu = mu_plus - mu_minus) to handle unrestricted
	variables in the LP framework.

	The optimization problem is:
	    minimize: sum_j [(tau_j - lambda_j) * mu_minus_j - (tau_j - lambda_j) * mu_plus_j
	                     + (tau_j + lambda_j) * mu_minus_j + (tau_j + lambda_j) * mu_plus_j]
	              + (nu_pos - nu_neg)
	    subject to:
	        - For positive samples i: -X[i] @ (mu_plus - mu_minus) + (nu_pos - nu_neg) >= 0
	        - For negative samples i: X[i] @ (mu_plus - mu_minus) + (nu_pos - nu_neg) >= 0
	        - Objective constraint: tau @ (mu_minus - mu_plus) + lambda @ (mu_minus + mu_plus)
	                                + (nu_pos - nu_neg) >= 0
	        - All variables >= 0

	Parameters
	----------
	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features. Each row represents
		a training sample and each column represents a feature.

	idx_samples_plus_constr : list of int
		Indices of samples with positive class constraints. These are samples
		that should be classified as positive (y=+1).

	idx_samples_minus_constr : list of int
		Indices of samples with negative class constraints. These are samples
		that should be classified as negative (y=-1).

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.
		Used in the objective function and constraints.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. Used in
		the objective function and constraints.

	idx_cols : list of int or None
		Indices of features to include in the model. If None, all features
		are included. This allows solving a restricted problem with a subset
		of features.

	nu_init : float, optional, default=None
		Initial value for the nu parameter (intercept term). If provided,
		used as a warm start for the optimization. The sign determines which
		part (nu_pos or nu_neg) is initialized.

	warm_start : numpy.ndarray, optional, default=None
		Initial values for mu parameters corresponding to features in idx_cols.
		If provided, used as a warm start for the optimization. The sign of
		each value determines which part (mu_plus or mu_minus) is initialized.
		Must have length equal to len(idx_cols).

	dict_nnz : dict, default={}
		Dictionary mapping sample indices (int) to lists of non-zero feature
		indices (list of int) for efficient sparse matrix operations. Keys
		are sample indices, values are lists of feature indices where the
		sample has non-zero values.

	Returns
	-------
	MRC_model : gurobipy.Model
		Solved Gurobi optimization model. The model contains:
		- Variables: mu_+_j, mu_-_j for each feature j in idx_cols, and nu_+, nu_-
		- Constraints: Sample constraints and objective constraint
		- Objective: Minimization of worst-case error bound
		
		Access solution values using:
		- MRC_model.getVarByName("mu_+_<index>").x for feature coefficients
		- MRC_model.objVal for the optimal objective value

	Notes
	-----
	The function uses variable splitting to handle unrestricted variables:
	- mu_j = mu_plus_j - mu_minus_j (actual coefficient)
	- nu = nu_pos - nu_neg (actual intercept)
	
	Both mu_plus_j, mu_minus_j, nu_pos, and nu_neg are constrained to be >= 0.

	The model is configured with:
	- LogToConsole = 0 (suppress console output)
	- OutputFlag = 0 (suppress all output)
	- LPWarmStart = 1 (enable warm starting)
	- TimeLimit = 3600 seconds (1 hour)

	A constraint "constr_nu" ensures nu >= 0.5 to avoid trivial solutions.

	The function efficiently handles sparse matrices by only including non-zero
	feature coefficients in each constraint using dict_nnz.

	Examples
	--------
	>>> import numpy as np
	>>> from scipy.sparse import csr_matrix
	>>> X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
	>>> tau = np.array([0.5, 0.5, 0.5])
	>>> lambda_ = np.array([0.1, 0.1, 0.1])
	>>> idx_cols = [0, 1, 2]
	>>> idx_plus = [0]
	>>> idx_minus = [1]
	>>> dict_nnz = {0: [0, 2], 1: [1, 2], 2: [0, 1]}
	>>> model = mrc_lp_large_n_m_model_gurobi(
	...     X, idx_plus, idx_minus, tau, lambda_, idx_cols, dict_nnz=dict_nnz
	... )
	>>> print(f"Optimal objective: {model.objVal:.4f}")
	"""

	if idx_cols is None:
		idx_cols = np.arange(X.shape[1])

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_primal")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	# MRC_model.setParam('Method', 0)
	MRC_model.setParam('LPWarmStart', 1)
	MRC_model.setParam('TimeLimit', 3600)

	# Define the variable.
	mu_plus = []
	mu_minus = []

	for i, index in enumerate(idx_cols):
		mu_plus_i = MRC_model.addVar(lb=0, name="mu_+_" + str(index))
		mu_minus_i = MRC_model.addVar(lb=0, name="mu_-_" + str(index))

		if warm_start is not None:
			if warm_start[i] < 0:
				mu_minus_i.PStart = (-1) * warm_start[i]
				mu_plus_i.PStart = 0
			else:
				mu_plus_i.PStart = warm_start[i]
				mu_minus_i.PStart = 0

		mu_plus.append(mu_plus_i)
		mu_minus.append(mu_minus_i)

	nu_pos = MRC_model.addVar(name="nu_+")
	nu_neg = MRC_model.addVar(name="nu_-")

	if nu_init is not None:
		if nu_init < 0:
			nu_neg.PStart = (-1) * nu_init
			nu_pos.PStart = 0

		else:
			nu_pos.PStart = nu_init
			nu_neg.PStart = 0

	MRC_model.update()

	mu_plus = np.asarray(mu_plus)
	mu_minus = np.asarray(mu_minus)

	# Define all the constraints.
	nfeat = len(idx_cols)

	# Objective positive constraint to avoid unbounded solutions.
	MRC_model.addConstr(gp.quicksum(tau_[idx_cols[j]] * (mu_minus[j] - mu_plus[j]) for j in range(nfeat))
						+ gp.quicksum(lambda_[idx_cols[j]] * (mu_minus[j] + mu_plus[j]) for j in range(nfeat))
						+ (nu_pos - nu_neg) >= 0, name="constr_+")
	MRC_model.addConstr((nu_pos - nu_neg) >= 0.5, name="constr_nu")

	for i in range(len(idx_samples_plus_constr)):
		# Get the non zero active constraints
		inter_index_columns_nnz_i = list( set(idx_cols) & set(dict_nnz[idx_samples_plus_constr[i]]) )
		indexes_coeffs            = [idx_cols.index(inter_index) for inter_index in inter_index_columns_nnz_i]

		# Add the constraint
		f1 = []
		for j in range(len(inter_index_columns_nnz_i)):
			f1.append(-1 * X[idx_samples_plus_constr[i], inter_index_columns_nnz_i[j]] * (mu_plus[indexes_coeffs[j]] - mu_minus[indexes_coeffs[j]]))

		MRC_model.addConstr(gp.quicksum(f1) + (nu_pos - nu_neg) >= 0, name="constr_+_" + str(idx_samples_plus_constr[i]))

	for i in range(len(idx_samples_minus_constr)):
		# Get the non zero active constraints
		inter_index_columns_nnz_i = list( set(idx_cols) & set(dict_nnz[idx_samples_minus_constr[i]]) )
		indexes_coeffs            = [idx_cols.index(inter_index) for inter_index in inter_index_columns_nnz_i]

		# Add the constraint
		f2 = []
		for j in range(len(inter_index_columns_nnz_i)):
			f2.append(X[idx_samples_minus_constr[i], inter_index_columns_nnz_i[j]] * (
					mu_plus[indexes_coeffs[j]] - mu_minus[indexes_coeffs[j]]))

		MRC_model.addConstr(gp.quicksum(f2) + (nu_pos - nu_neg) >= 0, name="constr_-_" + str(idx_samples_minus_constr[i]))

	MRC_model.setObjective(gp.quicksum(tau_[idx_cols[j]] * (mu_minus[j] - mu_plus[j]) for j in range(nfeat))
						   + gp.quicksum(lambda_[idx_cols[j]] * (mu_minus[j] + mu_plus[j]) for j in range(nfeat))
						   + (nu_pos - nu_neg), GRB.MINIMIZE)

	MRC_model.optimize()

	return MRC_model
