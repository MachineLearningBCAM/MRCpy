import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def mrc_lp_large_n_m_model_gurobi(X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, idx_cols, nu_init=None, warm_start=None, is_sparse=False, dict_nnz={}):
	"""
	Function to build and return the linear model of MRC 0-1 loss using the given
	constraint matrix and objective vector.

	Parameters:
	-----------
	F : array-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : array-like of shape (no_of_constraints)
		Right handside of the constraints.

	index_colums: array-like
		Selects the columns of the constraint matrix and objective vector.

	Return:
	-------
	model : A model object of MOSEK
		A solved MOSEK model of the MRC 0-1 linear model using the given constraints
		and objective.

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

	print('Shape of input matrix: ', X.shape)

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
