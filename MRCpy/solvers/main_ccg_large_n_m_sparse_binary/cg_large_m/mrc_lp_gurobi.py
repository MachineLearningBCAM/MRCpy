import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random

def mrc_lp_model_gurobi(F, b, tau_, lambda_, index_columns=None, nu_init=None, warm_start=None):
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

	if index_columns is None:
		index_columns = np.arange(F.shape[1])

	time_limit = 3600

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_primal")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	MRC_model.setParam('Method', 0)
	MRC_model.setParam('LPWarmStart', 2)
	MRC_model.setParam('TimeLimit', time_limit)

	# Define the variable.
	mu_plus = []
	mu_minus = []

	for i, index in enumerate(index_columns):
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

	# Objective positive constraint
	MRC_model.addConstr(gp.quicksum(tau_[index_columns[j]] * (mu_minus[j] - mu_plus[j]) for j in range(len(index_columns)))
						+ gp.quicksum(lambda_[index_columns[j]] * (mu_minus[j] + mu_plus[j]) for j in range(len(index_columns)))
						+ (nu_pos - nu_neg) >= 0, "constr_+")

	# Define all the constraints.
	for i in range(F.shape[0]):
		MRC_model.addConstr(F[i, index_columns] @ (mu_minus - mu_plus) +
							nu_pos - nu_neg >= (-1) * b[i], "constr_" + str(i))

	# Define the objective.
	MRC_model.setObjective(tau_[index_columns] @ (mu_minus - mu_plus) +
						   lambda_[index_columns] @ (mu_minus + mu_plus) + 
						   nu_pos - nu_neg, GRB.MINIMIZE)

	# Solve the model
	# MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize()

	return MRC_model
