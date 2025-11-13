import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mrc_dual_lp_model(F, b, tau_, lambda_, warm_start=None):
	"""
	Function to build and return the linear model of MRC 0-1 loss using the given
	constraint matrix and objective vector.

	Parameters:
	-----------
	F : array-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : array-like of shape (no_of_constraints)
		Right handside of the constraints.

	Return:
	-------
	model : A model object of MOSEK
		A solved MOSEK model of the MRC 0-1 linear model using the given constraints
		and objective.

	"""

	n = F.shape[0]

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_dual")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	# MRC_model.setParam('Method', 0)
	# MRC_model.setParam('LPWarmStart', 2)

	# Define the variable.
	alpha = []
	alpha_0 = MRC_model.addVar(lb=0, name='var_0')
	for i in range(n):
		alpha_i = MRC_model.addVar(lb=0)

		# Warm starting mu
		if warm_start is not None:
			alpha_i.PStart = warm_start[i]

		alpha.append(alpha_i)

	MRC_model.update()

	alpha = np.asarray(alpha)

	F_transpose = F.T

	print(F_transpose.shape)
	# Define all the dual constraints
	for i in range(F_transpose.shape[0]):
		MRC_model.addConstr(F_transpose[i, :] @ alpha <= (tau_[i] + lambda_[i]) * (1 - alpha_0), "constr_-_" + str(i))
		MRC_model.addConstr(((-1) * F_transpose[i, :]) @ alpha <= (- tau_[i] + lambda_[i]) * (1 - alpha_0), "constr_+_" + str(i))

	MRC_model.addConstr(np.ones(n).T @ alpha + alpha_0 == 1, "constr_=")

	# Define the objective.
	MRC_model.setObjective(((-1) * b.T) @ alpha, GRB.MAXIMIZE)

	# Solve the model
	# MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize()

	return MRC_model
