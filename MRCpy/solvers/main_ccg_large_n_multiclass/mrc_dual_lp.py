import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mrc_dual_lp_model(X, constr_dict, tau_, lambda_, warm_start=None):
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

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_dual")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	# MRC_model.setParam('Method', 0)
	# MRC_model.setParam('LPWarmStart', 1)

	# Define the variable.
	alpha_0 = MRC_model.addVar(lb=0, name='var_0')
	n_classes, d = tau_.shape

	# Define all the dual constraints
	# Compute the tau constraint estimates for each class iterating along the different x and subsets.
	constr = {} # Efficiently compute the dual constraints corresponding to each class in a dictionary
	constr_index = 0
	b = []
	alpha = []
	constr_var_dict = {} # Keep track of the dual variable in order to remove redundant variables along the cg iterations
	for x_i, subset_arr in constr_dict.items():
		for subset in subset_arr:

			# Add the dual variable corresponding to the constraint
			alpha_i = MRC_model.addVar(lb=0)
			# Warm starting mu
			if warm_start is not None:
				alpha_i.PStart = warm_start[constr_index]

			alpha.append(alpha_i)
			for y_i in subset:
				# Store the dual variables corresponding to the constraints in the dictionary
				if x_i in constr_var_dict:
					constr_var_dict[x_i].append(alpha_i)
				else:
					constr_var_dict[x_i] = [alpha_i]
				# Build the constraints
				if y_i in constr:
					constr[y_i].append((X[x_i, :] / len(subset)) * alpha_i)
				else:
					constr[y_i] = [(X[x_i, :] / len(subset)) * alpha_i]
			b.append((1 / len(subset)) - 1)
			constr_index = constr_index + 1

	nconstr_primal = constr_index
	alpha = np.asarray(alpha)
	b = np.asarray(b)

	for y_i in range(n_classes):
		constr_mat_y_i = constr[y_i]
		for i in range(d):
			MRC_model.addConstr(gp.quicksum(constr_mat_y_i[j][i] for j in range(len(constr_mat_y_i))) <= (tau_[y_i, i] + lambda_[y_i, i]) * (1 - alpha_0), "constr_-_" + str(d*y_i + i))
			MRC_model.addConstr((-1) * gp.quicksum(constr_mat_y_i[j][i] for j in range(len(constr_mat_y_i))) <= (- tau_[y_i, i] + lambda_[y_i, i]) * (1 - alpha_0), "constr_+_" + str(d*y_i + i))

	# for i in range(F_transpose.shape[0]):


	# 	MRC_model.addConstr(F_transpose[i, :] @ alpha <= (tau_[i] + lambda_[i]) * (1 - alpha_0), "constr_-_" + str(i))
	# 	MRC_model.addConstr(((-1) * F_transpose[i, :]) @ alpha <= (- tau_[i] + lambda_[i]) * (1 - alpha_0), "constr_+_" + str(i))

	MRC_model.addConstr(np.ones(nconstr_primal).T @ alpha + alpha_0 == 1, "constr_=")

	# Define the objective.
	MRC_model.setObjective(((-1) * b.T) @ alpha, GRB.MAXIMIZE)

	# Solve the model
	MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize()

	return MRC_model, constr_var_dict
