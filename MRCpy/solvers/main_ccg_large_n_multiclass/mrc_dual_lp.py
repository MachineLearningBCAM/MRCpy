"""
Gurobi dual LP model construction for multiclass large-sample MRC.

This module provides functions to build and solve the dual linear programming
formulation of the Minimax Risk Classifier for multiclass problems with large
numbers of samples using the Gurobi optimizer.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mrc_dual_lp_model(X, constr_dict, tau_mat, lambda_mat, warm_start=None):
	"""
	Build and solve the dual linear programming model for multiclass MRC with
	0-1 loss using Gurobi.

	This function constructs the dual formulation of the Minimax Risk Classifier
	for multiclass problems as a linear program and solves it using the Gurobi
	optimizer. The dual formulation is particularly efficient for multiclass
	problems with large numbers of samples.

	The dual optimization problem is:
	    maximize: -b^T @ alpha
	    subject to:
	        - For each feature i and class y: sum_constraints <= (tau[i,y] + lambda[i,y])(1 - alpha_0)
	        - For each feature i and class y: -sum_constraints <= (-tau[i,y] + lambda[i,y])(1 - alpha_0)
	        - Sum constraint: 1^T @ alpha + alpha_0 = 1
	        - All variables >= 0

	Parameters
	----------
	X : numpy.ndarray of shape (n_samples, n_features)
		Feature matrix including both training samples and artificial centroid
		samples. Each row represents a sample.

	constr_dict : dict
		Dictionary mapping sample indices (int) to lists of class subsets (list
		of lists). Each subset represents a constraint. Keys are sample indices,
		values are lists of class subsets.
		Example: {0: [[0], [1]], 1: [[0, 1]]} means sample 0 has constraints
		for classes {0} and {1}, and sample 1 has a constraint for {0, 1}.

	tau_mat : numpy.ndarray of shape (n_classes, d)
		Mean estimates for each feature across each class. tau_mat[y, i]
		represents the mean of feature i for class y.

	lambda_mat : numpy.ndarray of shape (n_classes, d)
		Deviation estimates (uncertainty bounds) for each feature across each
		class. lambda_mat[y, i] represents the uncertainty bound for feature i
		in class y.

	warm_start : numpy.ndarray, optional, default=None
		Initial values for dual variables (alpha). If provided, used as a warm
		start for the optimization. Must have length equal to the total number
		of constraints.

	Returns
	-------
	MRC_model : gurobipy.Model
		Solved Gurobi optimization model. The model contains:
		- Variables: alpha_0 and alpha_i for each constraint
		- Constraints: Dual constraints for each (feature, class) pair and sum constraint
		- Objective: Maximization of -b^T @ alpha
		
		Access solution values using:
		- MRC_model.getConstrByName("constr_+_<index>").Pi for primal mu_plus
		- MRC_model.getConstrByName("constr_-_<index>").Pi for primal mu_minus
		- MRC_model.getConstrByName("constr_=").Pi for primal nu
		- MRC_model.objVal for the optimal objective value

	constr_var_dict : dict
		Dictionary mapping sample indices to lists of Gurobi variable objects.
		Used to track which dual variables correspond to which samples for
		efficient constraint removal in later iterations.

	Notes
	-----
	The dual formulation allows us to recover the primal solution (mu, nu)
	from the dual constraints' Pi values:
	- mu_plus[d*y + i] = Pi of constraint "constr_+_<d*y + i>"
	- mu_minus[d*y + i] = Pi of constraint "constr_-_<d*y + i>"
	- mu[y, i] = mu_plus[d*y + i] - mu_minus[d*y + i]
	- nu = Pi of constraint "constr_="

	where d is the number of features and y is the class index.

	The model is configured with:
	- LogToConsole = 0 (suppress console output)
	- OutputFlag = 0 (suppress all output)
	- DualReductions = 0 (disable dual reductions for stability)

	Constraint naming convention:
	- "constr_+_<d*y + i>" for upper bound on feature i of class y
	- "constr_-_<d*y + i>" for lower bound on feature i of class y
	- "constr_=" for the sum constraint

	The function builds constraints efficiently by grouping contributions
	from each class subset to the corresponding (feature, class) pairs.

	Examples
	--------
	>>> import numpy as np
	>>> X = np.random.randn(10, 50)
	>>> constr_dict = {0: [[0], [1]], 1: [[0, 1]]}
	>>> tau_mat = np.random.randn(2, 50)
	>>> lambda_mat = np.abs(np.random.randn(2, 50))
	>>> model, var_dict = mrc_dual_lp_model(X, constr_dict, tau_mat, lambda_mat)
	>>> print(f"Optimal objective: {model.objVal:.4f}")
	>>> # Recover primal solution
	>>> n_classes, d = tau_mat.shape
	>>> mu_plus = [[model.getConstrByName(f"constr_+_{d*y + i}").Pi 
	...             for i in range(d)] for y in range(n_classes)]
	>>> mu_minus = [[model.getConstrByName(f"constr_-_{d*y + i}").Pi 
	...              for i in range(d)] for y in range(n_classes)]
	>>> mu = np.array(mu_plus) - np.array(mu_minus)
	>>> nu = model.getConstrByName("constr_=").Pi
	"""

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_dual")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	# MRC_model.setParam('Method', 0)
	# MRC_model.setParam('LPWarmStart', 1)

	# Define the variable.
	alpha_0 = MRC_model.addVar(lb=0, name='var_0')
	n_classes, d = tau_mat.shape

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
			MRC_model.addConstr(gp.quicksum(constr_mat_y_i[j][i] for j in range(len(constr_mat_y_i))) <= (tau_mat[y_i, i] + lambda_mat[y_i, i]) * (1 - alpha_0), "constr_-_" + str(d*y_i + i))
			MRC_model.addConstr((-1) * gp.quicksum(constr_mat_y_i[j][i] for j in range(len(constr_mat_y_i))) <= (- tau_mat[y_i, i] + lambda_mat[y_i, i]) * (1 - alpha_0), "constr_+_" + str(d*y_i + i))


	MRC_model.addConstr(np.ones(nconstr_primal).T @ alpha + alpha_0 == 1, "constr_=")

	# Define the objective.
	MRC_model.setObjective(((-1) * b.T) @ alpha, GRB.MAXIMIZE)

	# Solve the model
	MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize()

	return MRC_model, constr_var_dict
