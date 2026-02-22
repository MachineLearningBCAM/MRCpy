"""
Gurobi dual LP model construction for large-sample MRC.

This module provides functions to build and solve the dual linear programming
formulation of the Minimax Risk Classifier for large-sample problems using
the Gurobi optimizer.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

def mrc_dual_lp_model(F, b, tau_, lambda_, warm_start=None):
	"""
	Build and solve the dual linear programming model for MRC with 0-1 loss using Gurobi.

	This function constructs the dual formulation of the Minimax Risk Classifier
	as a linear program and solves it using the Gurobi optimizer. The dual
	formulation is particularly efficient for problems with large numbers of
	samples but manageable feature dimensions.

	The dual optimization problem is:
	    maximize: -b^T @ alpha
	    subject to:
	        - For each feature j: F[:, j]^T @ alpha <= (tau_j + lambda_j)(1 - alpha_0)
	        - For each feature j: -F[:, j]^T @ alpha <= (-tau_j + lambda_j)(1 - alpha_0)
	        - Sum constraint: 1^T @ alpha + alpha_0 = 1
	        - All variables >= 0

	Parameters
	----------
	F : numpy.ndarray of shape (n_constraints, n_features)
		Constraint coefficient matrix. Each row represents a constraint
		(corresponding to a sample or subset of samples) and each column
		represents a feature.

	b : numpy.ndarray of shape (n_constraints,)
		Right-hand side values for the constraints. These are the constants
		in the primal constraints.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.
		Used in the dual constraints to define the uncertainty set.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. Used in
		the dual constraints to define the uncertainty set.

	warm_start : numpy.ndarray, optional, default=None
		Initial values for dual variables (alpha). If provided, used as a warm
		start for the optimization. Must have length equal to F.shape[0].

	Returns
	-------
	MRC_model : gurobipy.Model
		Solved Gurobi optimization model. The model contains:
		- Variables: alpha_0 and alpha_i for i=1,...,n_constraints
		- Constraints: Dual constraints for each feature and sum constraint
		- Objective: Maximization of -b^T @ alpha
		
		Access solution values using:
		- MRC_model.getConstrByName("constr_+_<index>").Pi for primal mu_plus
		- MRC_model.getConstrByName("constr_-_<index>").Pi for primal mu_minus
		- MRC_model.getConstrByName("constr_=").Pi for primal nu
		- MRC_model.objVal for the optimal objective value

	Notes
	-----
	The dual formulation allows us to recover the primal solution (mu, nu)
	from the dual constraints' Pi values (dual of dual = primal):
	- mu_plus[j] = Pi of constraint "constr_+_j"
	- mu_minus[j] = Pi of constraint "constr_-_j"
	- mu[j] = mu_plus[j] - mu_minus[j]
	- nu = Pi of constraint "constr_="

	The model is configured with:
	- LogToConsole = 0 (suppress console output)
	- OutputFlag = 0 (suppress all output)

	The function prints the shape of the transposed constraint matrix for
	debugging purposes.

	Variable naming convention:
	- "var_0" for alpha_0
	- Unnamed variables for alpha_i (accessed by index)

	Constraint naming convention:
	- "constr_+_<j>" for upper bound on feature j
	- "constr_-_<j>" for lower bound on feature j
	- "constr_=" for the sum constraint

	Examples
	--------
	>>> import numpy as np
	>>> F = np.array([[1, 2, 3], [4, 5, 6]])
	>>> b = np.array([0.5, 0.3])
	>>> tau = np.array([0.5, 0.5, 0.5])
	>>> lambda_ = np.array([0.1, 0.1, 0.1])
	>>> model = mrc_dual_lp_model(F, b, tau, lambda_)
	>>> print(f"Optimal objective: {model.objVal:.4f}")
	>>> # Recover primal solution
	>>> mu_plus = [model.getConstrByName(f"constr_+_{i}").Pi for i in range(3)]
	>>> mu_minus = [model.getConstrByName(f"constr_-_{i}").Pi for i in range(3)]
	>>> mu = np.array(mu_plus) - np.array(mu_minus)
	>>> nu = model.getConstrByName("constr_=").Pi
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
