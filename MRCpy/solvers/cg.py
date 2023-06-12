import itertools as it
import scipy.special as scs
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import random

def mrc_cg(M, b, tau_, lambda_, I, n_max, k_max, warm_start, nu_init, eps):
	"""
	Constraint generation algorithm for Minimax Risk Classifiers.

	Parameters:
	-----------
	M : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : `array`-like of shape (no_of_constraints)
		Right handside of the constraints.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	I : `list`
		List of feature indices corresponding to features in matrix M.
		This is the initialization for the constraint generation method.

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm

	k_max : `int`, default=`20`
		Maximum number of iterations allowed for termination of the algorithm

	warm_start : `list`, default=`None`
		Coefficients corresponding to I as a warm start
		for the initial problem.

	nu_init : `int`, default=`None`
		Coefficient nu corresponding to the warm start (mu)

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	Return:
	-------
	mu_ : `array`-like of shape (`n_features`) or `float`
        Parameters learnt by the algorithm.

    nu_ : `float`
        Parameter learnt by the algorithm.

	mrc_upper : `float`
        Optimized upper bound of the MRC classifier.

	I : `list`
		List of indices of the features selected
	"""

	# Generate the matrices for the linear optimization of 0-1 MRC
	# from the feature mappings.
	N_constr = M.shape[0]

	# Column selection array
	if type(I) != list:
		I = I.tolist()

#---> Initial optimization
	mu = np.zeros(tau_.shape[0])

	MRC_model = mrc_cg_init_model(M,
							 	  b,
							 	  tau_,
							 	  lambda_,
							 	  I,
							 	  nu_init,
							 	  warm_start)

	# Dual solution
	alpha = [(MRC_model.getConstrByName("constr_" + str(i))).Pi for i in range(N_constr)]

	# Primal solution
	mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))).x for i in I])
	mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))).x for i in I])
	nu_pos = MRC_model.getVarByName("nu_+").x
	nu_neg = MRC_model.getVarByName("nu_-").x

#---> ADD THE COLUMNS TO THE MODEL
	MRC_model, J = select(MRC_model, M, tau_, lambda_, I, alpha, eps, n_max)

	k = 0
	while((len(set(J).difference(set(I))) != 0) and (k < k_max)):

	#---> Solve the new optimization and get the dual
		MRC_model.optimize()
		alpha = np.asarray([(MRC_model.getConstrByName("constr_" + str(i))).Pi for i in range(N_constr)])

	#---> ADD THE COLUMNS
		I = J.copy()
		MRC_model, J = select(MRC_model, M, tau_, lambda_, I, alpha, eps, n_max)

		k = k + 1

#---> GET THE PRIMAL SOLUTION
	mu_plus = [(MRC_model.getVarByName("mu_+_" + str(i))).x for i in I]
	mu_minus = [(MRC_model.getVarByName("mu_-_" + str(i))).x for i in I]
	nu_pos = MRC_model.getVarByName("nu_+").x
	nu_neg = MRC_model.getVarByName("nu_-").x
	mu[I] = np.asarray(mu_plus) - np.asarray(mu_minus)
	nu = nu_pos - nu_neg
	mrc_upper = MRC_model.objVal

	return mu, nu, mrc_upper, I

def mrc_cg_init_model(M, b, tau_, lambda_, I=None, nu_init=None, warm_start=None):
	"""
	Function to build and return the linear model of MRC 0-1 loss using the given
	constraint matrix and objective vector.

	Parameters:
	-----------
	M : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : `array`-like of shape (no_of_constraints)
		Right handside of the constraints.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	I : `array`-like, default=`None`
		Selects the columns of the constraint matrix and objective vector.

	warm_start : `list`, default=`None`
		Coefficients corresponding to I as a warm start
		for the initial problem.

	nu_init : `int`, default=`None`
		Coefficient nu corresponding to the warm start (mu)

	Return:
	-------
	MRC_model : A MRC object in GUROBI
		A solved GUROBI model of the MRC 0-1 LP using the given constraints
		and objective.

	"""

	if I is None:
		I = np.arange(M.shape[1])

	# Define the MRC 0-1 linear model (primal).
	MRC_model = gp.Model("MRC_0_1_primal")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	MRC_model.setParam('Method', 0)
	MRC_model.setParam('LPWarmStart', 2)

	# Define the variable.
	mu_plus = []
	mu_minus = []

	for i, index in enumerate(I):
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

	nu_pos = MRC_model.addVar(lb=0, name="nu_+")
	nu_neg = MRC_model.addVar(lb=0, name="nu_-")

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
	for i in range(M.shape[0]):
		MRC_model.addConstr(M[i, I] @ (mu_minus - mu_plus) -
							nu_pos + nu_neg >= b[i], "constr_" + str(i))


	# Define the objective.
	MRC_model.setObjective(tau_[I] @ (mu_minus - mu_plus) +
						   lambda_[I] @ (mu_minus + mu_plus) - 
						   nu_pos + nu_neg, GRB.MINIMIZE)

	# Solve the model
	MRC_model.setParam('DualReductions', 0)
	MRC_model.optimize()

	return MRC_model

def add_var(MRC_model, M, tau_, lambda_, col_ind):
	"""
	Adds feature/column to the given GUROBI model of MRC.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		A solved GUROBI model of the MRC 0-1 LP using the given constraints
		and objective.

	M : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	col_ind : `int`
		Index of the feature/column to be added.

	Returns:
	--------
	MRC_model : A MRC object in GUROBI
		The model updated with the columns
	"""

	N_constr = M.shape[0]

	# Add to the gurobi model
	mu_plus_i = MRC_model.addVar(obj=(((-1) * (tau_ - lambda_)))[col_ind],
								   column=gp.Column((-1) * M[:, col_ind],
												    [MRC_model.getConstrByName("constr_" + str(j)) for j in range(N_constr)]),
								   name='mu_+_' + str(col_ind))
	mu_plus_i.PStart = 0

	mu_minus_i = MRC_model.addVar(obj=(tau_ + lambda_)[col_ind],
									column=gp.Column(M[:, col_ind],
													 [MRC_model.getConstrByName("constr_" + str(j)) for j in range(N_constr)]),
									name='mu_-_' + str(col_ind))
	mu_minus_i.PStart = 0

	return MRC_model

def select(MRC_model, M, tau_, lambda_, I, alpha, eps, n_max):
	"""
	Function to update existing MRC model by adding new feature (variable).

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		A solved GUROBI model of the MRC 0-1 LP using the given constraints
		and objective.

	M : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	I : `list`
		List of feature indices corresponding to features in matrix M.
		This is the initialization for the constraint generation method.

	alpha : `array`-like of shape (no_of_constraints)
		Dual solution.

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm

	Returns:
	--------
	MRC_model : A MOSEK model
		Updated MRC model object.

	J : `list`
		List of features selected by the select function.

	"""

	I_c = list(set(np.arange(M.shape[1])) - set(I))
	J = I.copy()
	m = M.transpose() @ alpha

	# Violations in the constraint.
	v = np.maximum((m[I_c] - tau_[I_c] - lambda_[I_c]), 0.) + np.maximum((tau_[I_c] - lambda_[I_c] - m[I_c]), 0.)

	fetch_time = 0
	# Remove the redundant features
	for i in I:

		mu = MRC_model.getVarByName("mu_+_" + str(i)).x - MRC_model.getVarByName("mu_-_" + str(i)).x
		basic_status_plus = MRC_model.getVarByName("mu_+_" + str(i)).VBasis
		basic_status_minus = MRC_model.getVarByName("mu_-_" + str(i)).VBasis
		basic_status = True
		if (basic_status_plus == -1) and (basic_status_minus == -1):
			basic_status = False

		if (mu == 0) and (basic_status == False):
			J.remove(i)

			# Remove from the gurobi model
			MRC_model.remove(MRC_model.getVarByName("mu_+_" + str(i)))
			MRC_model.remove(MRC_model.getVarByName("mu_-_" + str(i)))

	# Add the features
	n_violations = np.sum(v > eps)
	if n_violations <= n_max:
		i = 0
		j = 0
		while(i < v.shape[0] and j < n_violations):
			if v[i] > eps:
				J.append(I_c[i])
				j = j + 1
				MRC_model = add_var(MRC_model, M, tau_, lambda_, I_c[i])
			i = i + 1

	else:
		I_sorted_ind = np.argsort(v)[::-1]
		for i in range(n_max):
			j = I_sorted_ind[i]
			J.append(I_c[j])
			MRC_model = add_var(MRC_model, M, tau_, lambda_, I_c[j])

	return MRC_model, J


