import numpy as np
from .mrc_lp_gurobi import mrc_lp_model_gurobi
from .cg_utils import select

def alg1(F, b, tau_, lambda_, I, n_max=100, k_max=20, warm_start=None, nu_init=None, eps=1e-4):
	"""
	Constraint generation algorithm for Minimax Risk Classifiers.

	Parameters:
	-----------
	F : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
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
		Coefficients corresponding to features in I as a warm start
		for the initial problem.

	nu_init : `int`, default=`None`
		Coefficient nu corresponding to the warm start (mu)

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	Return:
	-------
	mu : `array`-like of shape (`n_features`) or `float`
		Parameters learnt by the algorithm.

	nu : `float`
		Parameter learnt by the algorithm.

	R : `float`
		Optimized upper bound of the MRC classifier.

	I : `list`
		List of indices of the features selected

	R_k : `list` of shape (no_of_iterations)
		List of worst-case error probabilites
		obtained for the subproblems at each iteration.
	"""

	# Generate the matrices for the linear optimization of 0-1 MRC
	# from the feature mappings.

	print('MRC-CG with n_max = ' + str(n_max) + ', k_max = ' + str(k_max) + ', epsilon = ' + str(eps))
	R_k = []

	N_constr = F.shape[0]

	# Column selection array
	if type(I) != list:
		I = I.tolist()

	# Initial optimization
	mu = np.zeros(tau_.shape[0])

	MRC_model = mrc_lp_model_gurobi(F,
							 		b,
							 		tau_,
							 		lambda_,
							 		I,
							 		nu_init,
							 		warm_start)

	R_k.append(MRC_model.objVal)

	# Dual solution
	alpha = [(MRC_model.getConstrByName("constr_" + str(i))).Pi for i in range(N_constr)]

	print('The initial worst-case error probability : ', MRC_model.objVal)

	mu_plus 	= np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))).x for i in I])
	mu_minus 	= np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))).x for i in I])
	nu_pos 		= MRC_model.getVarByName("nu_+").x
	nu_neg 		= MRC_model.getVarByName("nu_-").x

	# Add the columns to the model.
	MRC_model, J = select(MRC_model,
						  F,
						  tau_ ,
						  lambda_ ,
						  I,
						  alpha,
						  eps,
						  n_max)

	k = 0
	while((len(set(J).difference(set(I))) != 0) and (k < k_max)):

		# Solve the updated optimization and get the dual solution.
		MRC_model.optimize()
		alpha = np.asarray([(MRC_model.getConstrByName("constr_" + str(i))).Pi for i in range(N_constr)])
		
		print('The worst-case error probability at iteration ' + str(k) + ' is ', MRC_model.objVal)
		R_k.append(MRC_model.objVal)

		# Select the columns/features for the next iteration.
		I = J.copy()
		MRC_model, J = select(MRC_model, F, tau_, lambda_, I, alpha, eps, n_max)

		k = k + 1

	# Obtain the final primal solution.
	mu_plus 	= [(MRC_model.getVarByName("mu_+_" + str(i))).x for i in I]
	mu_minus 	= [(MRC_model.getVarByName("mu_-_" + str(i))).x for i in I]
	nu_pos 		= MRC_model.getVarByName("nu_+").x
	nu_neg 		= MRC_model.getVarByName("nu_-").x
	mu[I] 		= np.asarray(mu_plus) - np.asarray(mu_minus)
	nu 			= nu_pos - nu_neg
	R 			= MRC_model.objVal

	print('###### The total number of features selected : ', len(I))

	return mu, nu, R, I, R_k
