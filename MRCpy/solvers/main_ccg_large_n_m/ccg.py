import numpy as np
from .ccg_utils import generate_cols, generate_rows, add_constr, add_var
from .mrc_lp_large_n_m import mrc_lp_large_n_m_model_gurobi
import time

def mrc_ccg_large_n_m(F_, b_, X, phi_ob, tau_, lambda_, idx_cols, n_max=400, m_max=400, nu_init=None, mu_init=None, eps_1=1e-2, eps_2=1e-5, max_iters=150):
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

	m_max : `int`, default=`20`
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

	print('MRC-CCG with n_max = ' + str(n_max) + ', m_max = ' + str(m_max) + ', eps_1 = ' + str(eps_1) + ', eps_2 = ' + str(eps_2))
	
	solver_initTime = time.time()

	# Initialization
	R_k = []
	n_tries = []
	last_checked = 0
	idx_cols = idx_cols.tolist()
	solver_times = []
	solver_times_gurobi = []
	# Indices of variables not selected
	not_idx_cols = list(set(np.arange(tau_.shape[0])) - set(idx_cols))
	# Number of exponential constraints selected for each sample
	n_constraint_xi = np.zeros(X.shape[0])

	initTime = time.time()
	# Solve the initial optimization.
	MRC_model = mrc_lp_large_n_m_model_gurobi(F_,
							 	  			  b_,
							 	  			  tau_,
							      			  lambda_,
											  idx_cols,
							      			  nu_init,
							      			  mu_init)

	initTime = time.time() - initTime
	R_k.append(MRC_model.objVal)
	print('The initial worst-case error probability : ', MRC_model.objVal)

	solver_times.append(time.time() - solver_initTime)

	# Gurobi model variables
	mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
	mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
	nu_pos = MRC_model.getVarByName("nu_+")
	nu_neg = MRC_model.getVarByName("nu_-")

	# Obtain the primal and dual solutions to generate columns and constraints.
	alpha = [constr.Pi for constr in MRC_model.getConstrs()]
	mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
	nu = nu_pos.x - nu_neg.x

	# Generate the columns to be added to the model.
	cols_to_add, n_features_generated = generate_cols(F_,
													  tau_,
													  lambda_,
													  alpha,
													  not_idx_cols,
													  m_max,
													  eps_2)

	# Generate the constraints to be added to the model.
	F_new, b_new, n_constr_generated, n_tries, last_checked, n_constraint_xi = generate_rows(X,
																							 phi_ob,
																							 idx_cols,
																							 mu,
																							 nu,
																							 n_max,
																							 eps_1,
																							 last_checked,
																							 n_constraint_xi)

	if n_constr_generated > 0:
		MRC_model = add_constr(MRC_model, F_new, b_new, idx_cols, mu_plus, mu_minus, nu_pos, nu_neg)

		# Update the coefficient matrix
		if F_.shape[0] == 0:
			F_ = F_new
			b_ = b_new
		else:
			F_ = np.vstack((F_, F_new))
			b_ = np.append(b_, b_new)
		MRC_model.update()

	# Add the cols and rows to the model
	if n_features_generated > 0:
		MRC_model = add_var(MRC_model, F_, tau_, lambda_, cols_to_add)

		# Add and remove column indices from the current set		
		idx_cols.extend(cols_to_add)

		# Add and remove column indices from the set to be checked
		for i in cols_to_add:
			not_idx_cols.remove(i)

		MRC_model.update()

	k = 0
	while(n_features_generated + n_constr_generated > 0 and k < max_iters):

		# Solve the updated optimization and get the dual solution.
		time_solving_1 = time.time()
		MRC_model.optimize()
		solver_times_gurobi.append(time.time() - time_solving_1)
		solver_times.append(time.time() - solver_initTime)

		print('The worst-case error probability at iteration ' + str(k) + ' is ', MRC_model.objVal)
		R_k.append(MRC_model.objVal)
		
		# Gurobi model variables
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")

		# Obtain the primal and dual solutions to generate columns and constraints.
		alpha = [constr.Pi for constr in MRC_model.getConstrs()]
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x

		# Generate the columns to be added to the model.
		cols_to_add, n_features_generated = generate_cols(F_,
														  tau_,
														  lambda_,
														  alpha,
														  not_idx_cols,
														  m_max,
														  eps_2)

		# Generate the constraints to be added to the model.
		F_new, b_new, n_constr_generated, n_tries, last_checked, n_constraint_xi = generate_rows(X,
																								 phi_ob,
																								 idx_cols,
																								 mu,
																								 nu,
																								 n_max,
																								 eps_1,
																								 last_checked,
																								 n_constraint_xi)


		if n_constr_generated > 0:
			# Gurobi model variables
			mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
			mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
			MRC_model = add_constr(MRC_model, F_new, b_new, idx_cols, mu_plus, mu_minus, nu_pos, nu_neg)

			# Update the coefficient matrix
			F_ = np.vstack((F_, F_new))
			b_ = np.append(b_, b_new)
			MRC_model.update()

		# Add the cols and rows to the model
		if n_features_generated > 0:
			MRC_model = add_var(MRC_model, F_, tau_, lambda_, cols_to_add)

			# Add and remove column indices from the current set		
			idx_cols.extend(cols_to_add)

			# Add and remove column indices from the set to be checked
			for i in cols_to_add:
				not_idx_cols.remove(i)

			MRC_model.update()

		k = k + 1

	# Obtain the final primal solution.
	if k == max_iters:
		time_solving_1 = time.time()
		MRC_model.optimize()
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray(
			[mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x
		R_k.append(MRC_model.objVal)
		solver_times_gurobi.append(time.time() - time_solving_1)
		solver_times.append(time.time() - solver_initTime)

	R 			= MRC_model.objVal

	n_active_constr = 0
	for c in MRC_model.getConstrs():
		if c.Slack < 1e-6:
			n_active_constr = n_active_constr + 1

	return mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, initTime
