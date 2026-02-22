import numpy as np
from .ccg_utils import generate_cols, generate_rows, add_constr, add_var
import itertools as it
import scipy.special as scs
import scipy as sp
from .mrc_lp_large_n_m import mrc_lp_large_n_m_model_gurobi

def mrc_ccg_large_n_m_sparse_binary(X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, idx_cols, n_max=400, m_max=400, nu_init=None, mu_init=None, eps_1=1e-2, eps_2=1e-5, dict_nnz={}, max_iters=150):
	"""
	Column and constraint generation algorithm for Minimax Risk Classifiers
	with sparse binary features and large sample sizes.

	This function implements a dual approach that iteratively adds both
	features (columns) and constraints (rows) to solve the MRC optimization
	problem efficiently for high-dimensional sparse data.

	The algorithm alternates between:
	1. Solving the current restricted optimization problem
	2. Identifying violated dual constraints (features to add)
	3. Identifying violated primal constraints (samples to add)
	4. Adding the most violated features and constraints to the model

	This process continues until no significant violations remain or the
	maximum iteration limit is reached.

	Parameters
	----------
	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features (typically 
		scipy.sparse.csr_matrix). Each row represents a training sample
		and each column represents a feature.

	idx_samples_plus_constr : list of int
		Indices of samples with positive class constraints already added
		to the model. This list is modified in-place as new constraints
		are added during the algorithm.

	idx_samples_minus_constr : list of int
		Indices of samples with negative class constraints already added
		to the model. This list is modified in-place as new constraints
		are added during the algorithm.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.
		Used to define the uncertainty set in the MRC formulation.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature.
		Represents the maximum deviation from tau_ allowed in the
		uncertainty set.

	idx_cols : array-like of int
		Initial list of feature indices to include in the optimization.
		These features form the starting working set. Will be converted
		to a list and extended during the algorithm.

	n_max : int, default=400
		Maximum number of constraints (samples) to add per iteration.
		Controls the rate at which the constraint set grows. Larger
		values may speed up convergence but increase memory usage.

	m_max : int, default=400
		Maximum number of features (columns) to add per iteration.
		Controls the rate at which the feature set grows. Larger
		values may speed up convergence but increase computational cost.

	nu_init : float, optional, default=None
		Initial value for the nu parameter (intercept term). If provided,
		used as a warm start for the optimization. If None, the solver
		determines the initial value.

	mu_init : numpy.ndarray, optional, default=None
		Initial values for mu parameters corresponding to features
		in idx_cols. If provided, used as a warm start for the optimization.
		Must have length equal to len(idx_cols).

	eps_1 : float, default=1e-2
		Constraint violation threshold for primal constraints. Constraints
		violated by more than this amount will be added to the model.
		Smaller values lead to more constraints being added and tighter
		solutions.

	eps_2 : float, default=1e-5
		Feature violation threshold for dual constraints. Features with
		dual violations exceeding this amount will be added to the model.
		Smaller values lead to more features being added and potentially
		better solutions.

	dict_nnz : dict, default={}
		Dictionary mapping sample indices (int) to lists of non-zero feature
		indices (list of int) for efficient sparse matrix operations. Keys
		are sample indices, values are lists of feature indices where the
		sample has non-zero values. If empty, will be computed as needed.

	max_iters : int, default=150
		Maximum number of column/constraint generation iterations. The
		algorithm terminates when either no violations remain or this
		limit is reached.

	Returns
	-------
	mu : numpy.ndarray of shape (len(idx_cols),)
		Learned feature coefficients for the selected features. These
		are the optimal weights for features in the final working set.

	nu : float
		Learned intercept parameter. This is the bias term in the
		linear classifier.

	R : float
		Final objective value representing the optimized upper bound
		on the worst-case error probability.

	R_k : list of float
		Objective values at each iteration, tracking convergence.
		The length equals the number of iterations performed plus one
		(for the initial solution).

	idx_samples_plus_constr : list of int
		Final list of sample indices with positive class constraints.
		This is the input list extended with newly added constraints.

	idx_samples_minus_constr : list of int
		Final list of sample indices with negative class constraints.
		This is the input list extended with newly added constraints.

	idx_cols : list of int
		Final list of selected feature indices. This is the input list
		extended with newly added features.

	Notes
	-----
	The algorithm modifies the input lists `idx_samples_plus_constr`,
	`idx_samples_minus_constr`, and `idx_cols` in-place by extending them
	with newly added constraints and features.

	The stopping criteria are:
	- No features violate dual constraints by more than eps_2, AND
	- No samples violate primal constraints by more than eps_1, OR
	- Maximum iterations (max_iters) is reached

	The algorithm uses Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed.

	Examples
	--------
	>>> import numpy as np
	>>> from scipy.sparse import csr_matrix
	>>> # Create sparse binary feature matrix
	>>> X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
	>>> tau = np.array([0.5, 0.5, 0.5])
	>>> lambda_ = np.array([0.1, 0.1, 0.1])
	>>> idx_cols = [0, 1]  # Start with first two features
	>>> idx_plus = [0]  # Initial positive constraint
	>>> idx_minus = [1]  # Initial negative constraint
	>>> dict_nnz = {0: [0, 2], 1: [1, 2], 2: [0, 1]}
	>>> mu, nu, R, R_k, idx_plus, idx_minus, idx_cols = \\
	...     mrc_ccg_large_n_m_sparse_binary(
	...         X, idx_plus, idx_minus, tau, lambda_, idx_cols,
	...         dict_nnz=dict_nnz, max_iters=50
	...     )
	"""
	# Initialization
	R_k = []
	idx_cols = idx_cols.tolist()

	# Indices of variables not selected
	not_idx_cols = list(set(np.arange(tau_.shape[0])) - set(idx_cols))

	# Solve the initial optimization.
	MRC_model = mrc_lp_large_n_m_model_gurobi(X,
											  idx_samples_plus_constr,
											  idx_samples_minus_constr,
											  tau_,
											  lambda_,
											  idx_cols,
											  nu_init,
											  mu_init,
											  is_sparse,
											  dict_nnz)

	R_k.append(MRC_model.objVal)

	# Obtain the primal solutions to generate the constraints.
	mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
	mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
	nu_pos = MRC_model.getVarByName("nu_+")
	nu_neg = MRC_model.getVarByName("nu_-")

	# Obtain the dual solutions to generate features.
	alpha_0 = MRC_model.getConstrByName("constr_+").Pi
	alpha_pos = np.asarray([MRC_model.getConstrByName("constr_+_" + str(i)).Pi for i in idx_samples_plus_constr])
	alpha_neg = np.asarray([MRC_model.getConstrByName("constr_-_" + str(i)).Pi for i in idx_samples_minus_constr])
	mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
	nu = nu_pos.x - nu_neg.x

	# Generate the features to be added to the model.
	cols_to_add, n_features_generated = generate_cols(X,
													  idx_samples_plus_constr,
													  idx_samples_minus_constr,
													  tau_,
													  lambda_,
													  alpha_pos,
													  alpha_neg,
													  alpha_0,
													  not_idx_cols,
													  m_max,
													  eps_2)

	# Generate the constraints to be added to the model.
	rows_to_add_plus_constr, rows_to_add_minus_constr, n_constr_generated = generate_rows(X,
																						idx_samples_plus_constr,
																						idx_samples_minus_constr,
																						idx_cols,
																						mu,
																						nu,
																						n_max,
																						eps_1)

	# Add the constraints to the model
	if n_constr_generated > 0:
		MRC_model = add_constr(MRC_model,
							   X,
							   rows_to_add_plus_constr,
							   rows_to_add_minus_constr,
							   idx_cols,
							   mu_plus,
							   mu_minus,
							   nu_pos,
							   nu_neg,
							   dict_nnz)

		# Update the sample indices
		idx_samples_plus_constr.extend(rows_to_add_plus_constr)
		idx_samples_minus_constr.extend(rows_to_add_minus_constr)
		MRC_model.update()

	# Add variables to the model
	if n_features_generated > 0:
		MRC_model = add_var(MRC_model,
							X,
							idx_samples_plus_constr,
							idx_samples_minus_constr,
							tau_,
							lambda_,
							cols_to_add)

		# Add and remove column indices from the current set		
		idx_cols.extend(cols_to_add)

		# Add and remove column indices from the set to be checked
		for i in cols_to_add:
			not_idx_cols.remove(i)

		MRC_model.update()

	k = 0
	while(n_features_generated + n_constr_generated > 0 and k < max_iters):

		# Solve the updated optimization and get the dual solution.
		MRC_model.optimize()

		R_k.append(MRC_model.objVal)
		
		# Obtain the primal solutions to generate the constraints.
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")

		# Obtain the dual solutions to generate the features.
		alpha_0 = MRC_model.getConstrByName("constr_+").Pi
		alpha_pos = np.asarray([MRC_model.getConstrByName("constr_+_" + str(i)).Pi for i in idx_samples_plus_constr])
		alpha_neg = np.asarray([MRC_model.getConstrByName("constr_-_" + str(i)).Pi for i in idx_samples_minus_constr])
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x

		# Generate the features to be added to the model.
		cols_to_add, n_features_generated = generate_cols(X,
														  idx_samples_plus_constr,
														  idx_samples_minus_constr,
														  tau_,
														  lambda_,
														  alpha_pos,
														  alpha_neg,
														  alpha_0,
														  not_idx_cols,
														  m_max,
														  eps_2)

		# Generate the constraints to be added to the model.
		rows_to_add_plus_constr, rows_to_add_minus_constr, n_constr_generated = generate_rows(X,
																							idx_samples_plus_constr,
																							idx_samples_minus_constr,
																							idx_cols,
																							mu,
																							nu,
																							n_max,
																							eps_1)


		# Add the constraints to the model 
		if n_constr_generated > 0:
			# Gurobi model variables
			mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
			mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
			MRC_model = add_constr(MRC_model,
								   X,
								   rows_to_add_plus_constr,
								   rows_to_add_minus_constr,
								   idx_cols,
								   mu_plus,
								   mu_minus,
								   nu_pos,
								   nu_neg,
								   dict_nnz)

			# Update the sample indices
			idx_samples_plus_constr.extend(rows_to_add_plus_constr)
			idx_samples_minus_constr.extend(rows_to_add_minus_constr)
			MRC_model.update()

		# Add the variables to the model
		if n_features_generated > 0:
			MRC_model = add_var(MRC_model,
								X,
								idx_samples_plus_constr,
								idx_samples_minus_constr,
								tau_,
								lambda_,
								cols_to_add)
			idx_cols.extend(cols_to_add)

			# Remove column indices from the set to be checked
			for i in cols_to_add:
				not_idx_cols.remove(i)

			MRC_model.update()

		k = k + 1

	# Obtain the final primal solution.
	if k == max_iters:
		MRC_model.optimize()
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x
		R_k.append(MRC_model.objVal)

	R 			= R_k[-1]

	n_active_constr = 0
	for c in MRC_model.getConstrs():
		if c.Slack < 1e-6:
			n_active_constr = n_active_constr + 1

	return mu, nu, R, R_k, idx_samples_plus_constr, idx_samples_minus_constr, idx_cols
