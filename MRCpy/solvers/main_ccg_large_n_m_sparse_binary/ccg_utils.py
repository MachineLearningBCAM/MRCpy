"""
Utility functions for Column and Constraint Generation in sparse binary MRC.

This module provides helper functions for the CCG algorithm:
- generate_cols: Identifies features (columns) to add based on dual violations
- generate_rows: Identifies samples (constraints) to add based on primal violations  
- add_constr: Adds sample constraints to the Gurobi optimization model
- add_var: Adds feature variables to the Gurobi optimization model

These functions work together to incrementally build the optimization model
by adding the most violated constraints and features at each iteration.
"""

import gurobipy as gp
import numpy as np
from operator import itemgetter
from gurobipy import *

def generate_cols(X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, alpha_pos, alpha_neg, alpha_0, not_idx_cols, n_max, eps_2):
	"""
	Generate features (columns) to add to the optimization model based on
	dual constraint violations.

	This function identifies features that violate dual feasibility
	constraints and should be added to the working set to improve
	the solution. It computes the dual constraint violation for each
	candidate feature and selects the most violated ones.

	The dual constraint for feature j is:
	    |m_j - (1 - alpha_0)(tau_j ± lambda_j)| <= 0
	where m_j is the dual multiplier contribution from existing constraints.

	Parameters
	----------
	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features.

	idx_samples_plus_constr : list of int
		Indices of samples with positive class constraints in the model.
		These are samples where the classifier should predict positive.

	idx_samples_minus_constr : list of int
		Indices of samples with negative class constraints in the model.
		These are samples where the classifier should predict negative.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature.

	alpha_pos : numpy.ndarray of shape (len(idx_samples_plus_constr),)
		Dual variables (Lagrange multipliers) for positive class constraints.
		Obtained from the current optimization solution.

	alpha_neg : numpy.ndarray of shape (len(idx_samples_minus_constr),)
		Dual variables (Lagrange multipliers) for negative class constraints.
		Obtained from the current optimization solution.

	alpha_0 : float
		Dual variable for the objective constraint. This is the multiplier
		on the constraint that ensures the objective is non-negative.

	not_idx_cols : list of int
		Indices of features not currently in the working set. These are
		the candidate features that may be added.

	n_max : int
		Maximum number of features to add. If more than n_max features
		violate the threshold, only the n_max most violated are selected.

	eps_2 : float
		Violation threshold for adding features. Only features with
		violations exceeding this threshold are considered for addition.

	Returns
	-------
	cols_to_add : list of int
		Indices of features to add to the model. These are the features
		with the largest dual constraint violations.

	n_features_added : int
		Number of features actually added. This equals len(cols_to_add)
		and is at most min(n_max, number of violations > eps_2).

	Notes
	-----
	The function modifies no inputs but returns indices that will be used
	to update the working set in the calling function.

	The dual constraint violation is computed as:
	    v_j = max(m_j - (1-alpha_0)(tau_j + lambda_j), 0) +
	          max((1-alpha_0)(tau_j - lambda_j) - m_j, 0)
	where m_j = sum_i alpha_pos_i * X[i,j] - sum_i alpha_neg_i * X[i,j]
	"""
	cols_to_add = []

	if X.shape[0] == 0:
		m = 0
	else:
		X_t = X[idx_samples_plus_constr].transpose()
		neg_X_t = (-1) * X[idx_samples_minus_constr].transpose()

		m = X_t.dot(alpha_pos)
		m = m + neg_X_t.dot(alpha_neg)

	# Violations in the constraint.
	v = np.maximum((m[not_idx_cols] - (1 - alpha_0) * (tau_[not_idx_cols] + lambda_[not_idx_cols])), 0.) + \
		np.maximum(((1 - alpha_0) * (tau_[not_idx_cols] - lambda_[not_idx_cols]) - m[not_idx_cols]), 0.)

	# Add the n_max most violated features
	n_features_added = 0
	n_violations = np.sum(v > eps_2)
	if n_violations <= n_max:
		i = 0
		j = 0
		while (i < v.shape[0] and j < n_violations):
			if v[i] > eps_2:
				j = j + 1
				cols_to_add.append(not_idx_cols[i])
				n_features_added = n_features_added + 1
			i = i + 1

	else:
		# Sorting efficiently in python with O(m)
		sorted_v = list(sorted(enumerate(v), key = itemgetter(1)))[-n_max:]
		for i in range(n_max):
			j = sorted_v[i][0]
			cols_to_add.append(not_idx_cols[j])
			n_features_added = n_features_added + 1

	return cols_to_add, n_features_added

def generate_rows(X, idx_samples_plus_constr, idx_samples_minus_constr, idx_cols, mu, nu, n_max, eps_1):
	"""
	Generate constraints (rows) to add to the optimization model based on
	primal constraint violations.

	This function identifies sample constraints that are violated by the
	current primal solution and should be added to improve feasibility.
	It checks samples not yet in the model or with only one constraint type,
	and adds those with the largest violations.

	For each sample i, the primal constraints are:
	- Positive class: X[i] @ mu - nu >= 0 (sample should be classified positive)
	- Negative class: X[i] @ mu + nu >= 0 (sample should be classified negative)

	Parameters
	----------
	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features.

	idx_samples_plus_constr : list of int
		Indices of samples with positive class constraints already in model.
		These samples have the constraint that they should be classified
		as positive.

	idx_samples_minus_constr : list of int
		Indices of samples with negative class constraints already in model.
		These samples have the constraint that they should be classified
		as negative.

	idx_cols : list of int
		Indices of features currently in the working set. Only these
		features are used to compute constraint violations.

	mu : numpy.ndarray of shape (len(idx_cols),)
		Current feature coefficients for features in the working set.
		These are the primal variable values from the current solution.

	nu : float
		Current intercept parameter. This is the bias term in the
		linear classifier.

	n_max : int
		Maximum number of constraints to add. If more than n_max constraints
		are violated, only the first n_max encountered are added (not
		necessarily the most violated).

	eps_1 : float
		Violation threshold for adding constraints. Only constraints
		violated by more than this amount are considered for addition.
		Larger values lead to fewer constraints being added.

	Returns
	-------
	rows_to_add_plus_constr : list of int
		Sample indices to add as positive class constraints. These are
		samples where X[i] @ mu - nu > eps_1, indicating they should be
		constrained to be classified as positive.

	rows_to_add_minus_constr : list of int
		Sample indices to add as negative class constraints. These are
		samples where X[i] @ mu + nu < -eps_1, indicating they should be
		constrained to be classified as negative.

	count_added : int
		Total number of constraints added. This equals
		len(rows_to_add_plus_constr) + len(rows_to_add_minus_constr)
		and is at most n_max.

	Notes
	-----
	The function considers samples in the following priority:
	1. Samples not yet in the model (neither positive nor negative constraint)
	2. Samples with only one constraint type (only positive or only negative)

	This ensures that samples are progressively added to the model and that
	both constraint types can be added for the same sample if needed.

	The function does not modify any input parameters.
	"""
	i = 0

	count_added = 0
	nconstr = 0

	# Add randomly n_max violated constraints
	# First add the unvisited sample indices
	not_idx_samples = list(set(np.arange(X.shape[0])) - (set(idx_samples_plus_constr + idx_samples_minus_constr)))
	# Now add the sample indices with only one constraint set
	not_idx_samples.extend(list(set(idx_samples_plus_constr + idx_samples_minus_constr) - set(idx_samples_plus_constr).intersection(idx_samples_minus_constr)))

	rows_to_add_plus_constr = []
	rows_to_add_minus_constr = []

	while (i < len(not_idx_samples)) and (nconstr < n_max):
		sample_index = not_idx_samples[i]
		g = X[sample_index, idx_cols]

		g_mu = (g @ mu)[0]

		if (g_mu - nu) > eps_1:
			count_added = count_added + 1
			nconstr = nconstr + 1
			rows_to_add_plus_constr.append(sample_index)
		elif (nu + g_mu) < (-1 * eps_1):
			count_added = count_added + 1
			nconstr = nconstr + 1
			rows_to_add_minus_constr.append(sample_index)
		i = i + 1

	return rows_to_add_plus_constr, rows_to_add_minus_constr, count_added

def add_constr(MRC_model, X, rows_to_add_plus_constr, rows_to_add_minus_constr, idx_cols, mu_plus,  mu_minus, nu_pos, nu_neg, dict_nnz):
	"""
	Add sample constraints to the MRC primal Gurobi model.

	This function adds constraints corresponding to specific training samples
	to enforce classification requirements for positive and negative classes.
	It efficiently handles sparse matrices by only including non-zero features
	in each constraint.

	For each sample i:
	- Positive constraint: -X[i] @ (mu_plus - mu_minus) + (nu_pos - nu_neg) >= 0
	- Negative constraint: X[i] @ (mu_plus - mu_minus) + (nu_pos - nu_neg) >= 0

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi optimization model to update. This model will be
		modified in-place by adding new constraints.

	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features.

	rows_to_add_plus_constr : list of int
		Sample indices to add as positive class constraints. For each
		sample i in this list, a constraint ensuring positive classification
		will be added.

	rows_to_add_minus_constr : list of int
		Sample indices to add as negative class constraints. For each
		sample i in this list, a constraint ensuring negative classification
		will be added.

	idx_cols : list of int
		Indices of features currently in the model. These are the features
		for which mu_plus and mu_minus variables exist.

	mu_plus : numpy.ndarray of Gurobi variables
		Positive part of mu coefficients. Array of Gurobi variable objects
		corresponding to features in idx_cols. Length equals len(idx_cols).

	mu_minus : numpy.ndarray of Gurobi variables
		Negative part of mu coefficients. Array of Gurobi variable objects
		corresponding to features in idx_cols. Length equals len(idx_cols).

	nu_pos : Gurobi variable
		Positive part of nu (intercept). Single Gurobi variable object.

	nu_neg : Gurobi variable
		Negative part of nu (intercept). Single Gurobi variable object.

	dict_nnz : dict
		Dictionary mapping sample indices (int) to lists of non-zero feature
		indices (list of int) for efficient sparse operations. Keys are
		sample indices, values are lists of feature indices where the sample
		has non-zero values.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with new constraints added. This is the same
		object as the input, modified in-place.

	Notes
	-----
	This function modifies the MRC_model in-place by adding constraints.
	After calling this function, you should call MRC_model.update() to
	integrate the new constraints into the model.

	The function uses dict_nnz to efficiently identify which features have
	non-zero values for each sample, avoiding unnecessary computation for
	zero-valued features in the sparse matrix.

	Constraint names follow the pattern:
	- "constr_+_<sample_index>" for positive class constraints
	- "constr_-_<sample_index>" for negative class constraints
	"""

	for i in range(len(rows_to_add_plus_constr)):
		# Get the non zero active constraints
		inter_index_columns_nnz_i = list( set(idx_cols) & set(dict_nnz[rows_to_add_plus_constr[i]]) )
		indexes_coeffs            = [idx_cols.index(inter_index) for inter_index in inter_index_columns_nnz_i]

		# Add the constraint
		f1 = []
		for j in range(len(inter_index_columns_nnz_i)):
			f1.append(-1 * X[rows_to_add_plus_constr[i], inter_index_columns_nnz_i[j]] * (mu_plus[indexes_coeffs[j]] - mu_minus[indexes_coeffs[j]]))
		MRC_model.addConstr(gp.quicksum(f1) + (nu_pos - nu_neg) >= 0, name="constr_+_" + str(rows_to_add_plus_constr[i]))

	for i in range(len(rows_to_add_minus_constr)):
		# Get the non zero active constraints
		inter_index_columns_nnz_i = list( set(idx_cols) & set(dict_nnz[rows_to_add_minus_constr[i]]) )
		indexes_coeffs            = [idx_cols.index(inter_index) for inter_index in inter_index_columns_nnz_i]

		# Add the constraint
		f2 = []
		for j in range(len(inter_index_columns_nnz_i)):
			f2.append(X[rows_to_add_minus_constr[i], inter_index_columns_nnz_i[j]] * (mu_plus[indexes_coeffs[j]] - mu_minus[indexes_coeffs[j]]))
		MRC_model.addConstr(gp.quicksum(f2) + (nu_pos - nu_neg) >= 0, name="constr_-_" + str(rows_to_add_minus_constr[i]))

	return MRC_model

def add_var(MRC_model, X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, cols_to_add):
	"""
	Add feature variables to the MRC primal Gurobi model.

	This function adds new feature variables (columns) to the optimization
	model along with their corresponding coefficients in existing constraints.
	For each feature, two variables are added (mu_plus and mu_minus) to
	represent the positive and negative parts of the coefficient.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi optimization model to update. This model will be
		modified in-place by adding new variables.

	X : scipy.sparse matrix of shape (n_samples, n_features)
		Sparse feature matrix containing binary features.

	idx_samples_plus_constr : list of int
		Indices of samples with positive class constraints in the model.
		These constraints will be updated with coefficients for the new
		features.

	idx_samples_minus_constr : list of int
		Indices of samples with negative class constraints in the model.
		These constraints will be updated with coefficients for the new
		features.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature. Used to set objective coefficients
		and constraint coefficients for the new variables.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. Used to
		set objective coefficients and constraint coefficients for the new
		variables.

	cols_to_add : list of int
		Indices of features to add as new variables. For each feature index
		in this list, two variables (mu_plus and mu_minus) will be created.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with new variables added. This is the same
		object as the input, modified in-place.

	Notes
	-----
	This function modifies the MRC_model in-place by adding variables.
	After calling this function, you should call MRC_model.update() to
	integrate the new variables into the model.

	For each feature j in cols_to_add, two variables are created:
	- mu_plus_j: Positive part of the coefficient, with objective coefficient
	  -(tau_j - lambda_j)
	- mu_minus_j: Negative part of the coefficient, with objective coefficient
	  (tau_j + lambda_j)

	The actual coefficient is mu_j = mu_plus_j - mu_minus_j.

	Variable names follow the pattern:
	- "mu_+_<feature_index>" for positive parts
	- "mu_-_<feature_index>" for negative parts

	The variables are initialized with PStart=0 for warm starting.
	"""
	constrs = [MRC_model.getConstrByName("constr_+")]
	constrs.extend([MRC_model.getConstrByName("constr_+_" + str(i)) for i in idx_samples_plus_constr])
	constrs.extend([MRC_model.getConstrByName("constr_-_" + str(i)) for i in idx_samples_minus_constr])

	for col_ind in cols_to_add:


		F_column = np.zeros(len(idx_samples_plus_constr) + len(idx_samples_minus_constr))
		F_column[:len(idx_samples_plus_constr)] = X[idx_samples_plus_constr, col_ind].toarray()[:,0]
		F_column[len(idx_samples_plus_constr):] = (-1) * X[idx_samples_minus_constr, col_ind].toarray()[:,0]

		mu_plus_i = MRC_model.addVar(lb=0, obj=((-1) * (tau_ - lambda_))[col_ind],
									 column=gp.Column(np.append(-1 * (tau_ - lambda_)[col_ind], (-1) * np.asarray(F_column)),
													  constrs),
									 name='mu_+_' + str(col_ind))
		mu_plus_i.PStart = 0

		mu_minus_i = MRC_model.addVar(lb=0, obj=(tau_ + lambda_)[col_ind],
									  column=gp.Column(np.append((tau_ + lambda_)[col_ind], np.asarray(F_column)),
													   constrs),
									  name='mu_-_' + str(col_ind))
		mu_minus_i.PStart = 0

	return MRC_model
