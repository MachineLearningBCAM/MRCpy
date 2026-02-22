"""
Utility functions for Column and Constraint Generation in large-scale MRC.

This module provides helper functions for the CCG algorithm:
- generate_cols: Identifies features (columns) to add based on dual violations
- generate_rows: Identifies samples (constraints) to add based on primal violations
- add_constr: Adds sample constraints to the Gurobi optimization model
- add_var: Adds feature variables to the Gurobi optimization model
- constr_check_x: Checks constraint violations for a given sample

These functions work together to incrementally build the optimization model
by adding the most violated constraints and features at each iteration.
"""

import gurobipy as gp
import numpy as np
from operator import itemgetter
from gurobipy import *

def generate_cols(F, tau_, lambda_, alpha, not_idx_cols, m_max, eps_2):
	"""
	Generate features (columns) to add to the optimization model based on
	dual constraint violations.

	This function identifies features that violate dual feasibility constraints
	and should be added to the working set to improve the solution. It computes
	the dual constraint violation for each candidate feature and selects the
	most violated ones.

	Parameters
	----------
	F : numpy.ndarray of shape (n_constraints, n_features)
		Current constraint coefficient matrix. Used to compute dual multiplier
		contributions from existing constraints.

	tau_ : numpy.ndarray of shape (n_classes * d,)
		Flattened mean estimates for each feature across each class.

	lambda_ : numpy.ndarray of shape (n_classes * d,)
		Flattened deviation estimates (uncertainty bounds) for each feature.

	alpha : list of float
		Dual variables (Lagrange multipliers) for all constraints. The first
		element alpha[0] is the dual variable for the objective constraint,
		and alpha[1:] are dual variables for sample constraints.

	not_idx_cols : list of int
		Indices of features not currently in the working set. These are the
		candidate features that may be added.

	m_max : int
		Maximum number of features to add. If more than m_max features violate
		the threshold, only the m_max most violated are selected.

	eps_2 : float
		Violation threshold for adding features. Only features with violations
		exceeding this threshold are considered for addition.

	Returns
	-------
	cols_to_add : list of int
		Indices of features to add to the model. These are the features with
		the largest dual constraint violations.

	n_features_added : int
		Number of features actually added. This equals len(cols_to_add) and
		is at most min(m_max, number of violations > eps_2).

	Notes
	-----
	The dual constraint violation is computed as:
	    v_j = max(m_j - (1-alpha_0)(tau_j + lambda_j), 0) +
	          max((1-alpha_0)(tau_j - lambda_j) - m_j, 0)
	where m_j = F[:, j].T @ alpha[1:] is the dual multiplier contribution.

	The function does not modify any input parameters.
	"""
	cols_to_add = []
	if F.shape[0] == 0:
		m = 0
	else:
		m = np.dot((F[:, not_idx_cols]).T, alpha[1:])

	# Violations in the constraint.
	v = np.maximum((m - (1 - alpha[0]) * (tau_[not_idx_cols] + lambda_[not_idx_cols])), 0.) + \
		np.maximum(((1 - alpha[0]) * (tau_[not_idx_cols] - lambda_[not_idx_cols]) - m), 0.)

	# Add the m_max most violated features
	n_features_added = 0
	n_violations = np.sum(v > eps_2)
	if n_violations <= m_max:
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
		sorted_v = list(sorted(enumerate(v), key = itemgetter(1)))[-m_max:]

		# Get the m_max features
		for i in range(m_max):
			j = sorted_v[i][0]
			cols_to_add.append(not_idx_cols[j])
			n_features_added = n_features_added + 1

	return cols_to_add, n_features_added

def generate_rows(X, phi_ob, idx_cols, mu, nu, n_max, eps_1, last_checked, n_constraint_xi):
	"""
	Generate constraints (rows) to add to the optimization model based on
	primal constraint violations.

	This function identifies sample constraints that are violated by the current
	primal solution and should be added to improve feasibility. It iterates
	through samples, evaluates constraint violations using the feature mapping,
	and selects violated constraints to add.

	Parameters
	----------
	X : numpy.ndarray of shape (n_samples, n_features)
		Training feature matrix. Each row represents a training sample.

	phi_ob : BasePhi instance
		Feature mapping object that transforms input data. Must have method
		`eval_x` for evaluating feature mappings and attribute `n_classes`.

	idx_cols : list of int
		Indices of features currently in the working set. Only these features
		are used to compute constraint violations.

	mu : numpy.ndarray of shape (len(idx_cols),)
		Current feature coefficients for features in the working set. These
		are the primal variable values from the current solution.

	nu : float
		Current intercept parameter. This is the bias term in the linear
		classifier.

	n_max : int
		Maximum number of constraints to add. If more than n_max constraints
		are violated, only the first n_max encountered are added.

	eps_1 : float
		Violation threshold for adding constraints. Only constraints violated
		by more than this amount are considered for addition.

	last_checked : int
		Index of the last sample checked in the previous iteration. Used to
		cycle through samples fairly across iterations.

	n_constraint_xi : numpy.ndarray of shape (n_samples,)
		Number of constraints already added for each sample. Used to track
		which samples have had constraints added and avoid adding all possible
		constraints for a single sample.

	Returns
	-------
	F_new : numpy.ndarray of shape (n_added, n_features)
		Coefficient matrix for newly added constraints. Each row represents
		a constraint to be added to the model.

	b_new : numpy.ndarray of shape (n_added,)
		Right-hand side values for newly added constraints.

	count_added : int
		Number of constraints actually added. This equals F_new.shape[0] and
		is at most n_max.

	last_checked : int
		Updated index of the last sample checked. Used in the next iteration
		to continue cycling through samples.

	n_constraint_xi : numpy.ndarray of shape (n_samples,)
		Updated count of constraints added for each sample.

	Notes
	-----
	The function cycles through samples starting from `last_checked` to ensure
	fair coverage across all samples over multiple iterations.

	For each sample, the function evaluates the feature mapping and checks if
	any constraint is violated by more than eps_1. If so, the constraint is
	added to the model.

	The function modifies `n_constraint_xi` in-place to track constraint counts.

	The maximum number of constraints per sample is limited to (2^n_classes - 1)
	to avoid exponential growth.
	"""
	n = X.shape[0]
	F_new = []
	b_new = []
	i = 0

	count_added = 0
	nconstr = 0

	# Add randomly n_max violated constraints
	while (i < n) and (nconstr < n_max):
		sample_index = ((i + last_checked) % n)
		if n_constraint_xi[sample_index] < (2**phi_ob.n_classes) - 1:
			g, c, psi = constr_check_x(phi_ob.eval_x((X[sample_index,:])[np.newaxis, :])[0], mu, nu, idx_cols)
		else:
			i = i + 1
			continue
		if c is not None and (psi + 1 - nu) > eps_1:
			n_constraint_xi[sample_index] = n_constraint_xi[sample_index] + 1
			count_added = count_added + 1
			nconstr = nconstr + 1

			F_new.append(g)
			b_new.append(c)

		i = i + 1

	last_checked = (last_checked + i) % n

	F_new = np.asarray(F_new)
	b_new = np.asarray(b_new)

	return F_new, b_new, count_added, last_checked, n_constraint_xi

def add_constr(MRC_model, F_new, b_new, idx_cols, mu_plus,  mu_minus, nu_pos, nu_neg):
	"""
	Add sample constraints to the MRC primal Gurobi model.

	This function adds constraints corresponding to specific training samples
	to enforce classification requirements. Each constraint ensures that the
	classifier satisfies certain margin requirements for the samples.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi optimization model to update. This model will be modified
		in-place by adding new constraints.

	F_new : numpy.ndarray of shape (n_constraints, n_features)
		Coefficient matrix for new constraints. Each row represents a constraint
		to be added.

	b_new : numpy.ndarray of shape (n_constraints,)
		Right-hand side values for new constraints.

	idx_cols : list of int
		Indices of features currently in the model. These are the features for
		which mu_plus and mu_minus variables exist.

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

	Each constraint has the form:
	    F_new[i] @ (mu_minus - mu_plus) + (nu_pos - nu_neg) >= -b_new[i]

	The function uses variable splitting where mu = mu_plus - mu_minus and
	nu = nu_pos - nu_neg to handle unrestricted variables in the LP framework.
	"""

	nconstr = F_new.shape[0]
	nfeat = len(idx_cols)

	# Add the constraint to the gurobi primal model
	for i in range(nconstr):
		MRC_model.addConstr(gp.quicksum(F_new[i][idx_cols[j]] * (mu_minus[j] - mu_plus[j]) for j in range(nfeat)) +
							(nu_pos - nu_neg) >= (-1) * b_new[i])

	return MRC_model

def add_var(MRC_model, F, tau_, lambda_, cols_to_add):
	"""
	Add feature variables to the MRC primal Gurobi model.

	This function adds new feature variables (columns) to the optimization
	model along with their corresponding coefficients in existing constraints.
	For each feature, two variables are added (mu_plus and mu_minus) to
	represent the positive and negative parts of the coefficient.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi optimization model to update. This model will be modified
		in-place by adding new variables.

	F : numpy.ndarray of shape (n_constraints, n_features)
		Full constraint coefficient matrix. Used to extract coefficients for
		the new features in existing constraints.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature. Used to set objective coefficients
		for the new variables.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. Used to
		set objective coefficients for the new variables.

	cols_to_add : list of int
		Indices of features to add as new variables. For each feature index
		in this list, two variables (mu_plus and mu_minus) will be created.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with new variables added. This is the same object
		as the input, modified in-place.

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

	The function uses Gurobi's column-wise model building, where each variable
	is added along with its coefficients in all existing constraints.
	"""

	constrs = MRC_model.getConstrs()

	for col_ind in cols_to_add:

		mu_plus_i = MRC_model.addVar(lb=0, obj=((-1) * (tau_ - lambda_))[col_ind],
									 column=gp.Column(np.append(-1 * (tau_ - lambda_)[col_ind], (-1) * F[:, col_ind]),
													  constrs),
									 name='mu_+_' + str(col_ind))
		mu_plus_i.PStart = 0

		mu_minus_i = MRC_model.addVar(lb=0, obj=(tau_ + lambda_)[col_ind],
									  column=gp.Column(np.append((tau_ + lambda_)[col_ind], F[:, col_ind]),
													   constrs),
									  name='mu_-_' + str(col_ind))
		mu_minus_i.PStart = 0

	return MRC_model

def constr_check_x(phi_x, mu, nu, idx_cols):
	"""
	Check constraint violations for a given sample's feature mapping.

	This function evaluates whether any constraints are violated for a given
	sample by computing the maximum violation across all possible subsets of
	classes. It implements an efficient algorithm that iterates through class
	subsets in order of decreasing violation.

	Parameters
	----------
	phi_x : numpy.ndarray of shape (n_classes, n_features)
		Feature mapping matrix for a single sample. Each row represents the
		feature vector for one class. This is obtained by applying the feature
		mapping phi to the input sample.

	mu : numpy.ndarray of shape (len(idx_cols),)
		Current feature coefficients for features in the working set. These
		are the primal variable values from the current solution.

	nu : float
		Current intercept parameter. This is the bias term in the linear
		classifier.

	idx_cols : list of int
		Indices of features currently in the working set. Only these features
		are used to compute constraint violations.

	Returns
	-------
	g : numpy.ndarray of shape (n_features,) or None
		Coefficient vector for the most violated constraint. This is the
		average feature vector over the subset of classes that produces the
		maximum violation. Returns None if no constraint is violated.

	c : float or None
		Right-hand side value for the most violated constraint. Computed as
		(1/k - 1) where k is the size of the violating subset. Returns None
		if no constraint is violated.

	psi : float or None
		Maximum violation value. This is the maximum value of the constraint
		function over all subsets. Returns None if no constraint is violated.

	Notes
	-----
	The algorithm iterates through subsets of classes in order of decreasing
	score v[i] = phi_x[i] @ mu. For each subset size k, it computes:
	    psi_k = (sum of top k scores) / k - 1

	The function returns the subset that produces the maximum psi value,
	provided that (psi + 1 - nu) > 0 (indicating a violation).

	If no constraint is violated (psi + 1 - nu <= 0), the function returns
	(None, None, None).

	The function uses a greedy algorithm that runs in O(n_classes) time,
	making it efficient even for problems with many classes.

	Examples
	--------
	>>> phi_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	>>> mu = np.array([0.1, 0.2, 0.3])
	>>> nu = 0.5
	>>> idx_cols = [0, 1, 2]
	>>> g, c, psi = constr_check_x(phi_x, mu, nu, idx_cols)
	"""
	n_classes = phi_x.shape[0]

	v = phi_x[:, idx_cols] @ mu.T

	np.random.seed(42)
	indices = np.argsort(v)[::-1]

	psi = v[indices[0]] - 1

	g = phi_x[indices[0], :]
	c = 0

	if (psi + 2) < nu:
		# No constraints are violated for this instance
		return None, None, None

	# Iterate through all the classes.
	# Each iteration provides the maximum
	# value psi corresponding to the subset
	# of classes of length k.
	for k in range(2, (n_classes + 1)):  # O(|Y|)
		psi_ = ((k - 1) * psi + v[indices[k - 1]]) / k

		if psi_ > psi:
			psi = psi_
			g = ((k - 1) * g + phi_x[indices[k - 1], :]) / k
			c = (1 / k) - 1
		elif (psi + 1 - nu) < 0 or np.isclose((psi + 1 - nu), 0):
			return None, None, None
		else:
			return g, c, psi

	if (psi + 1 - nu) < 0 or np.isclose((psi + 1 - nu), 0):
		return None, None, None
	else:
		return g, c, psi
