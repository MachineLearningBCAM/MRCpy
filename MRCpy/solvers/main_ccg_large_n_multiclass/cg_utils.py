"""
Utility functions for Constraint Generation in multiclass large-sample MRC.

This module provides helper functions for the multiclass constraint generation
algorithm:
- select: Identifies and adds violated constraints while removing redundant ones
- add_var: Adds dual variables (constraints) to the Gurobi optimization model
- constr_check_x: Checks constraint violations for a given sample

These functions work together to incrementally build the dual optimization model
by adding the most violated constraints at each iteration for multiclass problems.
"""

import gurobipy as gp
import numpy as np

def select(MRC_model, X_full, constr_dict, constr_var_dict, n_max, mu, nu, eps, last_checked):
	"""
	Select and add violated constraints while removing redundant ones for
	multiclass problems.

	This function identifies sample constraints that are violated by the current
	primal solution and adds them to the dual model. It also removes over-satisfied
	constraints (those with non-zero slack) to keep the model size manageable.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi dual optimization model to update. This model will be
		modified in-place by adding new variables (dual constraints) and
		removing redundant ones.

	X_full : numpy.ndarray of shape (n_samples, n_features)
		Full feature matrix including both training samples and artificial
		centroid samples. Each row represents a sample.

	constr_dict : dict
		Dictionary mapping sample indices (int) to lists of class subsets
		(list of lists). Each subset represents a constraint that has been
		added for that sample. Will be modified in-place.

	constr_var_dict : dict
		Dictionary mapping sample indices (int) to lists of Gurobi variable
		objects. Tracks which dual variables correspond to which samples for
		efficient constraint removal. Will be modified in-place.

	n_max : int
		Maximum number of constraints to add. If more than n_max constraints
		are violated, only the first n_max encountered are added.

	mu : numpy.ndarray of shape (n_classes, n_features)
		Current feature coefficients from the primal solution. Used to compute
		constraint violations. Each row corresponds to one class.

	nu : float
		Current intercept parameter from the primal solution. Used to compute
		constraint violations.

	eps : float
		Violation threshold for adding constraints. Only constraints violated
		by more than this amount are considered for addition.

	last_checked : int
		Index of the last sample checked in the previous iteration. Used to
		cycle through samples fairly across iterations.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with new variables added and redundant ones
		removed. This is the same object as the input, modified in-place.

	constr_dict : dict
		Updated constraint dictionary after removing redundant constraints
		and adding new ones.

	constr_var_dict : dict
		Updated variable dictionary after removing redundant variables and
		adding new ones.

	nconstr : int
		Number of constraints actually added. This is at most n_max.

	last_checked : int
		Updated index of the last sample checked. Used in the next iteration
		to continue cycling through samples.

	Notes
	-----
	The function performs two main operations:
	1. **Redundancy removal**: Identifies constraints with non-zero slack
	   and removes them from the model and dictionaries.
	2. **Constraint addition**: Cycles through samples starting from
	   `last_checked` and adds violated constraints.

	The function modifies MRC_model, constr_dict, and constr_var_dict in-place.

	After calling this function, you should call MRC_model.update() to
	integrate the changes into the model.

	The slack for a constraint is computed as:
	    slack = |((sum of scores for subset) / |subset|) + 1 - nu|
	where scores are computed as X[i] @ mu[class, :].T
	"""

	N_constr_dual = mu.shape[0] * mu.shape[1]
	n = X_full.shape[0]
	nconstr = 0
	nconstr_removed = 0

	# Removal of over satisfied constraints
	for x_i, subset_arr in constr_dict.items():
		subset_removed = []
		for subset in subset_arr:
			# Evaluate the constraints and check the slack
			slack_primal_constr = np.abs(((np.sum(X_full[int(x_i), :] @ mu[subset, :].T) - 1) / len(subset)) + 1 - nu)

			# If the slack is positive, remove the variable
			if not np.isclose(slack_primal_constr, 0):
				nconstr_removed = nconstr_removed + 1
				subset_removed.append(subset)

		for subset in subset_removed:
			idx = constr_dict[x_i].index(subset)
			MRC_model.remove(constr_var_dict[x_i][idx])
			del constr_var_dict[x_i][idx]
			del constr_dict[x_i][idx]

	indices = (mu != 0)

	i = 0
	# Add randomly n_max violated constraints
	while (i < n) and (nconstr < n_max):

		# Evaluate the max constraint corresponding to a sample
		sample_idx = (i + last_checked) % n
		x = X_full[sample_idx, :]
		subset, psi = constr_check_x(x, mu, nu, indices)

		# If the constraint is violated, add the constraint
		if subset is not None and (psi + 1 - nu) > eps:

			# Add the constraint/dual variable to the model
			MRC_model, dual_var = add_var(MRC_model, x, subset, N_constr_dual)

			# Add the corresponding constraint
			if sample_idx in constr_dict:
				constr_dict[sample_idx].append(subset)
				constr_var_dict[sample_idx].append(dual_var)
			else:
				constr_dict[sample_idx] = [subset]
				constr_var_dict[sample_idx] = [dual_var]

			nconstr = nconstr + 1

		i = i + 1

	last_checked = (last_checked + i) % n

	return MRC_model, constr_dict, constr_var_dict, nconstr, last_checked

def add_var(MRC_model, x, subset, N_constr):
	"""
	Add a dual variable (primal constraint) to the MRC dual LP model for
	multiclass problems.

	This function adds a new dual variable to the Gurobi model, which
	corresponds to adding a new constraint in the primal formulation. The
	variable is added along with its coefficients in all existing dual
	constraints.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi dual optimization model to update. This model will be
		modified in-place by adding a new variable.

	x : numpy.ndarray of shape (n_features,)
		Feature vector for the sample. This is the feature representation
		of the sample for which a constraint is being added.

	subset : list of int
		List of class indices that form the subset for this constraint.
		For example, [0, 2] means this constraint involves classes 0 and 2.
		The constraint enforces that the average score over this subset
		should satisfy certain bounds.

	N_constr : int
		Total number of primal constraints (equals n_features * n_classes).
		Used to determine which dual constraints need to be updated.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with the new variable added. This is the same
		object as the input, modified in-place.

	alpha_i : gurobipy.Var
		The newly added dual variable (Gurobi variable object). This can be
		used to track or remove the variable later.

	Notes
	-----
	This function modifies the MRC_model in-place by adding a variable.
	After calling this function, you should call MRC_model.update() to
	integrate the new variable into the model.

	The new dual variable alpha_i is added with:
	- Objective coefficient: -((1/|subset|) - 1) for maximization in dual
	- Column coefficients: [-F_i, F_i, 1] corresponding to constraints for
	  mu_plus, mu_minus, and nu respectively
	- Lower bound: 0 (dual variables are non-negative)
	- Initial value: 0 (PStart for warm starting)

	The feature vector is distributed across classes in the subset:
	    F_i[y, :] = x / |subset| for y in subset
	    F_i[y, :] = 0 for y not in subset

	The function uses Gurobi's column-wise model building, where the variable
	is added along with its coefficients in all existing constraints.
	"""

	d = x.shape[0]
	n_classes = int(N_constr / d)

	F_i = np.zeros((n_classes, d))
	for y_i in subset:
		F_i[y_i, :] = x / len(subset)

	F_i = F_i.reshape((n_classes * d,))
	constr_coeff = np.append(-F_i, F_i)
	constr_coeff = np.append(constr_coeff, 1)

	# Obtain all the constraints in the current dual model 
	# to be updated with the addition of new variable
	constr_plus_list = []
	constr_minus_list = []

	# Constraints corresponding to mu+ and mu-
	for j in range(N_constr):
		constr_plus_list.append(MRC_model.getConstrByName("constr_+_" + str(j)))
		constr_minus_list.append(MRC_model.getConstrByName("constr_-_" + str(j)))

	constr_plus_list.extend(constr_minus_list)

	# Constraint corresponding to nu
	constr_plus_list.append(MRC_model.getConstrByName("constr_="))

	# Add to the gurobi model
	alpha_i = MRC_model.addVar(obj=(-1) * ((1 / len(subset)) - 1),
								   column=gp.Column(constr_coeff,
													constr_plus_list))
	alpha_i.PStart = 0

	return MRC_model, alpha_i

def constr_check_x(x, mu, nu, inds):
	"""
	Check constraint violations for a given sample in multiclass problems.

	This function evaluates whether any constraints are violated for a given
	sample by computing the maximum violation across all possible subsets of
	classes. It implements an efficient greedy algorithm that iterates through
	class subsets in order of decreasing score.

	Parameters
	----------
	x : numpy.ndarray of shape (n_features,)
		Feature vector for a single sample. This is the feature representation
		of the sample being checked for constraint violations.

	mu : numpy.ndarray of shape (n_classes, n_features)
		Current feature coefficients from the primal solution. Each row
		corresponds to one class. Used to compute scores for each class.

	nu : float
		Current intercept parameter from the primal solution. This is the bias
		term in the linear classifier.

	inds : tuple of numpy.ndarray
		Tuple of boolean arrays indicating which features are non-zero in mu
		for each class. inds[i] is a boolean array of length n_features
		indicating non-zero features for class i. Used to efficiently compute
		scores by only considering non-zero features.

	Returns
	-------
	subset : list of int or None
		List of class indices that form the most violated constraint. For
		example, [0, 2, 3] means the constraint involving classes 0, 2, and 3
		has the maximum violation. Returns None if no constraint is violated.

	psi : float or None
		Maximum violation value. This is the maximum value of the constraint
		function over all subsets. Returns None if no constraint is violated.

	Notes
	-----
	The algorithm iterates through subsets of classes in order of decreasing
	score v[i] = mu[i, inds[i]] @ x[inds[i]]. For each subset size k, it
	computes:
	    psi_k = (sum of top k scores) / k - 1

	The function returns the subset that produces the maximum psi value,
	provided that (psi + 1 - nu) > 0 (indicating a violation).

	If no constraint is violated (psi + 1 - nu <= 0), the function returns
	(None, None).

	The greedy algorithm runs in O(n_classes) time, making it efficient even
	for problems with many classes.

	The function uses a fixed random seed (42) for reproducibility when
	sorting classes by score.

	Examples
	--------
	>>> x = np.array([1.0, 2.0, 3.0])
	>>> mu = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
	>>> nu = 0.5
	>>> inds = (np.array([True, True, True]), 
	...         np.array([True, True, True]),
	...         np.array([True, True, True]))
	>>> subset, psi = constr_check_x(x, mu, nu, inds)
	"""
	d = x.shape[0]
	n_classes = mu.shape[0]

	v = []
	for i, row in enumerate(inds):
		v.append(mu[i, row] @ x[row])

	# v = mu @ X_feat # O(n|Y|d)
	np.random.seed(42)
	indices = np.argsort(v)[::-1]

	psi = v[indices[0]] - 1
	subset = [indices[0]]

	if (psi + 2) < nu:
		# No constraints are violated for this instance
		return None, None

	# Iterate through all the classes.
	# Each iteration provides the maximum
	# value psi corresponding to the subset
	# of classes of length k.
	for k in range(2, (n_classes + 1)):  # O(|Y|)
		psi_ = ((k - 1) * psi + v[indices[k - 1]]) / k

		if psi_ > psi:
			psi = psi_
			subset.append(indices[k - 1])
		elif (psi + 1 - nu) < 0 or np.isclose(psi + 1 - nu, 0):
			return None, None
		else:
			return subset, psi

	if (psi + 1 - nu) < 0 or np.isclose(psi + 1 - nu, 0):
		return None, None
	else:
		return subset, psi
