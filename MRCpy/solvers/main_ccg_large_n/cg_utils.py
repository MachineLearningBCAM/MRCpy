"""
Utility functions for Constraint Generation in large-sample MRC.

This module provides helper functions for the constraint generation algorithm:
- select: Identifies and adds violated constraints while removing redundant ones
- add_var: Adds dual variables (constraints) to the Gurobi optimization model
- constr_check_x: Checks constraint violations for a given sample

These functions work together to incrementally build the dual optimization model
by adding the most violated constraints at each iteration.
"""

import gurobipy as gp
import numpy as np

def select(MRC_model, phi, F_, b_, n_max, mu, nu, eps, last_checked):
	"""
	Select and add violated constraints while removing redundant ones.

	This function identifies sample constraints that are violated by the current
	primal solution and adds them to the dual model. It also removes redundant
	constraints (those with zero slack) to keep the model size manageable.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi dual optimization model to update. This model will be
		modified in-place by adding new variables (dual constraints) and
		removing redundant ones.

	phi : numpy.ndarray of shape (n_samples, n_classes, n_features)
		Feature mapping matrix for all samples. For each sample, phi[i] is a
		matrix of shape (n_classes, n_features) where each row represents the
		feature vector for one class.

	F_ : numpy.ndarray of shape (n_constraints, n_features)
		Current constraint coefficient matrix. Will be updated with new
		constraints.

	b_ : numpy.ndarray of shape (n_constraints,)
		Current right-hand side values. Will be updated with new constraints.

	n_max : int
		Maximum number of constraints to add. If more than n_max constraints
		are violated, only the first n_max encountered are added.

	mu : numpy.ndarray of shape (n_features,)
		Current feature coefficients from the primal solution. Used to compute
		constraint violations.

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

	F_new : numpy.ndarray of shape (n_remaining, n_features)
		Updated constraint coefficient matrix after removing redundant
		constraints and adding new ones.

	b_new : numpy.ndarray of shape (n_remaining,)
		Updated right-hand side values after removing redundant constraints
		and adding new ones.

	count_added : int
		Number of constraints actually added. This is at most n_max.

	last_checked : int
		Updated index of the last sample checked. Used in the next iteration
		to continue cycling through samples.

	Notes
	-----
	The function performs two main operations:
	1. **Redundancy removal**: Identifies constraints with zero slack
	   (d[j] ≈ 0) and keeps only those, removing over-satisfied constraints.
	2. **Constraint addition**: Cycles through samples starting from
	   `last_checked` and adds violated constraints.

	The function modifies the MRC_model in-place by:
	- Removing redundant dual variables using MRC_model.remove()
	- Adding new dual variables using add_var()

	After calling this function, you should call MRC_model.update() to
	integrate the changes into the model.

	The function handles both binary and multiclass classification by
	reshaping mu appropriately for one-hot encoded features.
	"""

	N_constr_dual = F_.shape[1]
	# Removal of over satisfied constraints
	d = F_ @ mu - b_ - nu * np.ones(b_.shape[0])

	F_new = []
	b_new = []

	# # Add the features
	n = phi.shape[0]
	count_added = 0
	nconstr = 0

	# Remove the redundant constraints
	j = 0
	remove_var_id = []
	while (j < d.shape[0]):
		if np.isclose(d[j], 0):
			F_new.append(F_[j, :])
			b_new.append(b_[j])
		else:

			# Redundant variable
			remove_var_id.append(j)
		j = j + 1

	vars_ = MRC_model.getVars()
	# Remove the variables
	for k in range(len(remove_var_id)):
		MRC_model.remove(vars_[remove_var_id[k] + 1])

	F_new = np.asarray(F_new)
	b_new = np.asarray(b_new)

	n_classes = phi.shape[1]
	if n_classes > 2:
		# Efficient multiplication for one-hot encoding
		mu_reshaped = np.reshape(mu, (n_classes, int(phi.shape[2] / n_classes)))
	else:
		mu_reshaped = mu
	indices = (mu_reshaped != 0)

	i = 0

	# Add randomly n_max violated constraints
	while (i < n) and (nconstr < n_max):
		g, c, psi = constr_check_x(phi[((i + last_checked) % n), :, :], mu_reshaped, nu, indices)
		if c is not None and (psi + 1 - nu) > eps:
			count_added = count_added + 1
			F_new = np.vstack((F_new, g))
			b_new = np.append(b_new, c)
			nconstr = nconstr + 1

			# Add the constraint/dual variable to the model
			MRC_model = add_var(MRC_model, g, c, N_constr_dual)

		i = i + 1

	last_checked = (last_checked + i) % n

	return MRC_model, F_new, b_new, count_added, last_checked

def add_var(MRC_model, F_i, b_i, N_constr):
	"""
	Add a dual variable (primal constraint) to the MRC dual LP model.

	This function adds a new dual variable to the Gurobi model, which
	corresponds to adding a new constraint in the primal formulation. The
	variable is added along with its coefficients in all existing dual
	constraints.

	Parameters
	----------
	MRC_model : gurobipy.Model
		The Gurobi dual optimization model to update. This model will be
		modified in-place by adding a new variable.

	F_i : numpy.ndarray of shape (n_features,)
		Coefficient vector for the new constraint in the primal (or new
		variable in the dual). Represents how the new constraint interacts
		with each feature.

	b_i : float
		Right-hand side value for the new constraint in the primal (or
		objective coefficient for the new variable in the dual).

	N_constr : int
		Number of primal constraints (dual variables for mu). Used to
		determine which dual constraints need to be updated.

	Returns
	-------
	MRC_model : gurobipy.Model
		Updated Gurobi model with the new variable added. This is the same
		object as the input, modified in-place.

	Notes
	-----
	This function modifies the MRC_model in-place by adding a variable.
	After calling this function, you should call MRC_model.update() to
	integrate the new variable into the model.

	The new dual variable alpha_i is added with:
	- Objective coefficient: -b_i (for maximization in dual)
	- Column coefficients: [-F_i, F_i, 1] corresponding to constraints for
	  mu_plus, mu_minus, and nu respectively
	- Lower bound: 0 (dual variables are non-negative)
	- Initial value: 0 (PStart for warm starting)

	The function uses Gurobi's column-wise model building, where the variable
	is added along with its coefficients in all existing constraints.
	"""

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
	alpha_i = MRC_model.addVar(obj=(-1) * b_i,
								   column=gp.Column(constr_coeff,
													constr_plus_list))
	alpha_i.PStart = 0

	return MRC_model

def constr_check_x(phi_x, mu, nu, inds):
	"""
	Efficiently compute and check constraint violations for a given sample.

	This function evaluates whether any constraints are violated for a given
	sample by computing the maximum violation across all possible subsets of
	classes. It implements an efficient greedy algorithm that iterates through
	class subsets in order of decreasing score.

	Parameters
	----------
	phi_x : numpy.ndarray of shape (n_classes, n_features)
		Feature mapping matrix for a single sample. Each row represents the
		feature vector for one class.

	mu : numpy.ndarray
		Current feature coefficients from the primal solution. For binary
		classification, shape is (n_features,). For multiclass with one-hot
		encoding, shape is (n_classes, n_features_per_class).

	nu : float
		Current intercept parameter from the primal solution. This is the bias
		term in the linear classifier.

	inds : numpy.ndarray or tuple
		Indices of non-zero features in mu. For multiclass, this is a tuple
		of arrays indicating which features are non-zero for each class. Used
		to efficiently compute scores by only considering non-zero features.

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
	score v[i]. For each subset size k, it computes:
	    psi_k = (sum of top k scores) / k - 1

	The function returns the subset that produces the maximum psi value,
	provided that (psi + 1 - nu) > 0 (indicating a violation).

	If no constraint is violated (psi + 1 - nu <= 0), the function returns
	(None, None, None).

	The function handles both binary and multiclass classification:
	- Binary: Directly computes v = phi_x @ mu
	- Multiclass: Computes v[i] = mu[i, inds[i]] @ phi_x[0, inds[i]] for
	  each class i

	The greedy algorithm runs in O(n_classes) time, making it efficient even
	for problems with many classes.

	Examples
	--------
	>>> phi_x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	>>> mu = np.array([0.1, 0.2, 0.3])
	>>> nu = 0.5
	>>> inds = np.array([True, True, True])
	>>> g, c, psi = constr_check_x(phi_x, mu, nu, inds)
	"""
	n_classes = phi_x.shape[0]
	d = int(phi_x.shape[1] / n_classes)

	if n_classes > 2:
		v = []
		X_feat = phi_x[0,:d]
		for i, row in enumerate(inds):
			v.append(mu[i, row] @ X_feat[row])
	else:
		v = phi_x[:, inds] @ mu[inds].T

	# v = mu @ X_feat # O(n|Y|d)
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
		elif (psi + 1 - nu) < 0 or np.isclose(psi + 1 - nu, 0):
			return None, None, None
		else:
			return g, c, psi

	if (psi + 1 - nu) < 0 or np.isclose(psi + 1 - nu, 0):
		return None, None, None
	else:
		return g, c, psi
