import gurobipy as gp
import numpy as np

def select(MRC_model, X_full, constr_dict, constr_var_dict, n_max, mu, nu, eps, last_checked):

	"""
	Function to randomly select constraints with violation at the current solution.

	Parameters:
	-----------
	MRC_model : A MOSEK model object
		MRC model to be updated.

	F : `array`-like
		A vector of coefficients to update the constraint matrix.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	I : `list`
		Current list of feature indices corresponding to features in matrix M.

	alpha : `array`-like
		Dual solution corresponding to the current set of features.

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm.

	Returns:
	--------
	MRC_model : A MOSEK model
		Updated MRC model object.

	J : `list`
		Selected list of features.

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
	Function to add new variable to the MRC GUROBI LP model

	Parameters:
	-----------
	MRC_model : A MOSEK model object
		MRC model to be updated.

	F : `array`-like
		A vector of coefficients to update the constraint matrix.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	col_ind : `int`
		Variable index to be added

	Returns:
	--------
	MRC_model : A MOSEK model
		Updated MRC model object.

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
	Parameters
	----------
	phi_x : array`-like of shape (n_samples, n_features)
		A matrix of features vectors for each class of an instance.

	mu : solution obtained by the primal to compute the constraints
		 and check violation.
	
	nu : solution obtained by the primal to used to check the violation.

	Returns
	-------

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
