import gurobipy as gp
import numpy as np

def select(MRC_model, phi, F_, b_, n_max, mu, nu, eps, last_checked):

	"""
	Function to randomly select constraints with violation at the current solution.

	Parameters:
	-----------
	MRC_model : A MOSEK model object
		MRC model to be updated.

	F_ : `array`-like
		A vector of coefficients to update the constraint matrix.

	b_ : `array`-like
		Coefficients to update the constraint matrix.

	n_max : `int`, default=`100`
		Maximum number of constraints selected in each iteration of the algorithm.

	mu : solution obtained by the primal to compute the constraints
		 and check violation.

	nu : solution obtained by the primal to used to check the violation.

	eps : `float`, default=`1e-4`
		Constraints' violation threshold. Maximum violation allowed in the constraints.

	last_checked : `int`
		index corresponding to the sample checked for violation in the previous iteration of the algorithm.

	Returns:
	--------
	MRC_model : A MOSEK model
		Updated MRC model object.

	F_new : `array`-like
		Updated coefficients matrix.

	b_new : `array`-like
		Updated coefficients.

	count_added : `int`
		Number of constraints added.

	last_checked : `int`
		Updated index of the last checked sample.
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
	Function to add new variable to the MRC dual LP.

	Parameters:
	-----------
	MRC_model : A MOSEK model object
		MRC model to be updated.

	F_i : `array`-like
		A vector of coefficients corresponding to added
		constraint in the primal
		or variable in the dual.

	b_i : `float`
		Coefficient corresponding to added
		constraint in the primal
		or variable in the dual.

	N_constr : `int`
		Number of constraints in the primal

	Returns:
	--------
	MRC_model : A MOSEK model
		Updated MRC model object.

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
	Efficiently compute the primal constraints corresponding to a sample.

	Parameters
	----------
	phi_x : array`-like of shape (n_samples, n_features)
		A matrix of features vectors for each class of an instance.

	mu : solution obtained by the primal to compute the constraints
		 and check violation.
	
	nu : solution obtained by the primal to used to check the violation.

	inds : set of feature indices corresponding to non-zero value in mu

	Returns
	-------

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
