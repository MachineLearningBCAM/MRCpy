import gurobipy as gp
import numpy as np
from operator import itemgetter
from gurobipy import *

def generate_cols(F, tau_, lambda_, alpha, not_idx_cols, m_max, eps_2):
	"""
	Return m_max most violated constraints
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
	Return n_max randomly violated constraints of the primal LP
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
	Function to add constraints to the MRC primal gurobi model.
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
	Function to add new variable to the MRC primal gurobi model.
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
