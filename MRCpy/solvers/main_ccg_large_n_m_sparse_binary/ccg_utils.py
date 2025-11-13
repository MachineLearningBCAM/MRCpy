import gurobipy as gp
import numpy as np
from operator import itemgetter
from gurobipy import *
import time

def generate_cols(X, idx_samples_plus_constr, idx_samples_minus_constr, tau_, lambda_, alpha_pos, alpha_neg, alpha_0, not_idx_cols, n_max, eps_2):
	"""
	Return n_max most violated constraints
	"""
	cols_to_add = []

	time_multiply = time.time()
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
	Return n_max randomly violated constraints of the primal LP
	"""
	i = 0

	count_added = 0
	nconstr = 0

	time_rows = time.time()
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
	Function to add constraints to the MRC primal gurobi model.
	"""

	time_constraint_adding = time.time()

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
	Function to add new variable to the MRC primal gurobi model.
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
