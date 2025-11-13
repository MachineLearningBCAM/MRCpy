import gurobipy as gp
from gurobipy import GRB
import numpy as np

def select(MRC_model, F, tau_, lambda_, I, alpha, alpha_0, eps, n_max):

	"""
	Function to update existing MRC model by adding new feature (variable).

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

	I_c = list(set(np.arange(F.shape[1])) - set(I))
	J = I.copy()
	m = F.transpose() @ alpha

	# Violations in the constraint.
	v = np.maximum((m[I_c] - (1-alpha_0) * (tau_[I_c] + lambda_[I_c])), 0.) + np.maximum(((1-alpha_0) * (tau_[I_c] - lambda_[I_c]) - m[I_c]), 0.)

	# Remove the redundant features
	for i in I:

		mu = MRC_model.getVarByName("mu_+_" + str(i)).x - MRC_model.getVarByName("mu_-_" + str(i)).x
		basic_status_plus = MRC_model.getVarByName("mu_+_" + str(i)).VBasis
		basic_status_minus = MRC_model.getVarByName("mu_-_" + str(i)).VBasis
		basic_status = True
		if (basic_status_plus == -1) and (basic_status_minus == -1):
			basic_status = False

		if (mu == 0) and (basic_status == False):
			J.remove(i)

			# Remove from the gurobi model
			MRC_model.remove(MRC_model.getVarByName("mu_+_" + str(i)))
			MRC_model.remove(MRC_model.getVarByName("mu_-_" + str(i)))

	# Add the features
	k = 0
	n_violations = np.sum(v > eps)
	if n_violations <= n_max:
		i = 0
		j = 0
		while(i < v.shape[0] and j < n_violations):
			if v[i] > eps:
				J.append(I_c[i])
				j = j + 1
				MRC_model = add_var(MRC_model, F, tau_, lambda_, I_c[i])
				k = k + 1
			i = i + 1

	else:
		I_sorted_ind = np.argsort(v)[::-1]
		for i in range(n_max):
			j = I_sorted_ind[i]
			J.append(I_c[j])
			MRC_model = add_var(MRC_model, F, tau_, lambda_, I_c[j])
			k = k + 1

	return MRC_model, J

def add_var(MRC_model, F, tau_, lambda_, col_ind):
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
	N_constr = F.shape[0]

	# Add to the gurobi model
	mu_plus_i = MRC_model.addVar(obj=(((-1) * (tau_ - lambda_)))[col_ind],
								   column=gp.Column((-1) * F[:, col_ind],
													[MRC_model.getConstrByName("constr_" + str(j)) for j in range(N_constr)]),
								   name='mu_+_' + str(col_ind))
	mu_plus_i.PStart = 0

	mu_minus_i = MRC_model.addVar(obj=(tau_ + lambda_)[col_ind],
									column=gp.Column(F[:, col_ind],
													 [MRC_model.getConstrByName("constr_" + str(j)) for j in range(N_constr)]),
									name='mu_-_' + str(col_ind))
	mu_minus_i.PStart = 0

	return MRC_model

