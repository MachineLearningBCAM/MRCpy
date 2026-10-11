'''Constraint and column generation for the 0-1 loss MRC linear program with
a large number of instances and features, as described in [1].

The 0-1 MRC is given by the linear program (4) in [1]

    P :  min   -(tau - lambda)^T mu_1 + (tau + lambda)^T mu_2 + nu
       mu_1, mu_2, nu
         s.t.  F (mu_1 - mu_2) - nu 1 <= b,  mu_1, mu_2 >= 0,

with a constraint for each instance x and non-empty subset C of the labels
Y, given by the row F_c = sum_{y in C} Phi(x, y)^T / |C| of F and the
component b_c = 1 / |C| - 1 of b. Then, mu = mu_1 - mu_2.

The feature mapping Phi(x, y) is not built. The parameters mu, the mean
estimates tau and their deviations lambda are matrices of shape
(|Y|, n_features), or (1, n_features) for binary classification, and
Phi(x, y)^T mu = s(y) x^T mu_{r(y)}, where each label y uses the row r(y)
of mu with the sign s(y):

    multiclass: r(y) = y, s(y) = 1,
    binary:     r(y) = 0, s(0) = 1, s(1) = -1  (Phi(x, 1) = -Phi(x, 0)).

Accordingly, F_c is a matrix of the shape of mu, whose row r is the sum of
s(y) x / |C| over the labels y in C that use the row r of mu.

Both the constraints and the features (the entries (r, j) of mu) are
generated (Algorithm 1 in [1]): the primal P_{I,J} over a subset I of
constraints and a subset J of features is solved at each iteration k,

    P_{I,J} :  min   -(tau - lambda)_J^T mu_1 + (tau + lambda)_J^T mu_2 + nu
            mu_1, mu_2, nu
               s.t.  F_{I,J} (mu_1 - mu_2) - nu 1 <= b_I,
                     -(tau - lambda)_J^T mu_1 + (tau + lambda)_J^T mu_2 + nu >= 0,
                     mu_1, mu_2 >= 0,

with the additional constraint of the objective being nonnegative, which
bounds the model for any I and J (Section III-E in [1]), with dual
variable alpha_0. Then:

- CONSTR: the constraint with the maximum violation of each instance,
  obtained with Algorithm 2 in [1], is added to I if its violation exceeds
  eps_1, checking the instances in order and adding at most n_max.
- FEAT: a feature not in J is added if its dual constraint (6) in [1],
  (tau - lambda) (1 - alpha_0) <= F_I^T alpha <= (tau + lambda) (1 - alpha_0),
  is violated by more than eps_2, adding the m_max most violated, where
  alpha are the dual variables of the constraints in I.

The constraints and features are not removed (Section III-C in [1]). The
initial subset I is given by the class centers X_hat in (13) in [1], for
the subsets with a single label: x_hat_y = tau_y / p_y for each label y,
with p the class proportions, for multiclass classification, and tau for
binary classification. For binary classification, the constraint of the
subset of both labels, nu >= 1/2, is the same for all the instances and
it is added.

The initial subset J of features is, for binary classification, the 1000
features with the largest |tau| - lambda, that is, with the largest
violation of their dual constraint in a model without constraints
(alpha = 0), and for multiclass classification, the entries of the 100
features with the smallest standard deviation of tau across the classes,
for all the classes.

The instances can be given as a dense array or a sparse (csr) matrix. For
sparse instances, only their nonzero entries are used to build the
constraints and the features of the model.

.. [1] Bondugula, K., Mazuelas, S., & Pérez, A. (2025).
       Efficient Large-Scale Learning of Minimax Risk Classifiers.
       arXiv preprint arXiv:2511.17626.
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy import sparse


def mrc_ccg_large_and_wide(X, tau_mat, lambda_mat, n_max=400, m_max=400,
						   k_max=150, eps_1=1e-2, eps_2=1e-5, p=None):
	"""
	Constraint and column generation algorithm for 0-1 loss Minimax Risk
	Classifiers with a large number of instances and features (Algorithm 1
	in [1]).

	Parameters:
	-----------
	X : `array`-like or sparse matrix of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	n_max : `int`, default=`400`
		Maximum number of constraints added in each iteration.

	m_max : `int`, default=`400`
		Maximum number of features added in each iteration.

	k_max : `int`, default=`150`
		Maximum number of iterations after solving the initial model.

	eps_1 : `float`, default=`1e-2`
		Violation threshold of the constraints.

	eps_2 : `float`, default=`1e-5`
		Violation threshold of the dual constraints of the features.

	p : `array`-like of shape (`n_classes`,), default=`None`
		Proportions of the classes, in the order of the rows of `tau_mat`,
		for the class centers tau_y / p_y (13) in [1] of the initial model
		for multiclass classification. If `None`, the initial model for
		multiclass classification has no class centers. Not used for binary
		classification, whose class center is tau.

	Return:
	-------
	mu : `array`-like of the shape of `tau_mat`
		Parameters mu = mu_1 - mu_2 learnt by the algorithm.

	nu : `float`
		Parameter nu of the classifier learnt by the algorithm, the negative
		of the variable nu of the linear program P.

	R : `float`
		Worst-case error probability given by the last model.

	R_k : `list`
		Worst-case error probability R^k given by the model at each
		iteration.
	"""

	tau_mat = np.atleast_2d(np.asarray(tau_mat, dtype=float))
	lambda_mat = np.atleast_2d(np.asarray(lambda_mat, dtype=float))

	# Row r(y) of mu and sign s(y) of each label
	if tau_mat.shape[0] == 1:
		rows = np.array([0, 0])
		signs = np.array([1., -1.])
	else:
		rows = np.arange(tau_mat.shape[0])
		signs = np.ones(tau_mat.shape[0])

	# Class centers X_hat (13) in [1]
	if tau_mat.shape[0] == 1:
		X_hat = tau_mat
	elif p is not None and len(p) == tau_mat.shape[0]:
		X_hat = tau_mat / np.asarray(p, dtype=float)[:, np.newaxis]
	else:
		X_hat = np.empty((0, tau_mat.shape[1]))

	# The class centers are checked along with the instances.
	if sparse.issparse(X):
		X = sparse.vstack((sparse.csr_matrix(X_hat), X)).tocsr()
	else:
		X = np.vstack((X_hat, np.asarray(X, dtype=float)))

	model = MRCModel(X, tau_mat, lambda_mat, rows, signs)

	# Initial model: the constraints of the class centers for the subsets
	# with a single label, nu >= 1/2 for binary classification, and the
	# initial features.
	if tau_mat.shape[0] == 1:
		model.add_constraint(0, (0, 1))
	for i in range(X_hat.shape[0]):
		for y in range(rows.shape[0]):
			model.add_constraint(i, (y,))
	initial_features(model)

	R_k = []
	last_checked = 0
	# The initial model is always solved, then at most k_max iterations.
	k_max = max(k_max, 0)
	count_added = 1
	k = 0
	while k <= k_max and count_added > 0:
		model.MRC_model.optimize()
		mu, nu = model.primal_solution()
		R_k.append(model.MRC_model.objVal)

		n_features_added = feat(model, m_max, eps_2)
		n_constraints_added, last_checked = constr(model, mu, nu, n_max,
												   eps_1, last_checked)
		count_added = n_features_added + n_constraints_added
		k = k + 1

	R = R_k[-1]

	# The parameter nu of the classifier is the negative of the variable nu
	# of the linear program.
	return mu, -nu, R, R_k

def initial_features(model):
	"""
	Adds the initial subset J of features to the model.

	For binary classification, the 1000 features with the largest
	|tau| - lambda, that is, with the largest violation of their dual
	constraint in a model without constraints (alpha = 0). For multiclass
	classification, the entries of the 100 features with the smallest
	standard deviation of tau across the classes, for all the classes.

	Parameters:
	-----------
	model : `MRCModel`
		The model with the initial constraints.
	"""

	tau_mat, lambda_mat = model.tau_mat, model.lambda_mat
	if tau_mat.shape[0] == 1:
		score = np.abs(tau_mat) - lambda_mat
		idx = np.argsort(score, axis=None)[::-1][:1000]
		features = zip(*np.unravel_index(idx, tau_mat.shape))
	else:
		cols = np.argsort(np.std(tau_mat, axis=0))[:100]
		features = [(r, j) for r in range(tau_mat.shape[0]) for j in cols]

	model.add_features([(int(r), int(j)) for r, j in features])

class MRCModel:
	"""
	Linear model P_{I,J} of MRC 0-1 loss over the subset I of constraints
	and the subset J of features.

	Parameters:
	-----------
	X : `array`-like or sparse matrix of shape (`n_samples`, `n_features`)
		Features corresponding with the instances, including the class
		centers.

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	rows : `array`-like of shape (`n_classes`,)
		Row r(y) of mu of each label.

	signs : `array`-like of shape (`n_classes`,)
		Sign s(y) of each label.

	Attributes:
	-----------
	The following lists have an element for each constraint c = (x, C) in
	I, aligned by position. They are updated as the constraints are added
	(`add_constraint`).

	instances : `list` of `int`
		Index of the instance x in X.

	coef_rows : `list` of `array`-like of shape (`n_rows`,)
		Coefficient of x in each row of F_c.

	constrs : `list` of `gurobipy.Constr`
		Constraint of the linear model, with dual variable alpha_c.

	The features in J are given by:

	in_J : `array`-like of the shape of `tau_mat`
		Boolean array with the features (r, j) in J.

	mu_vars : `dict`
		Variables (mu_1, mu_2) of each feature (r, j) in J.
	"""

	def __init__(self, X, tau_mat, lambda_mat, rows, signs):
		self.X = X
		self.is_sparse = sparse.issparse(X)
		self.tau_mat = tau_mat
		self.lambda_mat = lambda_mat
		self.rows = rows
		self.signs = signs

		self.instances = []
		self.coef_rows = []
		self.constrs = []
		self.in_J = np.zeros(tau_mat.shape, dtype=bool)
		self.mu_vars = {}

		# Define the MRC 0-1 linear model (primal).
		self.MRC_model = gp.Model("MRC_0_1_primal")
		self.MRC_model.Params.LogToConsole = 0
		self.MRC_model.Params.OutputFlag = 0
		self.MRC_model.setParam('DualReductions', 0)

		self.nu = self.MRC_model.addVar(lb=-GRB.INFINITY, obj=1, name='nu')
		self.MRC_model.update()

		# Objective nonnegative, with dual variable alpha_0. The terms of
		# the variables mu_1 and mu_2 are added with each feature.
		self.constr_alpha_0 = self.MRC_model.addConstr(self.nu >= 0)

	def add_constraint(self, i, C):
		"""
		Adds the constraint of instance i and subset C of the labels,
		nu - F_c (mu_1 - mu_2) >= -b_c = 1 - 1 / |C|, over the features in J.
		"""

		if self.is_sparse:
			# Nonzero columns and values of the row i of the csr matrix
			start, end = self.X.indptr[i], self.X.indptr[i + 1]
			nnz, x_nnz = self.X.indices[start:end], self.X.data[start:end]
		else:
			x = self.X[i]
			nnz = np.flatnonzero(x)
			x_nnz = x[nnz]

		mask = np.zeros(self.rows.shape[0], dtype=bool)
		mask[list(C)] = True
		size = len(C)

		# Coefficient of x in each row of F_c
		coef_rows = np.bincount(self.rows, weights=mask * self.signs,
								minlength=self.tau_mat.shape[0]) / size

		coeffs = []
		mu_vars = []
		for r in np.flatnonzero(coef_rows):
			in_J = self.in_J[r, nnz]
			for j, value in zip(nnz[in_J], coef_rows[r] * x_nnz[in_J]):
				mu_1, mu_2 = self.mu_vars[(r, j)]
				coeffs.extend((-value, value))
				mu_vars.extend((mu_1, mu_2))

		expr = gp.LinExpr(coeffs, mu_vars) + self.nu
		self.constrs.append(self.MRC_model.addLConstr(expr, GRB.GREATER_EQUAL,
													  1 - (1 / size)))
		self.instances.append(i)
		self.coef_rows.append(coef_rows)

	def add_features(self, features):
		"""
		Adds the features (r, j) to J, as the variables mu_1 and mu_2 with
		their coefficients in the constraints in I and in the constraint of
		the objective being nonnegative.
		"""

		# Instances and coefficients of the rows of the constraints in I
		X_I = self.X[self.instances]
		if self.is_sparse:
			X_I = X_I.tocsc()
		coef_rows = np.asarray(self.coef_rows).reshape(-1, self.tau_mat.shape[0])

		for r, j in features:
			if self.is_sparse:
				x_j = np.asarray(X_I[:, j].todense()).ravel()
			else:
				x_j = X_I[:, j]
			values = coef_rows[:, r] * x_j
			nnz = np.flatnonzero(values)

			constrs = [self.constrs[c] for c in nnz] + [self.constr_alpha_0]
			lower = self.lambda_mat[r, j] - self.tau_mat[r, j]
			upper = self.lambda_mat[r, j] + self.tau_mat[r, j]
			mu_1 = self.MRC_model.addVar(lb=0, obj=lower,
										 column=gp.Column((-values[nnz]).tolist() + [lower],
														  constrs))
			mu_2 = self.MRC_model.addVar(lb=0, obj=upper,
										 column=gp.Column(values[nnz].tolist() + [upper],
														  constrs))
			self.mu_vars[(r, j)] = (mu_1, mu_2)
			self.in_J[r, j] = True

	def primal_solution(self):
		"""
		Solution mu = mu_1 - mu_2 and nu of the solved model P_{I,J}.
		"""

		mu = np.zeros(self.tau_mat.shape)
		for (r, j), (mu_1, mu_2) in self.mu_vars.items():
			mu[r, j] = mu_1.X - mu_2.X

		return mu, self.nu.X

	def dual_constraints(self):
		"""
		F_I^T alpha and alpha_0 of the dual solution of the solved model.
		"""

		alpha = np.asarray(self.MRC_model.getAttr("Pi", self.constrs))
		alpha_0 = self.constr_alpha_0.Pi

		# F_I^T alpha = sum_c alpha_c F_c, with the row r of F_c given by
		# the coefficient of x in that row times x.
		W = alpha[:, np.newaxis] * np.asarray(self.coef_rows).reshape(-1, self.tau_mat.shape[0])
		X_I = self.X[self.instances]
		F_alpha = np.asarray((X_I.T @ W).T)

		return F_alpha, alpha_0

def feat(model, m_max, eps_2):
	"""
	Adds to J the features whose dual constraint is violated by more than
	eps_2, at most m_max, the most violated ones if there are more
	(function FEAT of Algorithm 1 in [1]).

	Parameters:
	-----------
	model : `MRCModel`
		The solved model.

	m_max : `int`
		Maximum number of features added.

	eps_2 : `float`
		Violation threshold of the dual constraints of the features.

	Return:
	-------
	n_features_added : `int`
		Number of features added.
	"""

	F_alpha, alpha_0 = model.dual_constraints()
	upper = (model.tau_mat + model.lambda_mat) * (1 - alpha_0)
	lower = (model.tau_mat - model.lambda_mat) * (1 - alpha_0)

	# Violations of the dual constraints of the features not in J
	v = np.maximum(F_alpha - upper, 0.) + np.maximum(lower - F_alpha, 0.)
	not_in_J = np.argwhere(~model.in_J)
	v = v[~model.in_J]

	n_violations = np.sum(v > eps_2)
	if n_violations <= m_max:
		features = not_in_J[v > eps_2]
	else:
		features = not_in_J[np.argsort(v)[::-1][:m_max]]

	model.add_features([(int(r), int(j)) for r, j in features])

	return features.shape[0]

def constr(model, mu, nu, n_max, eps_1, last_checked):
	"""
	Adds to I the constraint with the maximum violation of the instances,
	if its violation exceeds eps_1, checking the instances in order from
	last_checked and adding at most n_max constraints (function CONSTR of
	Algorithm 1 in [1]).

	The constraint with the maximum violation of an instance x corresponds
	to the subset C achieving the maximum value (9) in [1],

		psi = max_C (sum_{y in C} Phi(x, y)^T mu - 1) / |C|,

	obtained as in Algorithm 2 in [1] by sorting Phi(x, y)^T mu in
	decreasing order, and its violation is psi + 1 - nu.

	Parameters:
	-----------
	model : `MRCModel`
		The solved model.

	mu : `array`-like
		Parameters mu.

	nu : `float`
		Variable nu of the linear program.

	n_max : `int`
		Maximum number of constraints added.

	eps_1 : `float`
		Violation threshold of the constraints.

	last_checked : `int`
		Instance from which the instances are checked.

	Return:
	-------
	count_added : `int`
		Number of constraints added.

	last_checked : `int`
		Instance from which the instances are checked in the next selection.
	"""

	X = model.X
	n = X.shape[0]

	# Phi(x, y)^T mu for all the instances and labels, sorted in decreasing
	# order. For a fixed size of C, the maximum is given by the labels with
	# the largest values.
	scores = np.asarray(X @ mu.T)[:, model.rows] * model.signs
	order = np.argsort(-scores, axis=1, kind='stable')
	sorted_scores = np.take_along_axis(scores, order, axis=1)
	psi_size = (np.cumsum(sorted_scores, axis=1) - 1) / np.arange(1, scores.shape[1] + 1)
	size = np.argmax(psi_size, axis=1) + 1
	psi = psi_size[np.arange(n), size - 1]
	violation = psi + 1 - nu

	# Instances in order from last_checked
	checked = (last_checked + np.arange(n)) % n
	violated = np.flatnonzero(violation[checked] > eps_1)[:n_max]

	for q in violated:
		i = checked[q]
		model.add_constraint(i, tuple(order[i, :size[i]].tolist()))

	# With n_max constraints added, the next selection starts after the
	# last instance added. Otherwise, all the instances were checked and it
	# starts from the same instance.
	if n_max > 0 and violated.shape[0] == n_max:
		last_checked = (last_checked + violated[-1] + 1) % n

	return violated.shape[0], last_checked
