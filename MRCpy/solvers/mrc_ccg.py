'''Constraint generation, and column generation, for the 0-1 loss MRC linear
program with a large number of instances, and features, as described in
[1].

The 0-1 MRC is given by the linear program (4) in [1]

    P :  min   -(tau - lambda)^T mu_1 + (tau + lambda)^T mu_2 + nu
       mu_1, mu_2, nu
         s.t.  F (mu_1 - mu_2) - nu 1 <= b,  mu_1, mu_2 >= 0,

with a constraint for each instance x and non-empty subset C of the labels
Y, given by the row F_c = sum_{y in C} Phi(x, y)^T / |C| of F and the
component b_c = 1 / |C| - 1 of b. Then, mu = mu_1 - mu_2. Its dual is the
linear program (6) in [1]

    D :  max   -b^T alpha
        alpha
         s.t.  tau - lambda <= F^T alpha <= tau + lambda,
               1^T alpha = 1,  alpha >= 0.

The feature mapping Phi(x, y) is not built. The parameters mu, the mean
estimates tau and their deviations lambda are matrices of shape
(|Y|, n_features), or (1, n_features) for binary classification, and
Phi(x, y)^T mu = s(y) x^T mu_{r(y)}, where each label y uses the row r(y)
of mu with the sign s(y):

    multiclass: r(y) = y, s(y) = 1,
    binary:     r(y) = 0, s(0) = 1, s(1) = -1  (Phi(x, 1) = -Phi(x, 0)).

Accordingly, F_c is a matrix of the shape of mu, whose row r is the sum of
s(y) x / |C| over the labels y in C that use the row r of mu. For
multiclass classification, the row y is x / |C| if y is in C and zero
otherwise. For binary classification, the single row is x, -x and 0 for
C = {0}, {1} and {0, 1}.

The constraints are generated (Algorithm 1 in [1]), and for a large number
of features also the features, that is, the entries (r, j) of mu. At each
iteration k, the model over a subset I of constraints and a subset J of
features,

    P_{I,J} :  min   -(tau - lambda)_J^T mu_1 + (tau + lambda)_J^T mu_2 + nu
            mu_1, mu_2, nu
               s.t.  F_{I,J} (mu_1 - mu_2) - nu 1 <= b_I,
                     -(tau - lambda)_J^T mu_1 + (tau + lambda)_J^T mu_2 + nu >= 0,
                     mu_1, mu_2 >= 0,

is solved, with the additional constraint of the objective being
nonnegative, which bounds the model for any I and J (Section III-E in
[1]), with dual variable alpha_0. Its optimal value is R^k. Then:

- CONSTR: the constraint with the maximum violation of each instance,
  obtained with Algorithm 2 in [1], is added to I if its violation exceeds
  eps_1, checking the instances in order and adding at most n_max.
- FEAT: a feature not in J is added if its dual constraint,
  (tau - lambda) (1 - alpha_0) <= F_I^T alpha <= (tau + lambda) (1 - alpha_0),
  is violated by more than eps_2, adding the m_max most violated, where
  alpha are the dual variables of the constraints in I.

With a large number of instances only (`DualModel`), J has all the
features and the dual D_I of P_{I,J} is solved, in which a constraint c of
the primal is a variable alpha_c that keeps the previous solution feasible
when it is added with alpha_c = 0. The constraints in I that are not tight
are removed once alpha_0 = 0, that is, once the constraints in I bound the
model. With a large number of features (`PrimalModel`), the features are
also generated and the primal P_{I,J} is solved, in which a feature is a
pair of variables mu_1, mu_2 that keep the previous solution feasible when
they are added with value 0. Most iterations add features in this case.
The constraints are not removed (Section III-C in [1]).

The initial subset I is given by the class centers X_hat in (13) in [1],
for the subsets with a single label: x_hat_y = tau_y / p_y for each label
y, with p the class proportions, for multiclass classification, and tau
for binary classification, which is the average of the instances with
Phi(x, 1) = -Phi(x, 0). As averages of the instances, their constraints
are implied by the constraints of the instances. For binary
classification, the constraint of the subset of both labels, nu >= 1/2,
is the same for all the instances and it is added.

With a large number of features, the initial subset J is, for binary
classification, the 1000 features with the largest |tau| - lambda, that
is, with the largest violation of their dual constraint in a model without
constraints (alpha = 0), and for multiclass classification, the entries of
the 100 features with the smallest standard deviation of tau across the
classes, for all the classes.

The instances can be given as a dense array or a sparse (csr) matrix, of
which only the nonzero entries are used to build the model.

.. [1] Bondugula, K., Mazuelas, S., & Pérez, A. (2025).
       Efficient Large-Scale Learning of Minimax Risk Classifiers.
       arXiv preprint arXiv:2511.17626.
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy import sparse


def mrc_ccg(X, tau_mat, lambda_mat, n_max=400, m_max=400, k_max=400,
			eps_1=1e-2, eps_2=1e-5, p=None, generate_features=False):
	"""
	Constraint generation, and column generation, algorithm for 0-1 loss
	Minimax Risk Classifiers with a large number of instances, and features
	(Algorithm 1 in [1]).

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
		Maximum number of features added in each iteration, if the features
		are generated.

	k_max : `int`, default=`400`
		Maximum number of iterations after solving the initial model.

	eps_1 : `float`, default=`1e-2`
		Violation threshold of the constraints.

	eps_2 : `float`, default=`1e-5`
		Violation threshold of the dual constraints of the features, if the
		features are generated.

	p : `array`-like of shape (`n_classes`,), default=`None`
		Proportions of the classes, in the order of the rows of `tau_mat`,
		for the class centers tau_y / p_y (13) in [1] of the initial model
		for multiclass classification. If `None`, the initial model for
		multiclass classification has no class centers. Not used for binary
		classification, whose class center is tau.

	generate_features : `bool`, default=`False`
		Whether the features are generated, for a large number of features,
		solving the primal model. Otherwise, the model has all the features,
		the dual model is solved and the constraints that are not tight are
		removed.

	Return:
	-------
	mu : `array`-like of the shape of `tau_mat`
		Parameters mu = mu_1 - mu_2 learnt by the algorithm.

	nu : `float`
		Parameter nu of the classifier learnt by the algorithm, the negative
		of the variable nu of the linear program P.

	R : `float`
		Worst-case error probability given by the last model, which is the
		optimal one up to the violation thresholds.

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

	if generate_features:
		model = PrimalModel(X, tau_mat, lambda_mat, rows, signs)
	else:
		model = DualModel(X, tau_mat, lambda_mat, rows, signs)

	# Initial model: the constraints of the class centers for the subsets
	# with a single label, nu >= 1/2 for binary classification, and the
	# initial features if the features are generated.
	if tau_mat.shape[0] == 1:
		model.add_constraint(0, (0, 1))
	for i in range(X_hat.shape[0]):
		for y in range(rows.shape[0]):
			model.add_constraint(i, (y,))
	if generate_features:
		model.add_features(initial_features(tau_mat, lambda_mat))

	R_k = []
	last_checked = 0
	# The initial model is always solved, then at most k_max iterations.
	k_max = max(k_max, 0)
	count_added = 1
	k = 0
	while k <= k_max and count_added > 0:
		mu, nu, R = model.solve()
		R_k.append(R)

		if generate_features:
			n_features_added = feat(model, m_max, eps_2)
		else:
			n_features_added = 0
			# While alpha_0 > 0, the model is bounded by alpha_0 instead of
			# the constraints, the solution is degenerate and the
			# constraints that are not tight cannot be removed without
			# cycling.
			if np.isclose(model.alpha_0.X, 0):
				model.remove_not_tight(mu, nu)
		n_constraints_added, last_checked = constr(model, mu, nu, n_max,
												   eps_1, last_checked)
		count_added = n_features_added + n_constraints_added
		k = k + 1

	R = R_k[-1]

	# The parameter nu of the classifier is the negative of the variable nu
	# of the linear program.
	return mu, -nu, R, R_k

def initial_features(tau_mat, lambda_mat):
	"""
	Initial subset J of features if the features are generated.

	For binary classification, the 1000 features with the largest
	|tau| - lambda, that is, with the largest violation of their dual
	constraint in a model without constraints (alpha = 0). For multiclass
	classification, the entries of the 100 features with the smallest
	standard deviation of tau across the classes, for all the classes.

	Parameters:
	-----------
	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	Return:
	-------
	features : `list` of `tuple`
		Features (r, j) in J.
	"""

	if tau_mat.shape[0] == 1:
		score = np.abs(tau_mat) - lambda_mat
		idx = np.argsort(score, axis=None)[::-1][:1000]
		features = zip(*np.unravel_index(idx, tau_mat.shape))
	else:
		cols = np.argsort(np.std(tau_mat, axis=0))[:100]
		features = [(r, j) for r in range(tau_mat.shape[0]) for j in cols]

	return [(int(r), int(j)) for r, j in features]

def constraint_coefficients(X, i, C, rows, signs, n_rows):
	"""
	Nonzero columns and values of the instance i, and coefficient of the
	instance in each row of F_c, for the constraint of instance i and subset
	C of the labels.

	Parameters:
	-----------
	X : `array`-like or sparse (csr) matrix of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	i : `int`
		Index of the instance.

	C : `tuple`
		Subset of the labels.

	rows : `array`-like of shape (`n_classes`,)
		Row r(y) of mu of each label.

	signs : `array`-like of shape (`n_classes`,)
		Sign s(y) of each label.

	n_rows : `int`
		Number of rows of mu.

	Return:
	-------
	nnz : `array`-like
		Nonzero columns of the instance.

	x_nnz : `array`-like
		Values of the instance in the nonzero columns.

	mask : `array`-like of shape (`n_classes`,)
		Subset C of the labels, as a boolean array over the labels.

	coef_rows : `array`-like of shape (`n_rows`,)
		Coefficient of the instance in each row of F_c.
	"""

	if sparse.issparse(X):
		start, end = X.indptr[i], X.indptr[i + 1]
		nnz, x_nnz = X.indices[start:end], X.data[start:end]
	else:
		x = X[i]
		nnz = np.flatnonzero(x)
		x_nnz = x[nnz]

	mask = np.zeros(rows.shape[0], dtype=bool)
	mask[list(C)] = True
	coef_rows = np.bincount(rows, weights=mask * signs,
							minlength=n_rows) / len(C)

	return nnz, x_nnz, mask, coef_rows

class DualModel:
	"""
	Dual linear model D_I of MRC 0-1 loss over the subset I of constraints
	(of the primal), with all the features, in which each constraint c in I
	is a variable alpha_c,

		D_I :  max   sum_{c in I} (1 - 1 / |C|) alpha_c
			  alpha
			   s.t.  (tau - lambda) (1 - alpha_0) <= F_I^T alpha <= (tau + lambda) (1 - alpha_0),
					 1^T alpha + alpha_0 = 1,  alpha >= 0,

	with F_I^T alpha = sum_{c in I} alpha_c F_c. The solution mu_1, mu_2, nu
	of P_I is given by the dual variables of its constraints.

	Parameters:
	-----------
	X : `array`-like or sparse (csr) matrix of shape (`n_samples`, `n_features`)
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
	(`add_constraint`) and removed (`remove_not_tight`) along the
	iterations.

	instances : `list` of `int`
		Index of the instance x in X.

	masks : `list` of `array`-like of shape (`n_classes`,)
		Subset C of the labels, as a boolean array over the labels.

	vars : `list` of `gurobipy.Var`
		Variable alpha_c of the constraint in the dual linear model.

	The constraints of the dual linear model are named after their dual
	variables:

	constrs : `dict`
		`'mu_1'` and `'mu_2'`, arrays of the shape of `tau_mat` with the
		constraints for the lower bound tau - lambda and the upper bound
		tau + lambda of each entry of F_I^T alpha, and `'nu'` for the sum of
		the variables.
	"""

	def __init__(self, X, tau_mat, lambda_mat, rows, signs):
		self.X = X
		self.tau_mat = tau_mat
		self.lambda_mat = lambda_mat
		self.rows = rows
		self.signs = signs

		self.instances = []
		self.masks = []
		self.vars = []

		# Define the MRC 0-1 linear model (dual).
		self.MRC_model = gp.Model("MRC_0_1_dual")
		self.MRC_model.Params.LogToConsole = 0
		self.MRC_model.Params.OutputFlag = 0
		self.MRC_model.setParam('DualReductions', 0)
		# The constraints are added as variables, so that the basis of the
		# previous iteration is a warm start, used on the presolved model.
		self.MRC_model.setParam('LPWarmStart', 2)
		self.MRC_model.ModelSense = GRB.MAXIMIZE

		self.alpha_0 = self.MRC_model.addVar(lb=0, name='alpha_0')
		self.MRC_model.update()

		# F_I^T alpha <= (tau + lambda) (1 - alpha_0)
		# -F_I^T alpha <= (lambda - tau) (1 - alpha_0)
		# The terms of the variables alpha_c are added with each constraint c.
		upper = tau_mat + lambda_mat
		lower = lambda_mat - tau_mat
		constr_mu_2 = np.empty(tau_mat.shape, dtype=object)
		constr_mu_1 = np.empty(tau_mat.shape, dtype=object)
		for r in range(tau_mat.shape[0]):
			for j in range(tau_mat.shape[1]):
				constr_mu_2[r, j] = self.MRC_model.addConstr(upper[r, j] * self.alpha_0 <= upper[r, j])
				constr_mu_1[r, j] = self.MRC_model.addConstr(lower[r, j] * self.alpha_0 <= lower[r, j])
		constr_nu = self.MRC_model.addConstr(self.alpha_0 == 1)
		self.constrs = {'mu_1': constr_mu_1, 'mu_2': constr_mu_2, 'nu': constr_nu}

	def add_constraint(self, i, C):
		"""
		Adds the constraint of instance i and subset C of the labels, as the
		variable alpha_c of the dual linear model with column F_c, objective
		-b_c = 1 - 1 / |C| and coefficient 1 in 1^T alpha + alpha_0 = 1.
		"""

		nnz, x_nnz, mask, coef_rows = constraint_coefficients(
			self.X, i, C, self.rows, self.signs, self.tau_mat.shape[0])

		coeffs = []
		dual_constrs = []
		for r in np.flatnonzero(coef_rows):
			values = coef_rows[r] * x_nnz
			coeffs.extend(values.tolist())
			dual_constrs.extend(self.constrs['mu_2'][r, nnz].tolist())
			coeffs.extend((-values).tolist())
			dual_constrs.extend(self.constrs['mu_1'][r, nnz].tolist())
		coeffs.append(1.)
		dual_constrs.append(self.constrs['nu'])

		alpha = self.MRC_model.addVar(lb=0, obj=1 - (1 / len(C)),
									  column=gp.Column(coeffs, dual_constrs))
		self.instances.append(i)
		self.masks.append(mask)
		self.vars.append(alpha)

	def solve(self):
		"""
		Solves the model, giving the solution mu = mu_1 - mu_2 and nu of P_I
		from the dual variables of the constraints of D_I, and the optimal
		value R^k.
		"""

		self.MRC_model.optimize()
		mu_1 = self.MRC_model.getAttr("Pi", self.constrs['mu_1'].flatten().tolist())
		mu_2 = self.MRC_model.getAttr("Pi", self.constrs['mu_2'].flatten().tolist())
		mu = (np.asarray(mu_1) - np.asarray(mu_2)).reshape(self.tau_mat.shape)
		nu = self.constrs['nu'].Pi

		return mu, nu, self.MRC_model.objVal

	def values(self, mu):
		"""
		Value F_c mu - b_c of each constraint c in I at mu, that is,
		(sum_{y in C} Phi(x, y)^T mu - 1) / |C| + 1.
		"""

		scores = np.asarray(self.X[self.instances] @ mu.T)[:, self.rows] * self.signs
		masks = np.asarray(self.masks)
		return (np.sum(scores * masks, axis=1) - 1) / np.sum(masks, axis=1) + 1

	def remove_not_tight(self, mu, nu):
		"""
		Removes the constraints in I that are not tight at the solution
		(mu, nu), that is, with F_c mu - nu - b_c < 0.
		"""

		keep = np.isclose(self.values(mu) - nu, 0)
		for c in np.flatnonzero(~keep):
			self.MRC_model.remove(self.vars[c])
		self.instances = [i for i, k in zip(self.instances, keep) if k]
		self.masks = [m for m, k in zip(self.masks, keep) if k]
		self.vars = [v for v, k in zip(self.vars, keep) if k]

class PrimalModel:
	"""
	Linear model P_{I,J} of MRC 0-1 loss over the subset I of constraints
	and the subset J of features, in which each feature (r, j) in J is a
	pair of variables mu_1, mu_2.

	Parameters:
	-----------
	X : `array`-like or sparse (csr) matrix of shape (`n_samples`, `n_features`)
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
		# Both constraints and features are added, so that the basis of the
		# previous iteration is not a good warm start, and each model is
		# solved from scratch (with presolve).
		self.MRC_model.setParam('LPWarmStart', 0)

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

		nnz, x_nnz, _, coef_rows = constraint_coefficients(
			self.X, i, C, self.rows, self.signs, self.tau_mat.shape[0])

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
													  1 - (1 / len(C))))
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

	def solve(self):
		"""
		Solves the model, giving the solution mu = mu_1 - mu_2 and nu of
		P_{I,J}, with mu zero for the features not in J, and the optimal
		value R^k.
		"""

		self.MRC_model.optimize()
		mu = np.zeros(self.tau_mat.shape)
		for (r, j), (mu_1, mu_2) in self.mu_vars.items():
			mu[r, j] = mu_1.X - mu_2.X

		return mu, self.nu.X, self.MRC_model.objVal

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
	model : `PrimalModel`
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
	model : `DualModel` or `PrimalModel`
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
