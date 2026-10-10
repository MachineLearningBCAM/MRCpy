'''Constraint generation for the 0-1 loss MRC linear program with a large
number of instances, as described in [1].

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
s(y) x / |C| over the labels y in C that use the row r of mu. For
multiclass classification, the row y is x / |C| if y is in C and zero
otherwise. For binary classification, the single row is x, -x and 0 for
C = {0}, {1} and {0, 1}.

The number of constraints, n (2^|Y| - 1), is too large for a large number
of instances n. The constraints are generated (function CONSTR of
Algorithm 1 in [1]): a subset I of constraints is in the model, and the
dual linear program (6) in [1] over I,

    D_I :  max   -b_I^T alpha
          alpha
           s.t.  (tau - lambda) (1 - alpha_0) <= F_I^T alpha <= (tau + lambda) (1 - alpha_0)
                 1^T alpha + alpha_0 = 1,  alpha >= 0,

is solved at each iteration k, with the elementwise inequalities for each
entry of mu and F_I^T alpha = sum_{c in I} alpha_c F_c. The variable
alpha_0 corresponds to the additional constraint of the objective being
nonnegative, which bounds the model for any I (Section III-E in [1]).
The solution mu_1^k, mu_2^k, nu^k of P_I is obtained from the dual
solution of D_I, and R^k is its optimal value. Then, the constraints in I
that are not tight are removed (once alpha_0 = 0, that is, once the
constraints in I bound the model), and the constraint with the maximum
violation of each instance, obtained with Algorithm 2 in [1], is added to
I if its violation exceeds eps_1.

The initial subset I is given by the class centers X_hat in (13) in [1],
for the subsets with a single label: x_hat_y = tau_y / p_y for each label
y, with p the class proportions, for multiclass classification, and tau
for binary classification, which is the average of the instances with
Phi(x, 1) = -Phi(x, 0). As averages of the instances, their constraints
are implied by the constraints of the instances.

.. [1] Bondugula, K., Mazuelas, S., & Pérez, A. (2025).
       Efficient Large-Scale Learning of Minimax Risk Classifiers.
       arXiv preprint arXiv:2511.17626.
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def mrc_ccg_large_number_of_samples(X, tau_mat, lambda_mat, n_max=400, k_max=150, eps_1=1e-2,
					p=None):
	"""
	Constraint generation algorithm for 0-1 loss Minimax Risk Classifiers
	with a large number of instances (Algorithm 1 in [1], without the
	generation of features).

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	n_max : `int`, default=`400`
		Maximum number of constraints added in each iteration.

	k_max : `int`, default=`150`
		Maximum number of iterations after solving the initial model.

	eps_1 : `float`, default=`1e-2`
		Violation threshold of the constraints.

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
		Parameter nu learnt by the algorithm.

	R : `float`
		Worst-case error probability given by the last model, which is the
		optimal one up to the violation threshold eps_1 of the constraints.

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

	# The initial model has the constraints of the class centers X_hat for
	# the subsets with a single label, and it is also bounded by the
	# variable alpha_0 of the dual linear model. The class centers are
	# checked along with the instances.
	X = np.vstack((X_hat, X))
	n_centers = X_hat.shape[0]

	MRC_model, constrs = mrc_dual_lp_model(tau_mat, lambda_mat)
	I = Constraints(MRC_model, constrs, X, rows, signs)
	if tau_mat.shape[0] == 1: # For binary
		I.add(0, (0, 1))  # Add nu >= 1/2 (same for all instances)
	for i in range(n_centers):
		for y in range(rows.shape[0]):
			I.add(i, (y,))

	R_k = []
	last_checked = 0
	# The initial model is always solved, then at most k_max iterations.
	k_max = max(k_max, 0)
	count_added = 1
	k = 0
	while k <= k_max and count_added > 0:
		MRC_model.optimize()
		mu, nu = dual_solution(MRC_model, constrs, tau_mat.shape)
		R_k.append(MRC_model.objVal)

		# While alpha_0 > 0, the model is bounded by alpha_0 instead of the
		# constraints, the solution is degenerate and the constraints that
		# are not tight cannot be removed without cycling.
		if np.isclose(MRC_model.getVarByName('alpha_0').X, 0):
			I.remove_not_tight(mu, nu)
		count_added, last_checked = select(I, mu, nu, n_max, eps_1, last_checked)
		k = k + 1

	R = R_k[-1]

	return mu, nu, R, R_k

def mrc_dual_lp_model(tau_mat, lambda_mat):
	"""
	Builds the dual linear model D_I of MRC 0-1 loss with an empty subset I
	of constraints (of the primal), that is, with only the variable alpha_0.

	Parameters:
	-----------
	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates tau.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Deviations lambda of the mean estimates.

	Return:
	-------
	MRC_model : A MRC object in GUROBI
		The dual linear model.

	constrs : `dict`
		Constraints of the dual linear model, named after their dual
		variables: `'mu_1'` and `'mu_2'`, of the shape of `tau_mat`, for the
		lower bound tau - lambda and the upper bound tau + lambda of each
		entry of F_I^T alpha, and `'nu'` for the sum of the variables.
	"""

	MRC_model = gp.Model("MRC_0_1_dual")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	MRC_model.setParam('DualReductions', 0)
	MRC_model.ModelSense = GRB.MAXIMIZE

	alpha_0 = MRC_model.addVar(lb=0, name='alpha_0')
	MRC_model.update()

	# F_I^T alpha <= (tau + lambda) (1 - alpha_0)
	# -F_I^T alpha <= (lambda - tau) (1 - alpha_0)
	# The terms of the variables alpha_c are added with each constraint c.
	upper = tau_mat + lambda_mat
	lower = lambda_mat - tau_mat
	constr_mu_2 = np.empty(tau_mat.shape, dtype=object)
	constr_mu_1 = np.empty(tau_mat.shape, dtype=object)
	for r in range(tau_mat.shape[0]):
		for j in range(tau_mat.shape[1]):
			constr_mu_2[r, j] = MRC_model.addConstr(upper[r, j] * alpha_0 <= upper[r, j])
			constr_mu_1[r, j] = MRC_model.addConstr(lower[r, j] * alpha_0 <= lower[r, j])
	constr_nu = MRC_model.addConstr(alpha_0 == 1)

	return MRC_model, {'mu_1': constr_mu_1, 'mu_2': constr_mu_2, 'nu': constr_nu}

def dual_solution(MRC_model, constrs, shape):
	"""
	Solution mu = mu_1 - mu_2 and nu of the primal P_I from the solution of
	the dual linear model D_I.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The solved dual linear model.

	constrs : `dict`
		Constraints of the dual linear model.

	shape : `tuple`
		Shape of mu, the shape of `tau_mat`.

	Return:
	-------
	mu : `array`-like of shape `shape`
		Parameters mu.

	nu : `float`
		Parameter nu.
	"""

	mu_1 = MRC_model.getAttr("Pi", constrs['mu_1'].flatten().tolist())
	mu_2 = MRC_model.getAttr("Pi", constrs['mu_2'].flatten().tolist())
	mu = (np.asarray(mu_1) - np.asarray(mu_2)).reshape(shape)
	nu = constrs['nu'].Pi

	return mu, nu

class Constraints:
	"""
	Subset I of constraints (of the primal) in the model, given by an
	instance and a subset C of the labels, with their variables alpha_c in
	the dual linear model.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The dual linear model.

	constrs : `dict`
		Constraints of the dual linear model.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	rows : `array`-like of shape (`n_classes`,)
		Row r(y) of mu of each label.

	signs : `array`-like of shape (`n_classes`,)
		Sign s(y) of each label.

	Attributes:
	-----------
	The following lists have an element for each constraint c = (x, C) in
	I, aligned by position. They are updated as the constraints are added
	(`add`) and removed (`remove_not_tight`) along the iterations.

	instances : `list` of `int`
		Index of the instance x in X.

	masks : `list` of `array`-like of shape (`n_classes`,)
		Subset C of the labels, as a boolean array over the labels.

	vars : `list` of `gurobipy.Var`
		Variable alpha_c of the constraint in the dual linear model.
	"""

	def __init__(self, MRC_model, constrs, X, rows, signs):
		self.MRC_model = MRC_model
		self.constrs = constrs
		self.X = X
		self.rows = rows
		self.signs = signs
		self.instances = []
		self.masks = []
		self.vars = []

	def add(self, i, C):
		"""
		Adds the constraint of instance i and subset C of the labels, as the
		variable alpha_c of the dual linear model with column F_c, objective
		-b_c = 1 - 1 / |C| and coefficient 1 in 1^T alpha + alpha_0 = 1.
		"""

		x = self.X[i]
		nnz = np.flatnonzero(x)
		mask = np.zeros(self.rows.shape[0], dtype=bool)
		mask[list(C)] = True
		size = len(C)

		# Coefficient of x in each row of F_c
		coef_rows = np.bincount(self.rows, weights=mask * self.signs,
								minlength=self.constrs['mu_2'].shape[0]) / size

		coeffs = []
		dual_constrs = []
		for r in np.flatnonzero(coef_rows):
			values = coef_rows[r] * x[nnz]
			coeffs.extend(values.tolist())
			dual_constrs.extend(self.constrs['mu_2'][r, nnz].tolist())
			coeffs.extend((-values).tolist())
			dual_constrs.extend(self.constrs['mu_1'][r, nnz].tolist())
		coeffs.append(1.)
		dual_constrs.append(self.constrs['nu'])

		alpha = self.MRC_model.addVar(lb=0, obj=1 - (1 / size),
									  column=gp.Column(coeffs, dual_constrs))
		self.instances.append(i)
		self.masks.append(mask)
		self.vars.append(alpha)

	def values(self, mu):
		"""
		Value F_c mu - b_c of each constraint c in I at mu, that is,
		(sum_{y in C} Phi(x, y)^T mu - 1) / |C| + 1.
		"""

		scores = (self.X[self.instances] @ mu.T)[:, self.rows] * self.signs
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

def select(I, mu, nu, n_max, eps_1, last_checked):
	"""
	Adds to I the constraint with the maximum violation of the instances,
	if its violation exceeds eps_1, checking the instances in order from
	last_checked and adding at most n_max constraints.

	The constraint with the maximum violation of an instance x corresponds
	to the subset C achieving the maximum value (9) in [1],

		psi = max_C (sum_{y in C} Phi(x, y)^T mu - 1) / |C|,

	obtained as in Algorithm 2 in [1] by sorting Phi(x, y)^T mu in
	decreasing order, and its violation is psi + 1 - nu.

	Parameters:
	-----------
	I : `Constraints`
		Subset of constraints in the model.

	mu : `array`-like
		Parameters mu.

	nu : `float`
		Parameter nu.

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

	X = I.X
	n = X.shape[0]

	# Phi(x, y)^T mu for all the instances and labels, sorted in decreasing
	# order. For a fixed size of C, the maximum is given by the labels with
	# the largest values.
	scores = (X @ mu.T)[:, I.rows] * I.signs
	order = np.argsort(-scores, axis=1, kind='stable')
	sorted_scores = np.take_along_axis(scores, order, axis=1)
	psi_size = (np.cumsum(sorted_scores, axis=1) - 1) / np.arange(1, scores.shape[1] + 1)
	size = np.argmax(psi_size, axis=1) + 1
	psi = psi_size[np.arange(n), size - 1]
	violation = psi + 1 - nu

	# Instances in order from last_checked
	checked = (last_checked + np.arange(n)) % n
	violated = np.flatnonzero(violation[checked] > eps_1)[:n_max]

	for p in violated:
		i = checked[p]
		I.add(i, tuple(order[i, :size[i]].tolist()))

	# With n_max constraints added, the next selection starts after the
	# last instance added. Otherwise, all the instances were checked and it
	# starts from the same instance.
	if n_max > 0 and violated.shape[0] == n_max:
		last_checked = (last_checked + violated[-1] + 1) % n

	return violated.shape[0], last_checked
