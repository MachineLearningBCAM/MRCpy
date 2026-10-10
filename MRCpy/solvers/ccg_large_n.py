'''Constraint generation for the 0-1 loss MRC linear program with a large
number of instances.

The parameters mu, the mean estimates tau and their deviations lambda are
matrices of shape (n_classes, n_features), or (1, n_features) for binary
classification. With R the number of rows, each class y uses the row r(y)
of mu with the sign s(y):

    multiclass: r(y) = y, s(y) = 1,
    binary:     r(y) = 0, s(0) = 1, s(1) = -1  (phi(x, 1) = -phi(x, 0)).

The linear program has a constraint for each instance x_i and non-empty
subset S of classes,

    nu >= (sum_{y in S} s(y) x_i . mu_{r(y)} - 1) / |S| + 1
        = <G_c, mu> + 1 - 1 / |S|,

too many to be built. G_c is the (R, n_features) matrix of coefficients of
the constraint c = (i, S): its row r is the sum of s(y) x_i / |S| over the
classes y in S that use the row r of mu. For multiclass classification,
the row y is x_i / |S| if y is in S and zero otherwise. For binary
classification, the single row is x_i, -x_i and 0 for S = {0}, {1} and
{0, 1}.

The constraints are generated as cuts. The dual linear program, with a
variable alpha_c for each constraint c in the model,

    max  sum_c (1 - 1 / |S|) alpha_c
    s.t. |sum_c alpha_c G_c - tau (1 - alpha_0)| <= lambda (1 - alpha_0)
             (elementwise, one constraint for each entry of mu)
         sum_c alpha_c + alpha_0 = 1,  alpha >= 0,

is solved, where alpha_0 is the variable of the constraint that the
objective is nonnegative, which bounds the model without any constraint
c. Then, mu and nu are recovered from its dual solution, the constraints
that are not tight are removed (once alpha_0 = 0, that is, once the
constraints c bound the model), and the most violated constraint of each
instance, given by sorting the scores of the classes, is added if its
violation exceeds eps.
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def mrc_ccg_large_n(X, tau_mat, lambda_mat, n_max=400, max_iters=150, eps=1e-2):
	"""
	Constraint generation algorithm for 0-1 loss Minimax Risk Classifiers
	with a large number of instances.

	Parameters:
	-----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances (without one-hot encoding).

	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates. One row for binary classification.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	n_max : `int`, default=`400`
		Maximum number of constraints added in each iteration.

	max_iters : `int`, default=`150`
		Maximum number of iterations after solving the initial model.

	eps : `float`, default=`1e-2`
		Violation threshold of the constraints.

	Return:
	-------
	mu : `array`-like of the shape of `tau_mat`
		Parameters learnt by the algorithm.

	nu : `float`
		Parameter learnt by the algorithm.

	R : `float`
		Upper bound of the MRC classifier given by the last model, which is
		the optimal one up to the violation threshold eps of the constraints.

	R_k : `list`
		Upper bound given by the model at each iteration.
	"""

	tau_mat = np.atleast_2d(np.asarray(tau_mat, dtype=float))
	lambda_mat = np.atleast_2d(np.asarray(lambda_mat, dtype=float))

	# Row of mu and sign of each class
	if tau_mat.shape[0] == 1:
		rows = np.array([0, 0])
		signs = np.array([1., -1.])
	else:
		rows = np.arange(tau_mat.shape[0])
		signs = np.ones(tau_mat.shape[0])

	# The initial model has no constraints of the instances, it is bounded
	# by the variable alpha_0 of the dual linear model. For binary
	# classification, the constraint of the subset of both classes,
	# nu >= 1/2, is the same for all the instances and it is added.
	MRC_model, constrs = mrc_dual_lp_model(tau_mat, lambda_mat)
	cuts = Cuts(MRC_model, constrs, X, rows, signs)
	if tau_mat.shape[0] == 1:
		cuts.add(0, (0, 1))

	R_k = []
	last_checked = 0
	count_added = 1
	k = 0
	while k == 0 or (k <= max_iters and count_added > 0):
		MRC_model.optimize()
		mu, nu = dual_solution(MRC_model, constrs, tau_mat.shape)
		R_k.append(MRC_model.objVal)

		# While alpha_0 > 0, the model is bounded by alpha_0 instead of the
		# constraints, the solution is degenerate and the constraints that
		# are not tight cannot be removed without cycling.
		if np.isclose(MRC_model.getVarByName('var_0').X, 0):
			cuts.remove_not_tight(mu, nu)
		count_added, last_checked = select(cuts, mu, nu, n_max, eps, last_checked)
		k = k + 1

	R = R_k[-1]

	return mu, nu, R, R_k

def mrc_dual_lp_model(tau_mat, lambda_mat):
	"""
	Builds the dual linear model of MRC 0-1 loss without any constraint
	(primal) yet, that is, with only the variable alpha_0.

	Parameters:
	-----------
	tau_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Mean estimates.

	lambda_mat : `array`-like of shape (`n_classes`, `n_features`) or (1, `n_features`)
		Standard deviation of the estimates.

	Return:
	-------
	MRC_model : A MRC object in GUROBI
		The dual linear model.

	constrs : `dict`
		Constraints of the dual linear model: `'-'` and `'+'`, of the shape
		of `tau_mat`, for the upper bound tau + lambda and the lower bound
		tau - lambda of each entry, and `'='` for the sum of the variables.
		Their dual variables are the negative part of mu, the positive part
		of mu and nu.
	"""

	MRC_model = gp.Model("MRC_0_1_dual")
	MRC_model.Params.LogToConsole = 0
	MRC_model.Params.OutputFlag = 0
	MRC_model.setParam('DualReductions', 0)
	MRC_model.ModelSense = GRB.MAXIMIZE

	alpha_0 = MRC_model.addVar(lb=0, name='var_0')
	MRC_model.update()

	# sum_c alpha_c G_c <= (tau + lambda) (1 - alpha_0)
	# -sum_c alpha_c G_c <= (lambda - tau) (1 - alpha_0)
	# The terms of the variables alpha_c are added with each constraint c.
	upper = tau_mat + lambda_mat
	lower = lambda_mat - tau_mat
	constr_minus = np.empty(tau_mat.shape, dtype=object)
	constr_plus = np.empty(tau_mat.shape, dtype=object)
	for r in range(tau_mat.shape[0]):
		for j in range(tau_mat.shape[1]):
			constr_minus[r, j] = MRC_model.addConstr(upper[r, j] * alpha_0 <= upper[r, j])
			constr_plus[r, j] = MRC_model.addConstr(lower[r, j] * alpha_0 <= lower[r, j])
	constr_eq = MRC_model.addConstr(alpha_0 == 1)

	return MRC_model, {'-': constr_minus, '+': constr_plus, '=': constr_eq}

def dual_solution(MRC_model, constrs, shape):
	"""
	Parameters mu and nu of the classifier from the solution of the dual
	linear model.

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

	pi_plus = MRC_model.getAttr("Pi", constrs['+'].flatten().tolist())
	pi_minus = MRC_model.getAttr("Pi", constrs['-'].flatten().tolist())
	mu = (np.asarray(pi_plus) - np.asarray(pi_minus)).reshape(shape)
	nu = constrs['='].Pi

	return mu, nu

class Cuts:
	"""
	Constraints (cuts) of the linear program in the model, as instances and
	subsets of classes, with their variables in the dual linear model.

	Parameters:
	-----------
	MRC_model : A MRC object in GUROBI
		The dual linear model.

	constrs : `dict`
		Constraints of the dual linear model.

	X : `array`-like of shape (`n_samples`, `n_features`)
		Features corresponding with the instances.

	rows : `array`-like of shape (`n_classes`,)
		Row of mu of each class.

	signs : `array`-like of shape (`n_classes`,)
		Sign of each class.
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

	def add(self, i, subset):
		"""
		Adds the constraint of instance i and the given subset of classes,
		as the variable of the dual linear model with column G_c.
		"""

		x = self.X[i]
		nnz = np.flatnonzero(x)
		mask = np.zeros(self.rows.shape[0], dtype=bool)
		mask[list(subset)] = True
		size = len(subset)

		# Coefficient of x_i in each row of G_c
		coef_rows = np.bincount(self.rows, weights=mask * self.signs,
								minlength=self.constrs['-'].shape[0]) / size

		coeffs = []
		dual_constrs = []
		for r in np.flatnonzero(coef_rows):
			values = coef_rows[r] * x[nnz]
			coeffs.extend(values.tolist())
			dual_constrs.extend(self.constrs['-'][r, nnz].tolist())
			coeffs.extend((-values).tolist())
			dual_constrs.extend(self.constrs['+'][r, nnz].tolist())
		coeffs.append(1.)
		dual_constrs.append(self.constrs['='])

		alpha = self.MRC_model.addVar(lb=0, obj=1 - (1 / size),
									  column=gp.Column(coeffs, dual_constrs))
		self.instances.append(i)
		self.masks.append(mask)
		self.vars.append(alpha)

	def values(self, mu):
		"""
		Right-hand side of each constraint in the model at mu,
		(sum_{y in S} s(y) x_i . mu_{r(y)} - 1) / |S| + 1.
		"""

		scores = (self.X[self.instances] @ mu.T)[:, self.rows] * self.signs
		masks = np.asarray(self.masks)
		return (np.sum(scores * masks, axis=1) - 1) / np.sum(masks, axis=1) + 1

	def remove_not_tight(self, mu, nu):
		"""
		Removes the constraints that are not tight at the solution (mu, nu).
		"""

		keep = np.isclose(self.values(mu) - nu, 0)
		for c in np.flatnonzero(~keep):
			self.MRC_model.remove(self.vars[c])
		self.instances = [i for i, k in zip(self.instances, keep) if k]
		self.masks = [m for m, k in zip(self.masks, keep) if k]
		self.vars = [v for v, k in zip(self.vars, keep) if k]

def select(cuts, mu, nu, n_max, eps, last_checked):
	"""
	Adds the most violated constraint of the instances, if its violation
	exceeds eps, checking the instances in order from last_checked and
	adding at most n_max constraints.

	Parameters:
	-----------
	cuts : `Cuts`
		Constraints in the model.

	mu : `array`-like
		Parameters mu.

	nu : `float`
		Parameter nu.

	n_max : `int`
		Maximum number of constraints added.

	eps : `float`
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

	X = cuts.X
	n = X.shape[0]

	# Scores of the classes, sorted in decreasing order. For a fixed size r,
	# the most violated subset is given by the r largest scores.
	scores = (X @ mu.T)[:, cuts.rows] * cuts.signs
	order = np.argsort(-scores, axis=1, kind='stable')
	sorted_scores = np.take_along_axis(scores, order, axis=1)
	values = (np.cumsum(sorted_scores, axis=1) - 1) / np.arange(1, scores.shape[1] + 1)
	size = np.argmax(values, axis=1) + 1
	violation = values[np.arange(n), size - 1] + 1 - nu

	# Instances in order from last_checked
	checked = (last_checked + np.arange(n)) % n
	violated = np.flatnonzero(violation[checked] > eps)[:n_max]

	for p in violated:
		i = checked[p]
		cuts.add(i, tuple(order[i, :size[i]].tolist()))

	# With n_max constraints added, the next selection starts after the
	# last instance added. Otherwise, all the instances were checked and it
	# starts from the same instance.
	if n_max > 0 and violated.shape[0] == n_max:
		last_checked = (last_checked + violated[-1] + 1) % n

	return violated.shape[0], last_checked
