from .cg import alg1
from .initialization import fo_init
import time
import numpy as np
import itertools as it
import scipy.special as scs

def mrc_cg_m_large(X, y, phi_ob, s, n_max, k_max, eps):
	"""
	Efficient learning of 0-1 MRCs.

	Parameters
    ----------
    X : `array`-like of shape (`n_samples`, `n_features`)
        Training instances used in

        `n_samples` is the number of training samples and
        `n_features` is the number of features.

    y : `array`-like of shape (`n_samples`, 1), default = `None`
        Labels corresponding to the training instances
        used only to compute the expectation estimates.

	phi_ob : `BasePhi` instance
		This is an instance of the `BasePhi` 
		feature mapping class of MRCs. Any feature
		mapping object derived from BasePhi can be used
		to provide a new interpretation to the input data `X`.

	s : `float`, default = `0.3`
        Parameter that tunes the estimation of expected values
        of feature mapping function. It is used to calculate :math:`\lambda`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \\lambda = s * \\text{std}(\\phi(X,Y)) / \\sqrt{\\left| X \\right|}

        where (X,Y) is the dataset of training samples and their
        labels respectively and
        :math:`\\text{std}(\\phi(X,Y))` stands for standard deviation
        of :math:`\\phi(X,Y)` in the supervised dataset (X,Y).

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm.

	k_max : `int`, default=`20`
		Maximum number of iterations allowed for termination of the algorithm.

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	Returns
	-------
	mu : `array`-like of shape (`n_features`) or `float`
		Parameters learnt by the algorithm.

	nu : `float`
		Parameter learnt by the algorithm.

	R : `float`
		Optimized upper bound of the MRC classifier.

	I : `list`
		List of indices of the features selected

	R_k : `list` of shape (no_of_iterations)
		List of worst-case error probabilites
		obtained for the subproblems at each iteration.

	totalTime : `float`
		Total time taken by the algorithm.

	initTime : `float`
		Time taken for the initialization to the algorithm.
	"""

	n = X.shape[0]

	phi_ = phi_ob.eval_x(X)
	tau_ = phi_ob.est_exp(X, y)
	lambda_ = s * (phi_ob.est_std(X, y)) / np.sqrt(X.shape[0])

	F_ = np.vstack((np.sum(phi_[:, S, ], axis=1)
				   for numVals in range(1, phi_ob.n_classes + 1)
				   for S in it.combinations(np.arange(phi_ob.n_classes), numVals)))

	cardS = np.arange(1, phi_ob.n_classes + 1).\
				repeat([n * scs.comb(phi_ob.n_classes, numVals)
						for numVals in np.arange(1, phi_ob.n_classes + 1)])

	# Constraint coefficient matrix
	F = F_ / (cardS[:, np.newaxis])

	# The bounds on the constraints
	b = 1 - (1 / cardS)

	# Calculate the time
	# Total time taken.
	totalTime = time.time()
	# Initialization time
	initTime = time.time()

	#-> Initialization.
	I, warm_start, nu_init = fo_init(F,
									 b,
									 tau_,
									 lambda_)
	initTime = time.time() - initTime

	#-> Run the CG code.
	mu, nu, R, I, R_k = alg1(F,
							 b,
							 tau_,
							 lambda_,
							 I,
							 n_max,
							 k_max,
							 warm_start,
							 nu_init,
							 eps)

	totalTime = time.time() - totalTime

	return mu, nu, R, I, R_k, totalTime, initTime