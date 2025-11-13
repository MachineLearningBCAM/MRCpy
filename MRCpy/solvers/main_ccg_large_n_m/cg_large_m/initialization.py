import numpy as np

def fo_optimization_mrc(F, b, tau_, lambda_, max_iters):
	'''
	Solution of the MRC convex optimization (minimization)
	using an optimized version of the Nesterov accelerated approach.

	.. seealso::         [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
							The Strength of Nesterov’s Extrapolation in
							the Individual Convergence of Nonsmooth
							Optimization. IEEE transactions on
							neural networks and learning systems,
							31(7), 2557-2568.
							<https://ieeexplore.ieee.org/document/8822632>`_

	Parameters
	----------
	F : `array`-like of shape (:math:`m_1`, :math:`m_2`)
		Where :math:`m_1` is approximately
		:math:`(2^{\\textrm{n_classes}}-1) *
		\\textrm{min}(5000,\\textrm{len}(X))`,
		where the second factor is the number of training samples used for
		solving the optimization problem.

	b : `array`-like of shape (:math:`m_1`,)
		Where :math:`m_1` is approximately
		:math:`(2^{\\textrm{n_classes}}-1) *
		\\textrm{min}(5000,\\textrm{len}(X))`,
		where the second factor is the number of training samples used for
		solving the optimization problem.

	Returns
	-------
	new_params_ : `dict`
		Dictionary that stores the optimal points
		(`w_k`: `array-like` shape (`m`,), `w_k_prev`: `array-like`
		 shape (`m`,)), where `m` is the length of the feature
		mapping vector, and best value
		for the upper bound (`best_value`: `float`) of the function and
		the parameters corresponding to the optimized function value
		(`mu`: `array-like` shape (`m`,),
		`nu`: `float`).
	'''
	b = np.reshape(b, (-1, 1))
	n, m = F.shape
	a = np.reshape(-tau_, (-1, 1))  # make it a column
	mu_k = np.zeros((m, 1))
	c_k = 1
	theta_k = 1
	nu_k = 0
	alpha = F @ a
	G = F @ F.transpose()
	H = 2 * F @ np.diag(lambda_)
	y_k = mu_k
	v_k = F @ mu_k + b
	w_k = v_k
	s_k = np.sign(mu_k)
	d_k = (1 / 2) * H @ s_k
	i_k = np.argmax(v_k)
	mu_star = mu_k
	v_star = -v_k[i_k]
	lambda_ = np.reshape(lambda_, (-1, 1))  # make it a column
	f_star = a.transpose() @ mu_k +\
		lambda_.transpose() @ np.abs(mu_k) + v_k[i_k]

	if n * n > (1024) ** 3:  # Large Dimension
		for k in range(1, max_iters + 1):
			g_k = a + lambda_ * s_k + F[[i_k], :].T
			y_k_next = mu_k - c_k * g_k
			mu_k_next = (1 + nu_k) * y_k_next - nu_k * y_k
			u_k = alpha + d_k + G[:, [i_k]]
			w_k_next = v_k - c_k * u_k
			v_k_next = (1 + nu_k) * w_k_next - nu_k * w_k
			i_k_next = np.argmax(v_k_next)
			s_k_next = np.sign(mu_k_next)
			delta_k = s_k_next - s_k

			d_k_next = d_k
			for i in range(m):
				if delta_k[i] == 2:
					d_k_next = d_k_next + H[:, [i]]
				elif delta_k[i] == -2:
					d_k_next = d_k_next - H[:, [i]]
				elif delta_k[i] == 1 or delta_k[i] == -1:
					d_k_next = d_k_next + (1 / 2)\
						* np.sign(delta_k[i]) * H[:, [i]]

			c_k_next = (k + 1) ** (-3 / 2)
			theta_k_next = 2 / (k + 1)
			nu_k_next = theta_k_next * ((1 / theta_k) - 1)
			f_k_next = a.transpose() @ mu_k_next +\
				lambda_.transpose() @ np.abs(mu_k_next) +\
				v_k_next[i_k_next]
			if f_k_next < f_star:
				f_star = f_k_next
				mu_star = mu_k_next
				v_star = -v_k_next[i_k_next]

			# Update variables
			mu_k = mu_k_next
			y_k = y_k_next
			nu_k = nu_k_next
			v_k = v_k_next
			w_k = w_k_next
			s_k = s_k_next
			d_k = d_k_next
			c_k = c_k_next
			i_k = i_k_next
			theta_k = theta_k_next

	else:  # Small Dimension

		MD = H / 2

		for k in range(1, max_iters + 1):
			g_k = a + lambda_ * s_k + F[[i_k], :].T
			y_k_next = mu_k - c_k * g_k
			mu_k_next = (1 + nu_k) * y_k_next - nu_k * y_k
			u_k = alpha + d_k + G[:, [i_k]]
			w_k_next = v_k - c_k * u_k
			v_k_next = (1 + nu_k) * w_k_next - nu_k * w_k
			i_k_next = np.argmax(v_k_next)
			s_k_next = np.sign(mu_k_next)
			delta_k = s_k_next - s_k

			index = np.where(delta_k != 0)[0]
			d_k_next = d_k + MD[:, index] @ delta_k[index]

			c_k_next = (k + 1) ** (-3 / 2)
			theta_k_next = 2 / (k + 1)
			nu_k_next = theta_k_next * ((1 / theta_k) - 1)
			f_k_next = a.transpose() @ mu_k_next +\
				lambda_.transpose() @ np.abs(mu_k_next) +\
				v_k_next[i_k_next]
			if f_k_next < f_star:
				f_star = f_k_next
				mu_star = mu_k_next
				v_star = -v_k_next[i_k_next]

			# Update variables
			mu_k = mu_k_next
			y_k = y_k_next
			nu_k = nu_k_next
			v_k = v_k_next
			w_k = w_k_next
			s_k = s_k_next
			d_k = d_k_next
			c_k = c_k_next
			i_k = i_k_next
			theta_k = theta_k_next

	new_params_ = {'w_k': w_k_next.flatten(),
				   'w_k_prev': w_k.flatten(),
				   'mu': mu_star.flatten(),
				   'nu': v_star,
				   'best_value': f_star[0][0],
				   }

	return new_params_

def fo_init(F, b, tau_, lambda_):

	"""
		Generate the initial set of features for MRC_CG using the MRCpy library's
		accelerated subgradient implementation. Pick only top 100 features.

	Parameters:
	-----------
	F : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : `array`-like of shape (no_of_constraints)
		Right handside of the constraints.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	Returns:
	--------
	I : `list`
		List of indices of the features selected

	warm_start : `list`, default=`None`
		Coefficients corresponding to features in I.

	nu_ : `float`, default=`None`
		Coefficient nu corresponding to the warm start (mu)

	"""

	# Reduce the feature space by restricting the number of features 10*N
	# based on the variance in the features, that is, picking features first 
	# 10*N minimum variance features.
	N 				= F.shape[0]
	argsort_columns = np.argsort(np.abs(lambda_))
	index_CG        = argsort_columns[:10*N]

	# Solve the optimization using the reduced training set
	# and first order subgradient methods to get an
	# initial low accuracy solution in minimum time.
	F_reduced = F[:, index_CG]
	F_reduced_t = F_reduced.transpose()

	# Calculate the upper bound
	upper_params_ = \
			 fo_optimization_mrc(F_reduced,
								 b,
								 tau_[index_CG],
								 lambda_[index_CG],
								 100)
	mu_ = upper_params_['mu']
	nu_ = upper_params_['nu']

	# Transform the solution obtained in the reduced space
	# to the original space
	initial_features_limit = 100
	if np.sum(mu_!=0) > initial_features_limit:
		I = (np.argsort(np.abs(mu_))[::-1])[:initial_features_limit]
	else:
		I = np.where(mu_!=0)[0]

	warm_start = mu_[I] 
	I = np.array(index_CG)[I].tolist()

	return I, warm_start, nu_
