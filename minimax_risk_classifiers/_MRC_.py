"""
	Super class for Minimax Risk Classifier
"""

import numpy as np
import cvxpy as cvx
from sklearn.utils import check_array

# Import the feature mapping
from minimax_risk_classifiers.phi import *

class _MRC_():

	"""
	Minimax Risk Classifier

	The class implements minimax risk classfication using two types of commonly used loss functions,
	namely logistic loss and 0-1 loss. The class also provides different feature mapping functions 
	that can enhance the performance of the classifier for different datasets.
		
	Parameters 
	----------
	n_classes : int
		The number of classes in the dataset used to determine the type constraints to be used
		during the optimization i.e., to use linear or nonlinear number of constraints
		to reduce the complexity.

	equality : bool, default=False
		The type of Learning. If true the LPC is asymptotically calibrated, if false the LPC is
		approximately calibrated.
		
	s : float, default=0.3 
		For tuning the estimation of expected values of feature mapping function (@phi). 
		Must be a positive float value and expected to be in the 0 to 1 in general cases.
		
	deterministic : bool, default=False
		For determining if the prediction of the labels should be done in a deterministic way or not.
		
	seed : int, default=0
		For randomization
		
	loss : {'0-1', 'log'}, default='0-1'
		The type of loss function to use for the risk minimization.

	solver : {'SCS', 'ECOS'}, default='SCS'
		The type of CVX solver to use for solving the problem. 
		In some cases, one solver might not work, 
		so we might need to use the other solver from the set.

	phi : {'gaussian', 'linear', 'threshold', 'custom'}, default='linear'
		The type of feature mapping function to use for mapping the input data.
		The 'gaussian', 'linear' and 'threshold' are predefined feature mapping.
		If the type is 'custom', 
		it means that the user has to define his own feature mapping function

	k : int, default=400
		Optional parameter required in case when 'threshold' type of feature mapping is used.
		It defines the maximum number of allowed threshold values for each dimension.

	gamma : {'scale', 'avg_ann', 'avg_ann_50', float} default = 'avg_ann_50'
		Optional parameter required in case when 'gaussian' type of feature mapping is used.
		It defines the type of heuristic to be used to 
		calculate the scaling parameter for the gaussian kernel.

	**phi_kwargs : multiple optional parameters for the corresponding feature mappings(phi).

	"""

	def __init__(self, n_classes, equality=False, s=0.3, deterministic=False, 
				seed=0, loss='0-1', solver='SCS', phi='gaussian', **phi_kwargs):

		self.r = n_classes
		self.equality = equality
		self.s = s
		self.deterministic = deterministic
		self.seed= seed
		self.loss = loss
		self.solver = solver

		if phi == 'gaussian':
			self.phi = PhiGaussian(n_classes = n_classes, **phi_kwargs)
		elif phi == 'linear':
			self.phi = PhiLinear(n_classes = n_classes)
		elif phi == 'threshold':
			self.phi = PhiThreshold(n_classes = n_classes, **phi_kwargs)
		elif isinstance(phi, Phi):
			self.phi = phi
		else:
			raise ValueError('Unexpected feature mapping type ... ')

		# Solver list available in cvxpy
		self.solvers = ['SCS', 'ECOS', 'ECOS_BB']

	def fit(self, X, Y= None, X_= None, _tau= None, _lambda= None):
		"""
		Fit the MRC model.

		Parameters
		----------
		X : array-like of shape (n_samples1, n_dimensions)
			Training instances used in the optimization.

		Y_ : array-like of shape (n_samples1/n_samples2), default = None
			Labels corresponding to the training instances 
			used in the optimization in case when the X_ is not defined i.e.,
			the instances used for calculating the estimates, required in the MRC,
			are not defined.

			These will be labels corresponding to the instances X_(if defined) used 
			for calculating the estimates for MRC.

			If the estimates used in the MRC are already given as a parameter
			to the this function, then the labels for the instances are not required
			unless and untill the threshold feature mappings are used.

			In case of the threshold feature mappings, these labels are required 
			to find the thresholds using the instance, label pair.


		X_ : array-like of shape (n_samples2, n_dimensions), default = None
			Additional Instances used to compute the estimates for the MRC
			if the estimates are not provided as input to this function. 
			Generally, these instances are expected to be large set 
			compared to the training instances used in the optimization.

		_tau : float, default=None
			The mean of the estimates 
			for the expections of the distributions in uncertainity sets

		_lambda : float, defautl=None
			The variance in the mean of the estimates 
			for the expections of the distributions in uncertainity sets

		Returns
		-------
		self : 
			Fitted estimator

		"""

		# Limit the number of training samples used in the optimization
		# Reduces the training time
		# n_max = 500

		X = check_array(X, accept_sparse=True)
		n = X.shape[0]

		# Set the type of configuration to be learnt i.e.,
		# whether to learn duplicate instances or not,
		# based on the MRC/CMRC model
		self.setLearnConfigType()

		# Check if separate instances are given to estimate lambda and tau
		# which will also be used to fit phi's hyperparameters.

		# Check if the instances are given for computing the estimates
		if X_ is not None:
			X_est = check_array(X_, accept_sparse=True)

		else:
			# Use the training samples used in optimization to compute the estimates
			X_est = X

		# # Learn the feature configurations to be used in optimization
		# # for only limited number of instances 
		# # to reduce training time for large datasets
		# if n_max < n:
		# 	# Fit and learn the feature mappings
		# 	self.phi.fit(X[:n_max, :], X_est, Y_, learn_duplicates= self.learn_duplicates)

		# else:
		# Fit the feature mappings
		self.phi.fit(X_est, Y)

		# Set the interval estimates if they are given
		# Otherwise compute the interval estimates
		if _tau is not None:
			self._tau = _tau
		else:
			self._tau= self.phi.estExp(X_est,Y)

		if _lambda is not None:
			self._lambda = _lambda
		else:
			self._lambda = (self.s * self.phi.estStd(X_est,Y))/np.sqrt(n)

		self.a = self._tau - self._lambda
		self.b = self._tau + self._lambda

		# Fit the MRC classifier
		self._minimaxRisk(X)

		return self

	def trySolvers(self, objective, constraints, mu, zhi, nu=None):
		"""
		Solves the MRC problem using different types of solvers available in CVXpy

		Parameters
		----------
		objective : cvxpy variable of float value
			Defines the minimization problem of the MRC.

		constraints : array-like of shape (n_constraints)
			Defines the constraints for the MRC optimization.

		mu : cvxpy array of shape (number of featuers in phi)
			Parameters used in the optimization problem

		zhi : cvxpy array of shape (number of featuers in phi)
			Parameters used in the optimization problem

		nu : cvxpy variable of float value, default = None
			Parameter used in the optimization problem

		Returns
		-------
		mu_ : cvxpy array of shape (number of featuers in phi)
			The value of the parameters
			corresponding to the optimum value of the objective function.

		zhi_ : cvxpy array of shape (number of featuers in phi)
			The value of the parameters 
			corresponding to the optimum value of the objective function.

		nu_ : cvxpy variable of float value, if None, not returned
			The value of the parameter 
			corresponding to the optimum value of the objective function.
			The parameter is not returned 
			in case the cvxpy variable is not defined (i.e, None) initially 
			when it is passed as the argument to this function.
		"""

		# Solve the problem
		prob = cvx.Problem(objective, constraints)
		_ = prob.solve(solver=self.solver, verbose=False)

		mu_= mu.value
		zhi_= zhi.value

		# Solving the constrained MRC problem which has only parameters mu and zhi
		if nu is not None:
			nu_= nu.value
		else:
			nu_= 0

		# if the solver could not find values of mu for the given solver
		if mu_ is None or zhi_ is None or nu_ is None:

			# try with a different solver for solution
			for s in self.solvers:
				if s != self.solver:
					# Solve the problem
					_ = prob.solve(solver=s, verbose=False)

					# Check the values
					mu_= mu.value
					zhi_= zhi.value
					if nu is not None:
						nu_= nu.value

					# Break the loop once the solution is obtained
					if mu_ is not None and zhi_ is not None and nu_ is not None:
						break

		# If no solution can be found for the optimization.
		if mu_ is None or zhi_ is None:
			raise ValueError('CVXpy solver couldn\'t find a solution .... \n \
							  The problem is ', prob.status)

		if nu is not None:
			return mu_, zhi_, nu_
		else:
			return mu_, zhi_

	def predict(self, X):
		"""
		Returns the predicted classes for X samples.

		Parameters
		----------
		X : array-like of shape (n_samples, n_dimensions)
			Test instances for which the labels are to be predicted
			by the MRC model.

		Returns
		-------
		y_pred : array-like of shape (n_samples, )
			The predicted labels corresponding to the given instances.

		"""

		X = check_array(X, accept_sparse=True)
		if self.loss == 'log':
			# In case of logistic loss function, 
			# the classification is always deterministic

			phi = self.phi.eval(X)

			# Deterministic classification
			v = np.dot(phi, self.mu)
			ind = np.argmax(v, axis=1)

		elif self.loss == '0-1':
			# In case of 0-1 loss function,
			# the classification can be done in deterministic or non-deterministic way
			# but by default, it is prefered to do it in non-deterministic way 
			# as it is designed for it.

			proba = self.predict_proba(X)

			if self.deterministic:
				ind = np.argmax(proba, axis=1)
			else:
				ind = [np.random.choice(self.r, size= 1, p=pc)[0] for pc in proba]

		return ind

