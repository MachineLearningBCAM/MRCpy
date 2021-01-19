'''
	Super class for Minimax Risk Classifier
'''

import numpy as np
import cvxpy as cvx
import time

# Import the feature mapping
from phi import Phi

class _MRC_():

	"""
	Minimax Risk Classifier

	The class implements minimax risk classfication using two types of commonly used loss functions,
	namely logistic loss and 0-1 loss. The class also provides different feature mapping functions 
	that can enhance the performance of the classifier for different datasets.
		
	Parameters 
	----------
	r : int
		The number of classes in the dataset used to determine the type constraints to be used
		during the optimization i.e., to use linear or nonlinear number of constraints
		to reduce the complexity.

	equality : bool, default=False
		The type of Learning. If true the LPC is asymptotically calibrated, if false the LPC is
		approximately calibrated.

	_tau : float, default=None
		The mean of the estimates for the expections of the distributions in uncertainity sets

	_lambda : float, defautl=None
		The variance in the mean of the estimates for the expections of the distributions in uncertainity sets
		
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

	phi : {'gaussian', 'linear', 'threshold', 'custom'}, default='gaussian'
		The type of feature mapping function to use for mapping the input data.
		The 'gaussian', 'linear' and 'threshold' are predefined feature mapping.
		The type is 'custom', it means that the user has to define his own feature mapping.

	k : int, default=400
		Optional parameter required in case when 'threshold' type of feature mapping is used.
		It defines the maximum number of allowed threshold values for each dimension.

	gamma : {'scale', 'avg_ann', 'avg_ann_50', float} default = 'avg_ann_50'
		Optional parameter required in case when 'gaussian' type of feature mapping is used.
		It defines the type of heuristic to be used to 
		calculate the scaling parameter for the gaussian kernel.

	"""

	def __init__(self, r, equality=False, s=0.3, deterministic=False, seed=0, loss='0-1', 
					solver='SCS', phi='threshold', k=300, gamma='avg_ann_50'):

		self.r = r
		self.equality = equality
		self.s = s
		self.deterministic = deterministic
		self.seed= seed
		self.loss = loss
		self.solver = solver

		if self.r> 4:
			self.linConstr= False
		else:
			self.linConstr= True

		# Define the feature mapping
		self.phi = Phi(r=self.r, _type=phi, k=k, gamma=gamma)

		# solver list available in cvxpy
		self.solvers = ['SCS', 'ECOS']

	def fit(self, X, Y, X_ = None, Y_ = None, _tau = None, _lambda = None):
		"""
		Fit learning using....

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)

		y : array-like, shape (n_samples)

		Returns
		-------
		self : Returns the fitted estimator

		"""

		self.phi.linConstr= self.linConstr
		self.phi.fit(X, Y)

		# Set the interval estimates
		if _tau is None or _lambda is None:
			if X_ is None or Y_ is None:
				self.setEstimates(X, Y)
			else:
				self.setEstimates(X_, Y_)
		else:
			self._tau = _tau
			self._lambda = _lambda

			self.a = self._tau - self._lambda
			self.b = self._tau + self._lambda

		self._minimaxRisk(X,Y)

		return self

	def setEstimates(self, X, Y):
		"""
		Set the estimates for the uncertainity sets

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)

		y : array-like, shape (n_samples)

		"""

		n = X.shape[0]

		self._tau= self.phi.estExp(X,Y)
		self._lambda= (self.s * self.phi.estStd(X,Y))/np.sqrt(n)

		self.a= self._tau- self._lambda
		self.b= self._tau+ self._lambda

	def trySolvers(self, objective, constraints, mu, zhi, nu=None):
		"""
		Solves the MRC problem using different types of solvers available in CVXpy

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

		if nu is not None:
			return mu_, zhi_, nu_
		else:
			return mu_, zhi_

	def predict(self, X):
		"""
		Returns the predicted classes for X samples.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)

		returns
		-------
		y_pred : array-like, shape (n_samples, )
			y_pred is of the same type as self.classes_.

		"""

		if self.loss == 'log':
			# In case of logistic loss function, 
			# the classification is always deterministic

			Phi = self.phi.eval(X)

			# Deterministic classification
			v = np.dot(Phi, self.mu)
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