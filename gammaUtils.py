import numpy as np
import statistics
import random
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

def heuristicGamma(X_train, y_train, r):

	'''
		Function to find the scale parameter for gaussian kernels using the heuristic - 

					sigma = median{ {min ||x_i-x_j|| for j|y_j != +1 } for i|y_i != -1 } 

		for two classes {+1, -1}. For multi-class, we use the same strategy by finding median 
		of min norm value for each class against all other classes and then taking the average value
		of the median of all the classes.

		NOTE: The heuristic calculates the value of sigma for gaussian kernels. So to find the 
			  gamma value, the following formula is used in our implementation- 
			  					gamma = 1/ (2 * sigma^2)
	'''

	# List to store the median of min norm value for each class
	dist_x_i = list()

	for i in range(r):
		x_i = X_train[y_train == i,:]
		x_not_i = X_train[y_train != i,:]

		# find the distance of each point of this class with every
		# other point of other class
		norm_vec = np.linalg.norm(np.tile(x_not_i,(x_i.shape[0], 1)) - \
			np.repeat(x_i, x_not_i.shape[0], axis=0), axis=1)
		dist_mat = np.reshape(norm_vec, (x_not_i.shape[0], x_i.shape[0]))

		# find the min distance for each point and take the median distance
		minDist_x_i = np.min(dist_mat, axis=1)
		dist_x_i.append(statistics.median(minDist_x_i))

	sigma = np.average(dist_x_i)

	# Evaluate gamma
	gamma = 1/(2 * sigma * sigma)

	return gamma

def rffGamma(X):

	'''
		Function to find the scale parameter for random fourier features obtained from 
		gaussian kernels using the heuristic given in - 
					
					"Compact Nonlinear Maps and Circulant Extensions"

		The heuristic to calculate the sigma states that it is a value that is obtained from
		the average distance to the 50th nearest neighbour estimated from 1000 samples of the dataset.

		Gamma value is given by - 

					gamma = 1/ (2 * sigma^2)
	'''

	# Number of training samples
	n = X.shape[0]

	neighbour_ind = 50

	# Find the nearest neighbors
	nbrs = NearestNeighbors(n_neighbors=(neighbour_ind+1), algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)

	# Compute the average distance to the 50th nearest neighbour
	sigma = np.average(distances[:, neighbour_ind])

	return 1/ (2 * sigma * sigma)

