from phiUtils import *
from gammaUtils import *

from sklearn.metrics.pairwise import rbf_kernel

import itertools as it
import scipy.special as scs
import time

class Phi():
    '''
    Phi(Feature mapping) function composed by different types of features. It provides the following types of features

    -- threshold features
    -- gaussian features
    -- linear features
    -- custom features (features given by the user)

    '''

    def __init__(self, r, _type, k, gamma):
        # number of classes
        self.r= r
        # type of feature mapping. If None, then custom feature mapping is used.
        self.type = _type

        # maximum number of univariate thresholds for each dimension 
        # hyperparameter for threshold features
        self.k= k

        # scale parameter for gaussian kernels
        # hyperparameter for gaussian features
        self.gamma = gamma

        # the type of constraints used: linear or non-linear
        if self.r > 4:
            self.linConstr= False
        else:
            self.linConstr = True

        self.is_fitted_ = False
        return

    def fit(self, X, y, learn_config=True):
        '''
        Learn the set of features for the given type. 
        By setting the learn_config as false, it can be used to transform the instances into features.

        @param X: unlabeled training instances / the custom features given by the user when type is None
        @param y: training class labels
        @param learn_config: learn the configurations of phi for constraints if true, 
        @return: None
        '''

        n, d= X.shape

        # Store the instances to be used for computing the gaussian kernel
        if self.type == 'gaussian':
            self.uniqX = np.unique(X, axis=0)

            # Evaluate the gamma according to the gamma type given in self.gamma
            if self.gamma == 'scale':
                self.gamma_val = 1 / (d*X.var())

            elif self.gamma == 'avg_ann':
                self.gamma_val = heuristicGamma(X, y, self.r)

            elif self.gamma == 'avg_ann_50':
                self.gamma_val = rffGamma(X)

            elif type(self.gamma) != str:
                self.gamma_val = self.gamma

            else:
                raiseValueError('Unexpected value for gamma ...')

        #Learn the product thresholds
        elif self.type == 'threshold':
            self.thrsDim, self.thrsVal = decTreeSplit(X, y, self.k)

        # Defining the length of the phi based on the features
        self.m = self.transform(X).shape[1]+1
        self.len = self.m*self.r

        # Learn the configurations of phi using the training
        if learn_config:
            self.learnF(X)

        self.is_fitted_ = True
        return

    def learnF(self, X):
        '''
        Stores all the unique configurations of x in X for every value of y

        @param X_feat: Features obtained from the instances X
        @return: None
        '''

        n= X.shape[0]
        phi= self.eval(X)

        # Disctinct configurations for phi_x,y for x in X and y=1,...,r.
        # Used in the definition of the constraints of the MRC
        # F is a tuple of floats with dimension n_intances X n_classes X m
        self.F= np.vstack({tuple(phi_xy) for phi_xy in phi.reshape((n,self.r*self.len))})
        self.F.shape = (self.F.shape[0], self.r, int(self.F.shape[1] / self.r))

        return

    def transform(self, X):

        '''
        Transform the given instances to the corresponding features

        @param X: unlabeled training instances / the custom features given by the user when type is None
        @param y: training class labels
        @return: Feature obtained from the instances X
        '''

        # Customized features defined by the user when type is None and
        # the matrix X is expected to contain the features instead of the instances
        if self.type == 'custom' or self.type == 'linear':
            X_feat = X

        # Features defined according to the type chosen 
        # from the list of features provided by this library as given at the top
        elif self.type == 'threshold':
            X_feat = thrsFeat(X, self.thrsDim, self.thrsVal)  

        elif self.type == 'gaussian':
            X_feat = rbf_kernel(X, self.uniqX, gamma=self.gamma_val)

        else:
            raiseValueError('Unexpected value for the type of feature mapping to use ...')

        return X_feat

    def eval(self, X):
        '''
        The optimized evaluation of the instances X, phi(x,y) for all x in X and y=0,...,r-1

        @param X_feat: features obtained from the instances
        @return: evaluation of the set of instances for all class labels.
            np.array(float), (n_instances X n_classes X phi.len)
        '''

        n = X.shape[0]

        # Get the features
        X_feat = self.transform(X)

        # Defining the length of the phi based on the features
        m = X_feat.shape[1]+1
        phi_len = m*self.r

        # product threshold values
        # [[p11,...,p1m],...,[p11,...,p1m],...,[pn1,...pnm],...,[pn1,...pnm]] where pij is the j-th prod theshold
        # for i-th unlabeled instance, r*n_samples X phi.len
        phi = np.zeros((n, self.r, phi_len), dtype=np.float)

        # adding the intercept
        phi[:, np.arange(self.r), np.arange(self.r)*m] = \
                np.tile(np.ones(n), (self.r, 1)).transpose()

        # Compute the phi function
        for dimInd in range(1, m):
            phi[:, np.arange(self.r), np.arange(self.r) * m + dimInd] = \
                np.tile(X_feat[:, dimInd-1], (self.r, 1)).transpose()

        return phi

    def evaluate(self, X, Y):
        '''
        Evaluation of a labeled set of instances (X,Y), phi(x,y) for (x,y) in (X,Y)

        Used in the learning stage for estimating the expected value of phi, tau

        @param X_feat: features obtained from the instances
        @param Y: np.array(numInstances)
        @return: The evaluation of phi the the set of instances (X,Y),
            np.array(int) with dimension n_instances X (n_classes * n_prod_thresholds)
        '''

        n = X.shape[0]

        # Get the features
        X_feat = self.transform(X)

        # Defining the length of the phi based on the features
        m = X_feat.shape[1]+1
        phi_len = m*self.r

        # product threshold values
        # [[p11,...,p1m],...,[p11,...,p1m],...,[pn1,...pnm],...,[pn1,...pnm]] where pij is the j-th prod theshold
        # for i-th unlabeled instance, r*n_samples X phi.len
        phi = np.zeros((n, phi_len), dtype=np.float)

        # adding the intercept
        phi[np.arange(n), Y * m] = np.ones(n)

        # Compute the phi function
        for dimInd in range(1, m):
            phi[np.arange(n), dimInd + Y * m] = X_feat[:, dimInd-1]

        return phi

    def estExp(self, X, Y):
        '''
        Average value of phi in the supervised dataset (X,Y)
        Used in the learning stage as an estimate of the expected value of phi, tau

        @param X: the set of unlabeled instances or the custom features of the users when type is None
        @param Y: np.array(numInstances)
        @return: Average value of phi, np.array(float) phi.len.
        '''

        return np.average(self.evaluate(X, Y), axis= 0)

    def estStd(self, X, Y):
        '''
        Standard deviation of phi in the supervised dataset (X,Y)
        Used in the learning stage to estimate the bounds, a and b

        @param X: the set of unlabeled instances or the custom features of the users when type is None
        @param Y: np.array(numInstances)
        @return: standard deviation of phi, np.array(float) phi.len.
        '''

        return np.std(self.evaluate(X, Y), axis= 0)

    def getLearnConstr(self, linConstr):
        '''
        Get the constraints required for determining the uncertainty set using phi with liner probabilistic
        classifiers, LPC.

        @return: The index of the variables that have to be added for creating the constraints of for learning
        the LPC. Two type of constraints: 1.exponential and 2:linear

        FORMAT:
        1.-Exponential: For each x with different phi_x average, value of F_x over every subset of the class values.
        The last row corresponds to the number of class values selected for averaging F_x. Returns a
        np.matrix(float), (n_instances * 2^r-1) X (num_classes * num_prod_feats + 1)
        '''

        n= self.F.shape[0]
        if linConstr:#self.r<4:
            #Linear constraints. Exponential number in r
            avgF= np.vstack((np.sum(self.F[:, S, ], axis=1)
                             for numVals in range(1, self.r+1)
                             for S in it.combinations(np.arange(self.r), numVals)))
            cardS= np.arange(1, self.r+1).repeat([n*scs.comb(self.r, numVals)
                                                 for numVals in np.arange(1, self.r+1)])[:, np.newaxis]

            constr= np.hstack((avgF, cardS))
        else:
            #Non-linear constraints (defined used the positive part). The number is independent from r
            constr= self.F

        return constr

    def getLowerConstr(self):
        '''
        Get the constraints required for determining the uncertainty set using phi with liner probabilistic
        classifiers, LPC.

        @return: The index of the variables that have to be added for creating the constraints of for learning
        the LPC. Two type of constraints: 1.exponential and 2:linear

        FORMAT:
        1.-Exponential: For each x with different phi_x average, value of F_x over every subset of the class values.
        The last row corresponds to the number of class values selected for averaging F_x. Returns a
        np.matrix(float), (n_instances * 2^r-1) X (num_classes * num_prod_feats + 1)
        '''

        constr= self.F

        return constr
