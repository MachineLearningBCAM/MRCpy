import numpy as np
from sklearn.tree import DecisionTreeClassifier

def decTreeSplit(X, Y, k=None):
    '''
    Learn the univariate thresholds by using the split points of decision trees for each dimension of data

    @param X: unlabeled training samples
    @param y: labels of the training samples
    @param k: maximum number of leaves of the tree
    @return: the univariate thresholds
    '''

    (n, d) = X.shape

    prodThrsVal = []
    prodThrsDim = []

    # One order thresholds: all the univariate thresholds
    for dim in range(d):
        if k== None:
            dt = DecisionTreeClassifier()
        else:
            dt= DecisionTreeClassifier(max_leaf_nodes=k+1)

        dt.fit(np.reshape(X[:,dim],(n,1)),Y)

        dimThrsVal= np.sort(dt.tree_.threshold[dt.tree_.threshold!= -2])

        for t in dimThrsVal:
            prodThrsVal.append([t])
            prodThrsDim.append([dim])

    return prodThrsDim, prodThrsVal

def thrsFeat(X, thrsDim, thrsVal):
    '''
    Find the features of the given instances based on the 
    thresholds obtained using the instances

    @param thrsDim: dimension of univariate thresholds given in the form of array of arrays like - [[0], [1], ...]
    @param thrsVal: value of the univariate thresholds given in the form of array of arrays like - [[0.5], [0.7], ...]
    @return: the 0-1 features developed using the thresholds
    '''

    n = X.shape[0]

    # Store the features based on the thresholds obtained
    X_feat = np.zeros((n, len(thrsDim)), dtype=int)

    # Calculate the threshold features
    for thrsInd in range(len(thrsDim)):
        X_feat[:, thrsInd] = np.all(X[:, thrsDim[thrsInd]] <= thrsVal[thrsInd],
                                axis=1).astype(np.int)

    return X_feat

