"""Example of using CMRC with some of the common classification datasets."""

import time

import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from MRCpy import CMRC
# Import the datasets
from MRCpy.datasets import *

# Data sets
loaders = [load_mammographic, load_haberman, load_indian_liver,
           load_diabetes, load_credit]
dataName = ["mammographic", "haberman", "indian_liver",
            "diabetes", "credit"]


def runCMRC(phi, loss):

    random_seed = 0
    res_mean = np.zeros(len(dataName))
    res_std = np.zeros(len(dataName))
    np.random.seed(random_seed)

    # Iterate through each of the dataset and fit the MRC classfier.
    for j, load in enumerate(loaders):

        # Loading the dataset
        X, Y = load(return_X_y=True)
        r = len(np.unique(Y))
        n, d = X.shape

        # Print the dataset name
        print(" ############## \n" + dataName[j] + " n= " + str(n) +
              " , d= " + str(d) + ", cardY= " + str(r))

        clf = CMRC(n_classes=r, phi=phi, loss=loss, max_iters=2000, s=0.5)

        # Preprocess
        trans = SimpleImputer(strategy='median')
        X = trans.fit_transform(X, Y)

        # Generate the partitions of the stratified cross-validation
        cv = StratifiedKFold(n_splits=10, random_state=random_seed,
                             shuffle=True)

        np.random.seed(random_seed)
        cvError = list()
        auxTime = 0

        # Paired and stratified cross-validation
        for train_index, test_index in cv.split(X, Y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Normalizing the data
            std_scale = preprocessing.StandardScaler().fit(X_train, y_train)
            X_train = std_scale.transform(X_train)
            X_test = std_scale.transform(X_test)

            # Save start time for computing training time
            startTime = time.time()

            clf.fit(X_train, y_train)

            # Calculate the training time
            auxTime += time.time() - startTime

            y_pred = clf.predict(X_test)

            cvError.append(np.average(y_pred != y_test))

        res_mean[j] = np.average(cvError)
        res_std[j] = np.std(cvError)

        print(" error= " + ":\t" + str(res_mean[j]) + "\t+/-\t" +
              str(res_std[j]) + "\navg_train_time= " + ":\t" +
              str(auxTime / 10) + ' secs' + "\n ############## \n\n\n")


if __name__ == '__main__':

    print('******************** \
          Example 2 (CMRC with the additional marginal constraints) \
          ********************** \n\n')

    print('\t\t 1. Using 0-1 loss and threshold feature mapping \n\n')
    runCMRC(phi='threshold', loss='0-1')

    print('\t\t 2. Using log loss and threshold feature mapping \n\n')
    runCMRC(phi='threshold', loss='log')
