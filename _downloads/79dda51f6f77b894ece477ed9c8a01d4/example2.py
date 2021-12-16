# -*- coding: utf-8 -*-
"""
.. _ex2:

Example: Use of CMRC with different settings
============================================

Example of using CMRC with some of the common classification datasets with different
losses and feature mappings settings. We load the different datasets and use 10-Fold 
Cross-Validation to generate the partitions for train and test. We separate 1 partition
each time for testing and use the others for training. On each iteration we calculate 
the classification error. We also calculate the mean training time.

You can check a more elaborated example in :ref:`ex_comp`.
"""

import time

import numpy as np
from sklearn import preprocessing
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

    res_mean = np.zeros(len(dataName))
    res_std = np.zeros(len(dataName))

    # We fix the random seed to that the stratified kfold performed
    # is the same through the different executions
    random_seed = 0

    # Iterate through each of the dataset and fit the CMRC classfier.
    for j, load in enumerate(loaders):

        # Loading the dataset
        X, Y = load(return_X_y=True)
        r = len(np.unique(Y))
        n, d = X.shape

        # Print the dataset name
        print(" ############## \n " + dataName[j] + " n= " + str(n) +
              " , d= " + str(d) + ", cardY= " + str(r))

        # Create the CMRC object initilized with the corresponding parameters
        clf = CMRC(phi=phi, loss=loss, use_cvx=True,
                   solver='MOSEK', max_iters=10000, s=0.3)

        # Generate the partitions of the stratified cross-validation
        cv = StratifiedKFold(n_splits=10, random_state=random_seed,
                             shuffle=True)

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

            # Train the model
            clf.fit(X_train, y_train)

            # Save the training time
            auxTime += time.time() - startTime
            
            # Predict the class for test instances
            y_pred = clf.predict(X_test)

            # Calculate the error made by CMRC classificator
            cvError.append(np.average(y_pred != y_test))

        res_mean[j] = np.average(cvError)
        res_std[j] = np.std(cvError)

        # Calculating the mean training time
        auxTime = auxTime / 10

        print(" error= " + ": " + str(res_mean[j]) + " +/- " +
              str(res_std[j]) + "\n avg_train_time= " + ": " +
              str(auxTime) + ' secs' + "\n ############## \n\n")


if __name__ == '__main__':

    print('*** Example (CMRC with the additional marginal constraints) *** \n\n')

    print('1. Using 0-1 loss and relu feature mapping \n\n')
    runCMRC(phi='relu', loss='0-1')

    print('2. Using log loss and relu feature mapping \n\n')
    runCMRC(phi='relu', loss='log')
