# -*- coding: utf-8 -*-
"""

.. _ex2:

Example: Use of CMRC with different settings
============================================

Example of using CMRC with some of the common classification datasets with
different losses and feature mappings settings. We load the different datasets
and use 10-Fold Cross-Validation to generate the partitions for train and test.
We separate 1 partition each time for testing and use the others for training.
On each iteration we calculate
the classification error. We also calculate the mean training time.

Note that we set the parameter use_cvx=False. In the case of CMRC classifiers
and random fourier feature mapping this means that we will use Stochastic
Gradient Descent (SGD) approach to perform the optimization.

You can check a more elaborated example in :ref:`ex_comp`.

"""

import time

import numpy as np
import pandas as pd
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
    results = pd.DataFrame()

    # We fix the random seed to that the stratified kfold performed
    # is the same through the different executions
    random_seed = 0

    # Iterate through each of the dataset and fit the CMRC classfier.
    for j, load in enumerate(loaders):

        # Loading the dataset
        X, Y = load()
        r = len(np.unique(Y))
        n, d = X.shape

        # Create the CMRC object initilized with the corresponding parameters
        clf = CMRC(phi=phi,
                   loss=loss,
                   random_state=random_seed,
                   solver='adam')

        # Generate the partitions of the stratified cross-validation
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_seed,
                             shuffle=True)

        cvError = list()
        upper = 0
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
            upper += clf.get_upper_bound()

            # Save the training time
            auxTime += time.time() - startTime

            # Predict the class for test instances
            y_pred = clf.predict(X_test)

            # Calculate the error made by CMRC classificator
            cvError.append(np.average(y_pred != y_test))

        upper = upper / n_splits
        res_mean = np.average(cvError)
        res_std = np.std(cvError)

        # Calculating the mean training time
        auxTime = auxTime / n_splits

        results = results._append({'dataset': dataName[j],
                                  'n_samples': '%d' % n,
                                  'n_attributes': '%d' % d,
                                  'n_classes': '%d' % r,
                                  "upper": "%1.2g" % upper,
                                  'error': '%1.2g' % res_mean + " +/- " +
                                  '%1.2g' % res_std,
                                  'avg_train_time (s)': '%1.2g' % auxTime},
                                 ignore_index=True)

    return results


####################################################################

r1 = runCMRC(phi='fourier', loss='0-1')
r1.style.set_caption('Using 0-1 loss and fourier feature mapping')

####################################################################

r2 = runCMRC(phi='fourier', loss='log')
r2.style.set_caption('Using log loss and fourier feature mapping')
