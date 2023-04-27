# -*- coding: utf-8 -*-
"""

.. _ex1:

Example: Use of MRC with different settings
===========================================

Example of using MRC with some of the common classification datasets with
different losses and feature mappings settings. We load the different datasets
and use 10-Fold Cross-Validation to generate the partitions for train and test.
We separate 1 partition each time for testing and use the others for training.
On each iteration we calculate the classification error as well as the upper
and lower bounds for the error. We also calculate the mean training time.

Note that we set the parameter use_cvx=False. In the case of MRC classifiers
this means that we will use nesterov subgradient optimized approach to
perform the optimization.

You can check a more elaborated example in :ref:`ex_comp`.

"""

import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from MRCpy import MRC
# Import the datasets
from MRCpy.datasets import *

# Data sets
loaders = [
    load_mammographic,
    load_haberman,
    load_indian_liver,
    load_diabetes,
    load_credit,
]
dataName = ["mammographic", "haberman", "indian_liver", "diabetes", "credit"]


def runMRC(phi, loss):

    results = pd.DataFrame()
    # We fix the random seed to that the stratified kfold performed
    # is the same through the different executions
    random_seed = 0

    # Iterate through each of the dataset and fit the MRC classfier.
    for j, load in enumerate(loaders):

        # Loading the dataset
        X, Y = load()
        r = len(np.unique(Y))
        n, d = X.shape

        clf = MRC(phi=phi,
                  loss=loss,
                  random_state=random_seed,
                  max_iters=5000,
                  solver='subgrad')

        # Generate the partitions of the stratified cross-validation
        n_splits = 5
        cv = StratifiedKFold(
            n_splits=n_splits, random_state=random_seed, shuffle=True
        )

        cvError = list()
        auxTime = 0
        upper = 0
        lower = 0

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

            # Train the model and save the upper and lower bounds
            clf.fit(X_train, y_train)
            upper += clf.get_upper_bound()
            lower += clf.get_lower_bound()

            # Save the training time
            auxTime += time.time() - startTime

            # Predict the class for test instances
            y_pred = clf.predict(X_test)

            # Calculate the error made by MRC classificator
            cvError.append(np.average(y_pred != y_test))

        res_mean = np.average(cvError)
        res_std = np.std(cvError)

        # Calculating the mean upper and lower bound and training time
        upper = upper / n_splits
        lower = lower / n_splits
        auxTime = auxTime / n_splits

        results = results._append(
            {
                "dataset": dataName[j],
                "n_samples": "%d" % n,
                "n_attributes": "%d" % d,
                "n_classes": "%d" % r,
                "error": "%1.2g" % res_mean + " +/- " + "%1.2g" % res_std,
                "upper": "%1.2g" % upper,
                "lower": "%1.2g" % lower,
                "avg_train_time (s)": "%1.2g" % auxTime,
            },
            ignore_index=True,
        )
    return results


####################################################################

r1 = runMRC(phi="fourier", loss="0-1")
r1.style.set_caption("Using 0-1 loss and fourier feature mapping")

####################################################################

r2 = runMRC(phi="fourier", loss="log")
r2.style.set_caption("Using log loss and fourier feature mapping")
