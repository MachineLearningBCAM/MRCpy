# -*- coding: utf-8 -*-
"""
.. _grid:

Hyperparameter Tuning: Upper Bound vs Cross-Validation
==============================================================================

Example of how to use the Upper Bounds provided by the `MRC` method in the
`MRCpy` library for hyperparameter tuning and comparison to Cross-Validation.
We will see that using the Upper Bound gets similar performances to
Cross-Validation but being four times faster.

We are using '0-1' loss and `RandomFourierPhi`
map (`phi='fourier'`). We are going to tune the scaling parameter
`sigma` and the regularization parameter `s` of the
feature mapping using a random grid. We will used the usual method
:ref:`RandomizedSearchCV<https://scikit-learn.org/stable/modules/
generated/sklearn.model_selection.RandomizedSearchCV.html>`
from `scikit-learn`.

Note that we set the parameter use_cvx=False. In the case of MRC classifiers
this means that we will use nesterov subgradient optimized approach to
perform the optimization.
"""

# Import needed modules
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from MRCpy import MRC
from MRCpy.datasets import *


############################################################################
# Random Grid using Upper Bound parameter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We select random `n_iter` random set of values for the parameters to tune in
# a given range and select the pair of parameters which minimizes the upper
# bound provided by the MRC method.
# On each repetition we calculate and store the upper bound for each possible
# value of sigma.
# The parameter `n_iter` means the amount of randomly selected vectors for the
# parameters to
# tune are chosen. We are selecting `n_iter = 10` because it is the default
# value for the RandomGridCV method.


def run_RandomGridUpper(X_train, Y_train, X_test, Y_test, sigma_ini, sigma_fin,
                        s_ini, s_fin):
    n_iter = 10
    startTime = time.time()
    sigma_id = [(sigma_fin - sigma_ini) * random.random() + sigma_ini
                for i in range(n_iter)]
    s_id = [(s_fin - s_ini) * random.random() + s_ini for i in range(n_iter)]
    upps = np.zeros(n_iter)

    for i in range(n_iter):
        clf = MRC(phi='fourier', sigma=sigma_id[i], s=s_id[i], random_state=0,
                  deterministic=False, use_cvx=False)
        clf.fit(X_train, Y_train)
        upps[i] = clf.get_upper_bound()

    min_upp = np.min(upps)
    best_sigma = sigma_id[np.argmin(upps)]
    best_s = s_id[np.argmin(upps)]
    clf = MRC(phi='fourier', sigma=best_sigma, s=best_s, random_state=0,
              deterministic=False, use_cvx=False)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    best_err = np.average(Y_pred != Y_test)
    totalTime = time.time() - startTime

    return {'upper': min_upp, 's': best_s,
            'sigma': best_sigma, 'time': totalTime, 'error': best_err}


############################################################################
# RandomGridCV
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def run_RandomGridCV(X_train, Y_train, X_test, Y_test, sigma_ini, sigma_fin,
                     s_ini, s_fin):
    n_iter = 10
    startTime = time.time()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
                                                        random_state=rep)
    # Normalizing the data
    std_scale = preprocessing.StandardScaler().fit(X_train, Y_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    sigma_values = np.linspace(sigma_ini, sigma_fin, num=5000)
    s_values = np.linspace(s_ini, s_fin, num=5000)
    param = {'sigma': sigma_values, 's': s_values}

    mrc = MRC(phi='fourier', random_state=0, deterministic=False,
              use_cvx=False)
    clf = RandomizedSearchCV(mrc, param, random_state=0, n_iter=n_iter)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    error = np.average(Y_pred != Y_test)

    totalTime = time.time() - startTime

    return {'upper': clf.best_estimator_.get_upper_bound(),
            's': clf.best_estimator_.s,
            'sigma': clf.best_estimator_.phi.sigma_val,
            'time': totalTime, 'error': error}


###################################
# Comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We are performing both of the previous methods for hyperparameter tuning
# over a set of different datasets and comparing the performances.
# Before calling them, we set a range of values for the hyperpatameters.
# An intuituve way of choosing sigma is to choose values in the range of the
# distance among the pairs of instances in the trainign set `X_train`.
# Empirical knowledge tells us that best values for s use to be around
# 0.3 and 0.6.
#
# We repeat these processes several times to make sure performances do not
# rely heavily on the train_test_split selected.


def plot_table(df, title, color):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    t = ax.table(cellText=df.values, colLabels=df.columns, loc='center',
                 colColours=color, cellColours=[color] * len(df))
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(col=list(range(len(df.columns))))
    fig.tight_layout()
    plt.title(title)
    plt.show()


loaders = [load_mammographic, load_haberman, load_indian_liver,
           load_diabetes, load_credit]
dataNameList = ["mammographic", "haberman", "indian_liver",
                "diabetes", "credit"]

dfCV = pd.DataFrame()
dfUpper = pd.DataFrame()
f = '%1.3g'  # format
for j, load in enumerate(loaders):

    # Loading the dataset
    X, Y = load()
    dataName = dataNameList[j]

    # In order to avoid the possible bias made by the choice of the train-test
    # split, we do this process several (20) times and average the
    # obtained results
    dfCV_aux = pd.DataFrame()
    dfUpper_aux = pd.DataFrame()
    for rep in range(10):
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=0.25, random_state=rep)
        # Normalizing the data
        std_scale = preprocessing.StandardScaler().fit(X_train, Y_train)
        X_train = std_scale.transform(X_train)
        X_test = std_scale.transform(X_test)

        # Select an appropiate range for sigma
        d = np.triu(distance.cdist(X_train, X_train)).flatten()
        d = d[d != 0]
        d.sort()
        sigma_ini = d[int(len(d) * 0.1)]
        sigma_fin = d[int(len(d) * 0.3)]
        s_ini = 0.3
        s_fin = 0.6

        # We tune the parameters using both method and store the results
        dfCV_aux = dfCV_aux.append(
            run_RandomGridCV(X_train, Y_train, X_test, Y_test, sigma_ini,
                             sigma_fin, s_ini, s_fin), ignore_index=True)
        dfUpper_aux = dfUpper_aux.append(
            run_RandomGridUpper(X_train, Y_train, X_test, Y_test, sigma_ini,
                                sigma_fin, s_ini, s_fin), ignore_index=True)

    # We save the mean results of the 20 repetitions
    mean_err = f % np.mean(dfCV_aux['error']) + ' ± ' + \
        f % np.std(dfCV_aux['error'])
    mean_sig = f % np.mean(dfCV_aux['sigma']) + ' ± ' + \
        f % np.std(dfCV_aux['sigma'])
    mean_s = f % np.mean(dfCV_aux['s']) + ' ± ' + f % np.std(dfCV_aux['s'])
    mean_time = f % np.mean(dfCV_aux['time']) + ' ± ' + \
        f % np.std(dfCV_aux['time'])
    mean_upper = f % np.mean(dfCV_aux['upper']) + ' ± ' + \
        f % np.std(dfCV_aux['upper'])
    dfCV = dfCV.append({'dataset': dataName, 'error': mean_err,
                        'sigma': mean_sig, 's': mean_s,
                        'upper': mean_upper,
                        'time': mean_time}, ignore_index=True)
    mean_err = f % np.mean(dfUpper_aux['error']) + ' ± ' + \
        f % np.std(dfUpper_aux['error'])
    mean_sig = f % np.mean(dfUpper_aux['sigma']) + ' ± ' + \
        f % np.std(dfUpper_aux['sigma'])
    mean_s = f % np.mean(dfUpper_aux['s']) + ' ± ' + \
        f % np.std(dfUpper_aux['s'])
    mean_time = f % np.mean(dfUpper_aux['time']) + ' ± ' + \
        f % np.std(dfUpper_aux['time'])
    mean_upper = f % np.mean(dfUpper_aux['upper']) + ' ± ' + \
        f % np.std(dfUpper_aux['upper'])
    dfUpper = dfUpper.append({'dataset': dataName, 'error': mean_err,
                              'sigma': mean_sig, 's': mean_s,
                              'upper': mean_upper,
                              'time': mean_time}, ignore_index=True)

######################

dfCV.style.set_caption('RandomGridCV Results').set_properties(
    **{'background-color': 'lightskyblue'}, subset=['error', 'time'])

######################

dfUpper.style.set_caption('RandomGridUpper Results').set_properties(
    **{'background-color': 'lightskyblue'}, subset=['error', 'time'])

######################################
# Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Comparing the resulting tables above we notice that both methods:
# RandomGridCV and Random Grid using Upper bounds are really similar in
# performance, one can do better than the other depending on the datasets but
# have overall the same error range.
#
# Furthermore we can see how using the Upper bounds results in a great
# improvement in the running time being around 4 times quicker than
# the usual RandomGrid method.
#
# We note that in every dataset the optimum value for the parameter s seems
# to be  always around 0.3, that is why this value has been chosen to be
# the default value for the library.
