# -*- coding: utf-8 -*-
"""

.. _exdwgcs:

Example: Use of DWGCS (Double-Weighting General Covariate Shift) for Covariate Shift Adaptation
============================================

Example of using DWGCS with the sythetic dataset used on the experiments 
of the corresponding paper. We load the dataset, train the DWGCS model using
data collected from training and testing distribution and predict the new instances
from the testing distribution.

"""

import numpy as np
from sklearn import preprocessing
import pandas as pd

from MRCpy import DWGCS
# Import the datasets
from MRCpy.datasets import *

# Data sets
loaders = [load_comp_vs_sci, load_comp_vs_talk, load_rec_vs_sci,
          load_rec_vs_talk, load_sci_vs_talk]
dataName = ["comp-vs-sci", "comp-vs-talk", "rec-vs-sci",
           "rec-vs-talk", "sci-vs-talk"]

rep = 2
n = 1000
t = 1000

def runCMRC(phi, loss):
    columns = ['dataset', 'n_samples', 'n_attributes', 'n_classes', 'upper', 'error']
    results = pd.DataFrame(columns=columns)

    for j, load in enumerate(loaders):

        # Loading the dataset
        X_TrainSet, Y_TrainSet, X_TestSet, Y_TestSet = load()

        r = len(np.unique(Y_TrainSet))
        d = X_TrainSet.shape[1]

        TrainSet = np.concatenate((X_TrainSet, np.reshape(Y_TrainSet,(Y_TrainSet.shape[0], 1))), axis=1)
        TestSet = np.concatenate((X_TestSet, np.reshape(Y_TestSet, (Y_TestSet.shape[0], 1))), axis=1)

        Error = list()
        upper = 0
        
        for i in range(rep):

            np.random.shuffle(TrainSet)
            np.random.shuffle(TestSet)

            X_train = TrainSet[:n, :-1]
            Y_train = TrainSet[:n, -1]
            X_test  = TestSet[:t, :-1]
            Y_test  = TestSet[:t, -1]

            x = np.concatenate((X_train, X_test), axis=0)
            x = preprocessing.StandardScaler().fit_transform(x)   
            X_train = x[:n, :]
            X_test = x[n:, :]

            clf = DWGCS(loss = loss, phi = phi, deterministic = True)
            clf.fit(X_train, Y_train, X_test)
            upper += clf.get_upper_bound()

            Y_pred = clf.predict(X_test)
            Error.append(np.average(Y_pred != Y_test))
        
        upper = upper / rep
        res_mean = np.average(Error)
        res_std = np.std(Error)

        new_row = {'dataset': dataName[j],
                                  'n_samples': '%d' % n,
                                  'n_attributes': '%d' % d,
                                  'n_classes': '%d' % r,
                                  "upper": "%1.2g" % upper,
                                  'error': '%1.2g' % res_mean + " +/- " +
                                  '%1.2g' % res_std}
        results.loc[len(results)] = new_row
        
    return results

####################################################################

r1 = runCMRC(phi='linear', loss='0-1')
r1.style.set_caption('Using 0-1 loss and linear feature mapping')

####################################################################

r2 = runCMRC(phi='linear', loss='log')
r2.style.set_caption('Using log loss and linear feature mapping')