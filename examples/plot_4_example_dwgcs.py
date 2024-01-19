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
import pandas as pd
import time

from MRCpy import DWGCS
from MRCpy import CMRC
# Import the datasets
from MRCpy.datasets import *

# Data sets
loaders = [load_comp_vs_sci, load_comp_vs_talk, load_rec_vs_sci,
          load_rec_vs_talk, load_sci_vs_talk]
dataName = ["comp-vs-sci", "comp-vs-talk", "rec-vs-sci",
           "rec-vs-talk", "sci-vs-talk"]
sigma = np.array([23.5628, 23.4890, 24.5642, 25.1129, 24.8320])

rep = 10
n = 1000
t = 1000

Errors_CMRC = np.zeros([rep,5])
Errors_DWGCS1 = np.zeros([rep,5])
Errors_DWGCS2 = np.zeros([rep,5])

def runDWGCS(phi, loss):

    columns = ['dataset', 'n_samples', 'n_attributes', 'n_classes', 'error']
    resultsCMRC = pd.DataFrame(columns=columns)
    resultsDWGCS1 = pd.DataFrame(columns=columns)
    resultsDWGCS2 = pd.DataFrame(columns=columns)

    

    for j, load in enumerate(loaders):

        # Loading the dataset
        X_TrainSet, Y_TrainSet, X_TestSet, Y_TestSet = load()

        r = len(np.unique(Y_TrainSet))
        d = X_TrainSet.shape[1]

        TrainSet = np.concatenate((X_TrainSet, np.reshape(Y_TrainSet,(Y_TrainSet.shape[0], 1))), axis=1)
        TestSet = np.concatenate((X_TestSet, np.reshape(Y_TestSet, (Y_TestSet.shape[0], 1))), axis=1)

        Error1 = list()
        Error2 = list()
        Error3 = list()
        
        for i in range(rep):
            
            np.random.seed(42)
            np.random.shuffle(TrainSet)
            np.random.seed(42)
            np.random.shuffle(TestSet)

            X_train = TrainSet[:n, :-1]
            Y_train = TrainSet[:n, -1]
            X_test  = TestSet[:t, :-1]
            Y_test  = TestSet[:t, -1]


            #CMRC
            clf = CMRC(loss = loss, phi = phi, fit_intercept = False, s = 0)
            clf.fit(X_train, Y_train, X_test)
            Error1.append(clf.error(X_test, Y_test))
            Errors_CMRC[i, j] = clf.error(X_test, Y_test)    
            #DWGCS D = 4
            clf2 = DWGCS(loss = loss, phi = phi, sigma_ = sigma[j], D = 1)
            clf2.fit(X_train, Y_train, X_test)
            Error2.append(clf2.error(X_test, Y_test))
            Errors_DWGCS1[i, j] = clf2.error(X_test, Y_test)
            #DWGCS D = 4
            clf3 = DWGCS(loss = loss, phi = phi, sigma_ = 23.5628)
            clf3.fit(X_train, Y_train, X_test)
            Error3.append(clf3.error(X_test, Y_test))
            Errors_DWGCS2[i, j] = clf3.error(X_test, Y_test)


        res_mean1 = np.average(Error1)
        res_std1 = np.std(Error1)
        res_mean2 = np.average(Error2)
        res_std2 = np.std(Error2)
        res_mean3 = np.average(Error3)
        res_std3 = np.std(Error3)

        
        new_row = {'dataset': dataName[j],
                                  'n_samples': '%d' % n,
                                  'n_attributes': '%d' % d,
                                  'n_classes': '%d' % r,
                                  'error': '%1.2g' % res_mean1 + " +/- " +
                                  '%1.2g' % res_std1}
        resultsCMRC.loc[len(resultsCMRC)] = new_row
        new_row = {'dataset': dataName[j],
                                  'n_samples': '%d' % n,
                                  'n_attributes': '%d' % d,
                                  'n_classes': '%d' % r,
                                  'error': '%1.2g' % res_mean2 + " +/- " +
                                  '%1.2g' % res_std2}
        resultsDWGCS1.loc[len(resultsDWGCS1)] = new_row
        new_row = {'dataset': dataName[j],
                                  'n_samples': '%d' % n,
                                  'n_attributes': '%d' % d,
                                  'n_classes': '%d' % r,
                                  'error': '%1.2g' % res_mean3 + " +/- " +
                                  '%1.2g' % res_std3}
        resultsDWGCS2.loc[len(resultsDWGCS2)] = new_row
    
    return resultsCMRC, resultsDWGCS1, resultsDWGCS2

####################################################################

r1 = runDWGCS(phi='linear', loss='0-1')
print(r1)
