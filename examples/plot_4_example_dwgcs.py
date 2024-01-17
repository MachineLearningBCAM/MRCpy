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

rep = 10
n = 1000
t = 1000

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

            #x = np.concatenate((X_train, X_test), axis=0)
            #x = preprocessing.StandardScaler().fit_transform(x)   
            #X_train = x[:n, :]
            #X_test = x[n:, :]
            starting_time = time.time()

            #CMRC
            clf = CMRC(loss = loss, phi = phi, one_hot = True)
            clf.fit(X_train, Y_train)
            Error1.append(clf.error(X_test, Y_test))
            #DWGCS D = 4
            clf2 = DWGCS(loss = loss, phi = phi, sigma_ = 23.5628, D = 1, one_hot = True)
            clf2.fit(X_train, Y_train, X_test)
            Error2.append(clf2.error(X_test, Y_test))
            #DWGCS D = 4
            clf3 = DWGCS(loss = loss, phi = phi, sigma_ = 23.5628, one_hot = True)
            clf3.fit(X_train, Y_train, X_test)
            Error3.append(clf3.error(X_test, Y_test))

            end_time = time.time()-starting_time
            print(end_time)

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
