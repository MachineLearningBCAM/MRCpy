#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:14:23 2022

@author: cguerrero
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:02:13 2022

@author: cguerrero
"""



from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn import preprocessing
import sys






# dataNames = ['adult','magic', 'vehicle','redwine','diabetes']
# dataNames = ['iris', 'haberman', 'glass', 'ecoli']
# dataName = dataNames[int(sys.argv[3])]
dataName = 'vehicle' 




# Loading the dataset
# myFile = np.genfromtxt('../../dipc/cguerrero/datasets/'+ dataName, delimiter=',')
# myFile = np.genfromtxt('../benchm_datasets/'+ dataName+'.csv', delimiter=',')
myFile = np.genfromtxt('../benchm_datasets/'+dataName+'.csv', delimiter=',')

numattr=len(myFile[0])-1
X = myFile[:,0:numattr]
Y = myFile[:,numattr]

r = len(np.unique(Y))
n, d= X.shape




X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=11)


std_scale = preprocessing.StandardScaler().fit(X_train, Y_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

std_scale = preprocessing.StandardScaler().fit(X, Y)
X = std_scale.transform(X)



# X_train = X_train[:1000,:]

# Y_train = Y_train[:1000]
from MRCpy import CMRC

# suffix = dataName+'_'+phi+'_'+str(stepsize)+'_'+str(max_iters)
# print("THIS IS "+suffix)
for loss in ['log','0-1']:
    for phi in ['linear','fourier']:
# ORIGINAL

        starttime = time.time()
        clf1 = CMRC(phi=phi, random_state=0, loss=loss, max_iters=30000,
                                one_hot=True, fit_intercept=False).fit(X_train, Y_train)
        t1 = time.time()-starttime
        Y_pred1 = clf1.predict(X_test)
        error1 = np.average(Y_pred1!=Y_test)

# np.savetxt('e_t_original'+suffix+'.csv',[error1,t1])
# np.savetxt('f_value_original'+suffix+'.csv',g1)
        print('loss = '+loss+', phi = '+phi)
        print('  time = '+str(t1))
        print('  error = '+str(error1))
        print('  f_value = '+str(clf1.params_['best_value']))
        
        

