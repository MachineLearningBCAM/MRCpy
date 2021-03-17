"""
Simple example of using MRC with 0-1 loss.
"""

import numpy as np
from sklearn.impute import SimpleImputer

from MRCpy import MRC
# Import the datasets
from MRCpy.datasets import load_mammographic


if __name__ == '__main__':

    # Loading the dataset
    X, Y = load_mammographic(return_X_y=True)
    r = len(np.unique(Y))

    # Preprocess
    trans = SimpleImputer(strategy='median')
    X = trans.fit_transform(X, Y)

    # Fit the MRC model
    clf = MRC(n_classes=r).fit(X, Y)

    # Prediction
    print('\n\nThe predicted values for the first 3 instances are : ')
    print(clf.predict(X[:3, :]))

    # Predicted probabilities
    print('\n\nThe predicted probabilities for the first 3 instances are : ')
    print(clf.predict_proba(X[:3, :]))

    print('\n\nThe score is : ')
    print(clf.score(X, Y))
