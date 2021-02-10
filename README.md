# Minimax-Risk-Classifier

Different variations of minimax risk classifiers(MRC) using different loss functions and uncertainity set of distributions.

The variants available here are - 

1) MRC with 0-1 loss (MRC.py)
2) MRC with log loss (MRC.py)
3) MRC with 0-1 loss and fixed instances' marginals (CMRC.py)
4) MRC with log loss and fixed instances' marginals (CMRC.py)

# Installation
[![Generic badge](https://img.shields.io/badge/Python-2.X|3.X-blue.svg)](https://shields.io/)<br/>

From a terminal (OS X & linux), you can install minimax_risk_classifiers and its requirements directly by running the setup.py script as follows

```
python3 setup.py install
```

# Getting started
To use the classification models (MRC and CMRC) of the minimax_risk_classifiers package, you have to define the instance of that object (as we do for any other classification model in scikit-learn) for which you can define the following parameters - 

n_classes : The number of classes

loss : The type of loss function to use

phi : Feature mapping type string or object (in case of the custom feature mappings)


## Fitting the models

MRC with 0-1 loss
```
clf = minimax_risk_classifiers.MRC(r=r, loss='0-1').fit(X, Y)
```

MRC with log loss
```
clf = minimax_risk_classifiers.MRC(r=r, loss='log').fit(X, Y)
```

MRC with 0-1 loss and fixed instances' marginals
```
clf = minimax_risk_classifiers.CMRC(r=r, loss='0-1').fit(X, Y)
```

MRC with log loss and fixed instances' marginals
```
clf = minimax_risk_classifiers.CMRC(r=r, loss='log').fit(X, Y)
```

## A small training example of MRC
```
from minimax_risk_classifiers.MRC import MRC
from sklearn.datasets import load_iris

X, Y = load_iris(return_X_y=True)
r = len(np.unique(Y))
clf = MRC(r=r).fit(X, Y)
y_pred = clf.predict(X[:2,:])
```

## Bounds on the error for MRC

```
clf = minimax_risk_classifiers.MRC(r=r).fit(X, Y)
upper_bound = clf.upper
lower_bound = clf.getLowerBound()
```

Only available for the MRC class


## Setting the interval estimates while fitting

Using instances X_ and Y to calculate tau and lambda
```
clf = minimax_risk_classifiers.MRC(r=r).fit(X, Y, X_)
```

By passing the values for tau and lambda
```
clf = minimax_risk_classifiers.MRC(r=r).fit(X, _tau=0.5, _lambda=0.1)
```

