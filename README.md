# Minimax-Risk-Classifier

Different variations of minimax risk classifiers(MRC) using different loss functions and uncertainity set of distributions.

The variants available here are - 

1) MRC with 0-1 loss (MRC.py)
2) MRC with log loss (MRC.py)
3) MRC with 0-1 loss and fixed instances' marginals (CMRC.py)
4) MRC with log loss and fixed instances' marginals (CMRC.py)

# Requirements
[![Generic badge](https://img.shields.io/badge/Python-2.X|3.X-blue.svg)](https://shields.io/)<br/>
We will need have installed the following libraries:
* numpy
* sklearn
* cvxpy

We can install the requirements directly from the file "requirements.txt"

```
pip install -r requirements.txt
```

# Usage

## Fitting the models

MRC with 0-1 loss
```
clf = MRC(r=r, loss='0-1').fit(X, Y)
```

MRC with log loss
```
clf = MRC(r=r, loss='log').fit(X, Y)
```

MRC with 0-1 loss and fixed instances' marginals
```
clf = CMRC(r=r, loss='0-1').fit(X, Y)
```

MRC with log loss and fixed instances' marginals
```
clf = CMRC(r=r, loss='log').fit(X, Y)
```

## A small training example of MRC
```
from MRC import MRC
from datasets import load_mammographic

X, Y = load_mammographic(return_X_y=True)
r = len(np.unique(Y))
clf = MRC(r=r).fit(X, Y)
y_pred = clf.predict(X[:2,:])
```

## Bounds on the error for MRC

```
clf = MRC(r=r).fit(X, Y)
upper_bound = clf.upper
lower_bound = clf.getLowerBound()
```

Only available for the MRC class


## Setting the interval estimates while fitting

Using instances X_ and Y_ to calculate tau and lambda
```
clf = MRC(r=r).fit(X, Y, X_, Y_)
```

By passing the values for tau and lambda
```
clf = MRC(r=r).fit(X, Y, _tau=0.5, _lambda=0.1)
```

