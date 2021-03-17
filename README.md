# MRCpy

Different variations of Minimax Risk Classifiers(MRC) using different loss functions and uncertainity set of distributions.

The variants available here are - 

1) MRC with 0-1 loss (MRC.py)
2) MRC with log loss (MRC.py)
3) MRC with 0-1 loss and fixed instances' marginals (CMRC.py)
4) MRC with log loss and fixed instances' marginals (CMRC.py)

# Installation
[![Generic badge](https://img.shields.io/badge/Python-2.X|3.X-blue.svg)](https://shields.io/)<br/>

From a terminal (OS X & linux), you can install ``MRCpy`` and its requirements directly by running the setup.py script as follows

```
git clone https://github.com/MachineLearningBCAM/Minimax-Risk-Classifiers.git
cd Minimax-Risk-Classifiers
python3 setup.py install
```

# Getting started
To use the classification models (MRC and CMRC) of the MRCpy package, you have to define the instance of that object (as we do for any other classification model in scikit-learn) for which you can define the following parameters - 

``n_classes`` : The number of classes

``loss`` : The type of loss function to use

``phi`` : Feature mapping type string or object (in case of the custom feature mappings)


## Fitting the models

MRC with 0-1 loss
```
clf = MRC(n_classes=n_classes, loss='0-1').fit(X, Y)
```

MRC with log loss
```
clf = MRC(n_classes=n_classes, loss='log').fit(X, Y)
```

MRC with 0-1 loss and fixed instances' marginals
```
clf = CMRC(n_classes=n_classes, loss='0-1').fit(X, Y)
```

MRC with log loss and fixed instances' marginals
```
clf = CMRC(n_classes=n_classes, loss='log').fit(X, Y)
```

## A small training example of MRC
```
from MRCpy import MRC
from sklearn.datasets import load_iris

X, Y = load_iris(return_X_y=True)
r = len(np.unique(Y))
clf = MRC(n_classes=n_classes).fit(X, Y)
y_pred = clf.predict(X[:2,:])
```

## Bounds on the error for MRC

```
clf = MRC(n_classes=n_classes).fit(X, Y)
upper_bound = clf.upper
lower_bound = clf.getLowerBound()
```

Only available for the MRC class


## Using different instances for setting interval estimates and optimization

Using instances X and Y to calculate ``tau`` and ``lambda`` and X_ for optimization
```
clf = MRC(n_classes=n_classes).fit(X, Y, X_)
```

By passing the values for ``tau`` and ``lambda``
```
clf = MRC(n_classes=n_classes).fit(X, tau_=[0.5, 0.5, ..., 0.1] , lambda_=[0.1, 0.02, ..., 0.04)
```

## Building your custom feature mappings

For building your own customized feature mappings, you can follow this [example(customPhi.py)](https://github.com/MachineLearningBCAM/MRCpy/blob/main/examples/customPhi.py) in the examples folder.

## Using CVXpy for optimization

By default, the classifier uses the subgradient methods for optimization. To use the CVXpy for optimization, set the ``use_cvx=True``.
```
clf = MRC(n_classes=n_classes, use_cvx=True).fit(X, Y, X_)
```

## Reusing previous solution of fit as initialization to the next call to fit (warm_start)

Reuse the solution of the previous call to fit as initialization in the next fit by setting ``warm_start=True``. This option is useful for faster convergence when you have to fit your classifier to an increasing dataset again and again.
```
clf = MRC(n_classes=n_classes, warm_start=True).fit(X, Y, X_)
```


