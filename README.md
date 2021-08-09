# MRCpy: A Library for Minimax Risk Classifiers

[![Build Status](https://app.travis-ci.com/MachineLearningBCAM/MRCpy.svg?branch=main)](https://travis-ci.com/github/MachineLearningBCAM/MRCpy)
[![Coverage Status](https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy)](https://codecov.io/gh/MachineLearningBCAM/MRCpy)

Variants of Minimax Risk Classifiers(MRC) using different loss functions and uncertainity set of distributions.

The variants available here are - 

1) MRC with 0-1 loss (MRC)
2) MRC with log loss (MRC)
3) MRC with 0-1 loss and fixed instances' marginals (CMRC)
4) MRC with log loss and fixed instances' marginals (CMRC)

# Installation
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
<br/>

From a terminal (OS X & linux), you can install ``MRCpy`` and its requirements directly by running the setup.py script as follows

```
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
python3 setup.py install
```

__NOTE:__ CVXpy optimization uses MOSEK optimizer(by default) which requires a license. You can get a free academic license from [here](https://www.mosek.com/products/academic-licenses/).

# Getting started
To use the classification models (MRC and CMRC) of the MRCpy package, you have to define the instance of that object (as we do for any other classification model in scikit-learn) for which you can define the following parameters - 

``loss`` : The type of loss function to use

``phi`` : Feature mapping type string or object (in case of the custom feature mapping)


## Fitting the models

MRC with 0-1 loss
```
clf = MRC(loss='0-1').fit(X, Y)
```

MRC with log loss
```
clf = MRC(loss='log').fit(X, Y)
```

MRC with 0-1 loss and fixed instances' marginals
```
clf = CMRC(loss='0-1').fit(X, Y)
```

MRC with log loss and fixed instances' marginals
```
clf = CMRC(loss='log').fit(X, Y)
```

## A small training example of MRC
```
from MRCpy import MRC
from sklearn.datasets import load_iris

X, Y = load_iris(return_X_y=True)
r = len(np.unique(Y))
clf = MRC().fit(X, Y)
y_pred = clf.predict(X[:2,:])
```

## Bounds on the error for MRC

```
clf = MRC().fit(X, Y)
upper_bound = clf.get_upper_bound()
lower_bound = clf.get_lower_bound()
```

Only available for the MRC class


## Customizing the MRCs learning process using your own interval estimates for optimization

By passing the values for ``tau`` and ``lambda`` (interval estimates), you set the constraint for the uncertainty sets. For this purpose, you can use the ``minimax_risk`` function for fitting the classifier instead of ``fit``.
```
clf = MRC().minimax_risk(X, tau_=[0.5, 0.5, ..., 0.1] , lambda_=[0.1, 0.02, ..., 0.04], n_classes=2)
```

## Building your custom feature mappings

For building your own customized feature mappings, you can follow this [example(customPhi.py)](https://github.com/MachineLearningBCAM/MRCpy/blob/main/examples/customPhi.py) in the examples folder.

## Using CVXpy for optimization

By default, the classifier uses the subgradient methods for optimization. To use the CVXpy for optimization, set the ``use_cvx=True``.
```
clf = MRC(use_cvx=True).fit(X, Y)
```

## Reusing previous solution of fit as initialization to the next call to fit (warm_start)

Reuse the solution of the previous call to fit as initialization in the next fit by setting ``warm_start=True``. This option is useful for faster convergence when you have to fit your classifier to an increasing dataset again and again.
```
clf = MRC(warm_start=True).fit(X, Y)
```

See the [documentation](https://MachineLearningBCAM.github.io/MRCpy/) for more details about the API and its usage.

# Updates and Discussion

You can subscribe to the MRCpy's mailing [list](https://mail.python.org/mailman3/lists/mrcpy.python.org/) for updates and discussion.
