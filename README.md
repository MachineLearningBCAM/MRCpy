# MRCpy: A Library for Minimax Risk Classifiers

[![Build Status](https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield)](https://circleci.com/gh/MachineLearningBCAM/MRCpy)
[![Coverage Status](https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy)](https://codecov.io/gh/MachineLearningBCAM/MRCpy)


MRCpy implements Minimax Risk Classifiers (MRCs), which are based on the robust risk minimization (RRM) framework. Unlike empirical risk minimization (ERM), RRM accounts for uncertainty in the underlying data distribution by optimizing the worst-case risk over a set of plausible distributions. These techniques give rise to a broad family of classification methods that provide guarantees in terms of an upper bound on the classification error at training.

MRCpy provides a unified interface for different variants of MRCs, following the design standards of popular Python machine learning libraries. The library includes efficient implementations of MRC-based methods designed to scale to large datasets and high-dimensional problems. It also provides implementations of established techniques that can be formulated as MRCs, including L1-regularized logistic regression, zero-one adversarial classification, and maximum entropy machines. In addition, MRCpy includes PyTorch-based classifiers that enable the integration of MRC objectives with deep neural networks, allowing users to train DNNs using minimax risk-based learning objectives.


## Installation
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
<br/>

The latest built version of ``MRCpy`` can be installed using `pip` as 

```
pip install MRCpy
```

Alternatively, the development version (GitHub) of ``MRCpy`` can be installed as follows

```
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
python3 setup.py install
```

__NOTE:__ The solver based on CVXpy in the library uses GUROBI optimizer which requires a license. You can get a free academic license from [here](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist).

### Dependencies

`MRCpy` requires:

- Python (>= 3.9)
- NumPy, SciPy, scikit-learn, cvxpy, pandas, pyarrow
- gurobipy (>= 9.5, tested with 9.5, 12 and 13; requires [license](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist) — free for academics)
- pycddlib (required only for LMRC — see [installation guide](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation))

Optional (for PyTorch MGCE classifier), installed with `pip install "MRCpy[pytorch]"`:

- torch, tqdm

## Usage

MRC classifiers follow the scikit-learn estimator API, so they work with the
usual `fit` / `predict` / `score` calls and with scikit-learn tooling such as
`cross_val_score` and `GridSearchCV`. In addition to predictions, every MRC
model exposes performance guarantees for the underlying classification task:

```python
from sklearn.model_selection import train_test_split

from MRCpy import MRC
from MRCpy.datasets import load_mammographic

X, y = load_mammographic()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = MRC(phi='threshold').fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('error:  %.3f' % (y_pred != y_test).mean())
print('bounds: [%.3f, %.3f]' % (clf.get_lower_bound(), clf.get_upper_bound()))
```

```
error:  0.207
bounds: [0.084, 0.240]
```

`get_upper_bound()` and `get_lower_bound()` bracket the minimum expected error
of the classification task, and are obtained from the training data alone — no
test set is required.

The library also provides a PyTorch classifier, `mgce_clf`, which trains any
`torch.nn.Module` under the minimax generalized cross-entropy loss [1]. It takes
a model and an optimizer, is fitted from a `DataLoader`, and then predicts on
NumPy arrays like the estimators above:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from MRCpy.datasets import load_mammographic
from MRCpy.pytorch.mgce.classifier import mgce_clf

X, y = load_mammographic()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long))
train_set.classes = [0, 1]  # fit() reads the class list from the dataset
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 2))

clf = mgce_clf(loss_parameter=1.4,
               model=model,
               optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
               device='cpu')
clf.fit(train_loader, n_epochs=20, verbose=False, save_model_weights=None)

y_pred = clf.predict(X_test)
print('error: %.3f' % (y_pred != y_test).mean())
```

See the [MRCpy documentation page](https://machinelearningbcam.github.io/MRCpy/) for full documentation about installation, API, usage, and examples.

## Citation

If you use MRCpy in a scientific publication, please cite:

[Bondugula, K., Álvarez, V., Segovia-Martín, J.I., Pérez, A., & Mazuelas, S. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv:2108.01952.](https://arxiv.org/abs/2108.01952)

```bibtex
@article{bondugula2021mrcpy,
  title={MRCpy: A Library for Minimax Risk Classifiers},
  author={Bondugula, Kartheek and {\'A}lvarez, Ver{\'o}nica and Segovia-Mart{\'i}n, Jos{\'e} I. and P{\'e}rez, Aritz and Mazuelas, Santiago},
  journal={arXiv preprint arXiv:2108.01952},
  year={2021}
}
```

The complete list of citations for the different minimax methods implemented in the library can be found in the [documentation](https://machinelearningbcam.github.io/MRCpy/citing.html).

## Contributing

Contributions are welcome — bug reports, new classifiers, feature mappings,
documentation and examples alike. If you are planning a larger change, please
[open an issue](https://github.com/MachineLearningBCAM/MRCpy/issues) first so
the design can be discussed before you write the code.

To set up a development install:

```bash
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
pip install -e .
pip install -r dev-requirements.txt
```

Then open a pull request against `main`. Continuous integration checks code
style and runs the full test suite, and both must pass before a pull request
can be merged.

### Running tests

```bash
python -m unittest discover -s tests
```

To reproduce the coverage report produced by CI:

```bash
coverage run --source=MRCpy/ -m unittest discover -s tests
coverage report
```

## Updates and Discussion

You can subscribe to the [MRCpy's mailing list](https://mail.python.org/mailman3/lists/mrcpy.python.org/) for updates and discussion
