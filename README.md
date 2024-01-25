# MRCpy: A Library for Minimax Risk Classifiers

[![Build Status](https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield)](https://circleci.com/gh/MachineLearningBCAM/MRCpy)
[![Coverage Status](https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy)](https://codecov.io/gh/MachineLearningBCAM/MRCpy)


MRCpy implements recently proposed supervised classification techniques called minimax risk classifiers (MRCs). MRCs are based on robust risk minimization and can utilize 0-1 loss, in contrast to existing libraries using techniques based on empirical risk minimization and surrogate losses. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss, enable efficient learning in high dimensions, and adapt to distribution shifts. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. This library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines.

## Algorithms

- [Minimax risk classifiers](https://arxiv.org/abs/2007.05447)
- [Adaptive minimax risk classifiers for classification under concept drift](https://proceedings.mlr.press/v162/alvarez22a/alvarez22a.pdf)
- [MRCs for covariate shift adaptation](https://proceedings.mlr.press/v202/segovia-martin23a/segovia-martin23a.pdf)
- [MRCs for classification in high dimensions](https://proceedings.mlr.press/v216/bondugula23a/bondugula23a.pdf)

## Installation
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
<br/>

From a terminal (OS X & linux), you can install ``MRCpy`` and its requirements directly by running the setup.py script as follows

```
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
python3 setup.py install
```

__NOTE:__ The solver based on CVXpy in the library uses GUROBI optimizer which requires a license. You can get a free academic license from [here](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist).

### Dependencies

- `Python` >= 3.6
- `numpy` >= 1.18.1, `scipy`>= 1.4.1, `scikit-learn` >= 0.21.0, `cvxpy`, `mosek`, `pandas`

## Usage

See the [MRCpy documentation page](https://machinelearningbcam.github.io/MRCpy/) for full documentation about installation, API, usage, and examples.

## Citations
This repository is the official implementation of Minimax Risk Classifiers proposed in the following papers. If you use MRCpy in a scientific publication, we would appreciate citations to:

- [1] [Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification with 0-1 Loss and Performance Guarantees. Advances in Neural Information Processing Systems, 33, 302-312.] (<https://arxiv.org/abs/2010.07964>)

		@article{mazuelas2020minimax,
		title={Minimax Classification with 0-1 Loss and Performance Guarantees},
		author={Mazuelas, Santiago and Zanoni, Andrea and P{\'e}rez, Aritz},
		journal={Advances in Neural Information Processing Systems},
		volume={33},
		pages={302--312},
		year={2020}
		}
               
- [2] [Mazuelas, S., Shen, Y., & Pérez, A. (2020). Generalized Maximum Entropy for Supervised Classification. arXiv preprint arXiv:2007.05447.](<https://arxiv.org/abs/2007.05447>)
		
		@article{mazuelas2020generalized,
		title={Generalized Maximum Entropy for Supervised Classification},
		author={Mazuelas, Santiago and Shen, Yuan and P{\'e}rez, Aritz},
		journal={arXiv preprint arXiv:2007.05447},
		year={2020}
		}
               
- [3] [Bondugula, K. et al (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952.](<https://arxiv.org/abs/2108.01952>)

		@article{bondugula2021mrcpy,
		title={MRCpy: A Library for Minimax Risk Classifiers},
		author={Bondugula, Kartheek, and Alvarez, Veronica and Segovia-Mart{\'i}n J. I. and Mazuelas, Santiago and P{\'e}rez, Aritz},
		journal={arXiv preprint arXiv:2108.01952},
		year={2021}
		}

## Updates and Discussion

You can subscribe to the [MRCpy's mailing list](https://mail.python.org/mailman3/lists/mrcpy.python.org/) for updates and discussion