# MRCpy: A Library for Minimax Risk Classifiers

[![Build Status](https://app.travis-ci.com/MachineLearningBCAM/MRCpy.svg?branch=main)](https://travis-ci.com/github/MachineLearningBCAM/MRCpy)
[![Coverage Status](https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy)](https://codecov.io/gh/MachineLearningBCAM/MRCpy)


MRCpy library implements minimax risk classifiers (MRCs) that are based on robust risk minimization and can utilize 0-1 loss, in contrast to existing libraries for supervised classification using techniques based on empirical risk minimization and surrogate losses.

Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. This library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines.

In addition, MRCpy implements recent feature mappings such as Fourier, ReLU, and threshold features. The library is designed with an object-oriented approach that facilitates collaborators and users. The source code is available under the MIT license at <https://github.com/MachineLearningBCAM/MRCpy>.

## Algorithms

- MRC with 0-1 loss (MRC)
- MRC with log loss (MRC)
- MRC with 0-1 loss and fixed instances' marginals (CMRC)
- MRC with log loss and fixed instances' marginals (CMRC)

## Installation
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
<br/>

From a terminal (OS X & linux), you can install ``MRCpy`` and its requirements directly by running the setup.py script as follows

```
git clone https://github.com/MachineLearningBCAM/MRCpy.git
cd MRCpy
python3 setup.py install
```

__NOTE:__ CVXpy optimization uses MOSEK optimizer(by default) which requires a license. You can get a free academic license from [here](https://www.mosek.com/products/academic-licenses/).

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
               
- [3] [Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952.](<https://arxiv.org/abs/2108.01952>)

		@article{bondugula2021mrcpy,
		title={MRCpy: A Library for Minimax Risk Classifiers},
		author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
		journal={arXiv preprint arXiv:2108.01952},
		year={2021}
		}

## Updates and Discussion

You can subscribe to the [MRCpy's mailing list](https://mail.python.org/mailman3/lists/mrcpy.python.org/) for updates and discussion