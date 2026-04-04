# MRCpy: A Library for Minimax Risk Classifiers

[![Build Status](https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield)](https://circleci.com/gh/MachineLearningBCAM/MRCpy)
[![Coverage Status](https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy)](https://codecov.io/gh/MachineLearningBCAM/MRCpy)


MRCpy implements recently proposed supervised classification techniques called minimax risk classifiers (MRCs). MRCs are based on robust risk minimization and can utilize 0-1 loss, in contrast to existing libraries using techniques based on empirical risk minimization and surrogate losses. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. This library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines.


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
- gurobipy (requires [license](https://portal.gurobi.com/iam/login/?target=https%3A%2F%2Fportal.gurobi.com%2Fiam%2Flicenses%2Flist) — free for academics)
- pycddlib (required only for LMRC — see [installation guide](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation))

Optional (for PyTorch MGCE classifier):

- torch, tqdm

## Usage

See the [MRCpy documentation page](https://machinelearningbcam.github.io/MRCpy/) for full documentation about installation, API, usage, and examples.

## Citations

If you use MRCpy in a scientific publication, we would appreciate citations to the relevant papers:

- [1] [Bondugula, K., Mazuelas, S., Pérez, A., & Liu, A. (2026). Minimax Generalized Cross-Entropy. AISTATS.](https://arxiv.org/abs/2603.19874)

- [2] [Bondugula, K., Mazuelas, S., & Pérez, A. (2025). Efficient Large-Scale Learning of Minimax Risk Classifiers. IEEE ICDM.](https://arxiv.org/abs/2511.17626)

- [3] [Mazuelas, S., Romero, M., & Grunwald, P. (2023). Minimax Risk Classifiers with 0-1 Loss. JMLR, 24(208), 1-48.](https://jmlr.org/papers/volume24/22-0339/22-0339.pdf)

- [4] [Bondugula, K., Mazuelas, S., & Pérez, A. (2023). Efficient Learning of Minimax Risk Classifiers in High Dimensions. UAI, 206-215.](https://proceedings.mlr.press/v216/bondugula23a.html)

- [5] [Segovia-Martín, J.I., Mazuelas, S., & Liu, A. (2023). Double-Weighting for Covariate Shift Adaptation. ICML, 30439-30457.](https://proceedings.mlr.press/v202/segovia-martin23a.html)

- [6] [Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized Maximum Entropy for Supervised Classification. IEEE Trans. Inf. Theory, 68(4), 2530-2550.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9682746)

- [7] [Álvarez, V., Mazuelas, S., & Lozano, J.A. (2022). Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees. ICML, 486-499.](https://proceedings.mlr.press/v162/alvarez22a.html)

- [8] [Bondugula, K., Álvarez, V., Segovia-Martín, J.I., Pérez, A., & Mazuelas, S. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv:2108.01952.](https://arxiv.org/abs/2108.01952)

- [9] [Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification with 0-1 Loss and Performance Guarantees. NeurIPS, 33, 302-312.](https://proceedings.neurips.cc/paper_files/paper/2020/file/02f657d55eaf1c4840ce8d66fcdaf90c-Paper.pdf)

<details>
<summary>BibTeX entries</summary>

```bibtex
@inproceedings{BonMazPerLiu:26,
  title={Minimax Generalized Cross-Entropy},
  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz and Liu, Anqi},
  booktitle={Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}

@inproceedings{BonMazPer:25,
  title={Efficient Large-Scale Learning of Minimax Risk Classifiers},
  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
  booktitle={IEEE International Conference on Data Mining (ICDM)},
  year={2025}
}

@article{MazRomGrun:23,
  title={Minimax Risk Classifiers with 0-1 Loss},
  author={Mazuelas, Santiago and Romero, Mauricio and Grunwald, Peter},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={208},
  pages={1--48},
  year={2023}
}

@inproceedings{BonMazPer:23,
  title={Efficient Learning of Minimax Risk Classifiers in High Dimensions},
  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
  booktitle={The 39th Conference on Uncertainty in Artificial Intelligence (UAI)},
  pages={206--215},
  year={2023}
}

@inproceedings{SegMazLiu:23,
  title={Double-Weighting for Covariate Shift Adaptation},
  author={Segovia-Mart{\'i}n, Jos{\'e} I. and Mazuelas, Santiago and Liu, Anqi},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  pages={30439--30457},
  year={2023}
}

@article{MazShePer:22,
  title={Generalized Maximum Entropy for Supervised Classification},
  author={Santiago Mazuelas and Yuan Shen and Aritz P\'{e}rez},
  journal={IEEE Transactions on Information Theory},
  volume={68},
  number={4},
  pages={2530-2550},
  year={2022}
}

@inproceedings{AlvMazLoz:22,
  title={Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees},
  author={{\'A}lvarez, Ver{\'o}nica and Mazuelas, Santiago and Lozano, Jos{\'e} A.},
  booktitle={Proceedings of the 39th International Conference on Machine Learning (ICML)},
  pages={486--499},
  year={2022}
}

@article{bondugula2021mrcpy,
  title={MRCpy: A Library for Minimax Risk Classifiers},
  author={Bondugula, Kartheek and {\'A}lvarez, Ver{\'o}nica and Segovia-Mart{\'i}n, Jos{\'e} I. and P{\'e}rez, Aritz and Mazuelas, Santiago},
  journal={arXiv preprint arXiv:2108.01952},
  year={2021}
}

@article{mazuelas2020minimax,
  title={Minimax Classification with 0-1 Loss and Performance Guarantees},
  author={Mazuelas, Santiago and Zanoni, Andrea and P{\'e}rez, Aritz},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={302--312},
  year={2020}
}
```

</details>

## Updates and Discussion

You can subscribe to the [MRCpy's mailing list](https://mail.python.org/mailman3/lists/mrcpy.python.org/) for updates and discussion
