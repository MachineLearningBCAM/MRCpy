.. MRCpy documentation master file, created by
   sphinx-quickstart on Sat Jun  5 15:07:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MRCpy: A Library for Minimax Risk Classifiers
=============================================
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


|Travis-CI Build Status| |Code coverage|

`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ library implements minimax risk classifiers (MRCs) that are based on robust risk minimization and can utilize 0-1-loss. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. The presented library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines. In addition, MRCpy implements recent feature mappings such as Fourier, ReLU, and threshold features.

`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ also includes a variety of datasets with convenient loader functions. See the :doc:`getting_started` guide for the full list of available datasets and usage examples.


Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   minimax_framework

.. toctree::
   :maxdepth: 2

   api

.. toctree::
   :maxdepth: 2

   auto_examples/index


References
----------

 For more information about the methods available in MRCpy library, one can refer to the following resources:

   - [1] `Bondugula, K., Mazuelas, S., Pérez, A., & Liu, A. (2026). Minimax Generalized Cross-Entropy. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS). <https://arxiv.org/abs/2411.07789>`_

         ::

               @inproceedings{BonMazPerLiu:26,
                  title={Minimax Generalized Cross-Entropy},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz and Liu, Anqi},
                  booktitle={Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)},
                  year={2026}
               }

   - [2] `Bondugula, K., Mazuelas, S., & Pérez, A. (2025). Efficient Large-Scale Learning of Minimax Risk Classifiers. IEEE International Conference on Data Mining (ICDM). <https://arxiv.org/abs/2406.11684>`_

         ::

               @inproceedings{BonMazPer:25,
                  title={Efficient Large-Scale Learning of Minimax Risk Classifiers},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
                  booktitle={IEEE International Conference on Data Mining (ICDM)},
                  year={2025}
               }

   - [3] `Mazuelas, S., Romero, M., & Grunwald, P. (2023). Minimax Risk Classifiers with 0-1 Loss. Journal of Machine Learning Research, 24(208), 1-48. <https://jmlr.org/papers/volume24/22-0339/22-0339.pdf>`_

         ::

               @article{MazRomGrun:23,
                         author = {Mazuelas, Santiago and Romero, Mauricio and Grunwald, Peter},
                         title = {Minimax Risk Classifiers with 0-1 Loss},
                         journal={Journal of Machine Learning Research},
                         volume = {24},
                         number = {208},
                         pages = {1--48},
                         year={2023}
                        }

   - [4] `Bondugula, K., Mazuelas, S., & Pérez, A. (2023). Efficient Learning of Minimax Risk Classifiers in High Dimensions. The 39th Conference on Uncertainty in Artificial Intelligence (UAI), 206-215. <https://proceedings.mlr.press/v216/bondugula23a.html>`_

         ::

               @inproceedings{BonMazPer:23,
                  title={Efficient Learning of Minimax Risk Classifiers in High Dimensions},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
                  booktitle={The 39th Conference on Uncertainty in Artificial Intelligence (UAI)},
                  pages={206--215},
                  year={2023}
               }

   - [5] `Segovia-Martín, J.I., Mazuelas, S., & Liu, A. (2023). Double-Weighting for Covariate Shift Adaptation. In Proceedings of the 40th International Conference on Machine Learning (ICML), pp. 30439-30457. <https://proceedings.mlr.press/v202/segovia-martin23a.html>`_

         ::

               @inproceedings{SegMazLiu:23,
                  title={Double-Weighting for Covariate Shift Adaptation},
                  author={Segovia-Mart{\'i}n, Jos{\'e} I. and Mazuelas, Santiago and Liu, Anqi},
                  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
                  pages={30439--30457},
                  year={2023}
               }

   - [6] `Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized Maximum Entropy for Supervised Classification. IEEE Transactions on Information Theory, 68(4), 2530-2550. <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9682746>`_

         ::

               @article{MazShePer:22,
                         author = {Santiago Mazuelas and Yuan Shen and Aritz P\'{e}rez},
                         title = {Generalized Maximum Entropy for Supervised Classification},
                         journal={IEEE Transactions on Information Theory},
                         volume = {68},
                         number = {4},
                         pages = {2530-2550},
                         year={2022}
                        }

   - [7] `Álvarez, V., Mazuelas, S., & Lozano, J.A. (2022). Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees. In Proceedings of the 39th International Conference on Machine Learning (ICML), pp. 486-499. <https://proceedings.mlr.press/v162/alvarez22a.html>`_

         ::

               @inproceedings{AlvMazLoz:22,
                  title={Minimax Classification under Concept Drift with Multidimensional Adaptation and Performance Guarantees},
                  author={{\'A}lvarez, Ver{\'o}nica and Mazuelas, Santiago and Lozano, Jos{\'e} A.},
                  booktitle={Proceedings of the 39th International Conference on Machine Learning (ICML)},
                  pages={486--499},
                  year={2022}
               }

   - [8] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952. <https://arxiv.org/abs/2108.01952>`_

         ::

               @article{bondugula2021mrcpy,
                  title={MRCpy: A Library for Minimax Risk Classifiers},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
                  journal={arXiv preprint arXiv:2108.01952},
                  year={2021}
               }

   - [9] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification with 0-1 Loss and Performance Guarantees. Advances in Neural Information Processing Systems, 33, 302-312. <https://proceedings.neurips.cc/paper_files/paper/2020/file/02f657d55eaf1c4840ce8d66fcdaf90c-Paper.pdf>`_

         ::
         
               @article{mazuelas2020minimax,
                  title={Minimax Classification with 0-1 Loss and Performance Guarantees},
                  author={Mazuelas, Santiago and Zanoni, Andrea and P{\'e}rez, Aritz},
                  journal={Advances in Neural Information Processing Systems},
                  volume={33},
                  pages={302--312},
                  year={2020}
               }

Funding
^^^^^^^^
Funding in direct support of this work has been provided through different research projects by the following institutions.

.. figure:: fund_logo.png
   :align: center
   :width: 150
   :alt: Spanish Ministry of Science and Innovation logo

   Spanish Ministry of Science and Innovation through the project **PID2019-105058GA-I00** funded by **MCIN/AEI/10.13039/501100011033**.

.. figure:: axalogo.png
   :align: center
   :width: 150
   :alt: AXA Research Fund logo

   AXA Research Fund through the project **"Early Prognosis of COVID-19 Infections via Machine Learning"** funded in the Exceptional Flash Call **"Mitigating risk in the wake of the COVID-19 pandemic"**.

.. figure:: logogobiernovasco.png
   :align: center
   :width: 150
   :alt: Basque Government logo

   Basque Government through the project **"Mathematical Modeling Applied to Health"**, and through the **"ELKARTEK Program"**.

.. |Travis-CI Build Status| image:: https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield
   :target: https://circleci.com/gh/MachineLearningBCAM/MRCpy
.. |Code coverage| image:: https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy
   :target: https://codecov.io/gh/MachineLearningBCAM/MRCpy

