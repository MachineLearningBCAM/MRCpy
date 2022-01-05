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



Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   api

.. toctree::
   :maxdepth: 2

   auto_examples/index
   
   .. toctree::
      :maxdepth: 2

      basic_examples

   .. toctree::
      :maxdepth: 2
      
      further_applications

References
----------

 For more information about the MRC method and the MRCpy library, one can refer to the following resources:

   - [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification with 0-1 Loss and Performance Guarantees. Advances in Neural Information Processing Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

         ::
         
               @article{mazuelas2020minimax,
                  title={Minimax Classification with 0-1 Loss and Performance Guarantees},
                  author={Mazuelas, Santiago and Zanoni, Andrea and P{\'e}rez, Aritz},
                  journal={Advances in Neural Information Processing Systems},
                  volume={33},
                  pages={302--312},
                  year={2020}
               }

   - [2] `Mazuelas, S., Shen, Y., & Pérez, A. (2020). Generalized Maximum Entropy for Supervised Classification. arXiv preprint arXiv:2007.05447. <https://arxiv.org/abs/2007.05447>`_

         ::

               @article{mazuelas2020generalized,
                  title={Generalized Maximum Entropy for Supervised Classification},
                  author={Mazuelas, Santiago and Shen, Yuan and P{\'e}rez, Aritz},
                  journal={arXiv preprint arXiv:2007.05447},
                  year={2020}
               }

   - [3] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952. <https://arxiv.org/abs/2108.01952>`_

         ::

               @article{bondugula2021mrcpy,
                  title={MRCpy: A Library for Minimax Risk Classifiers},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
                  journal={arXiv preprint arXiv:2108.01952},
                  year={2021}
               }

Funding
^^^^^^^^
Research carried out under the project **PID2019-105058GA-I00** funded by
**MCIN/AEI/10.13039/501100011033**.

.. image:: fund_logo.png
  :width: 150
  :alt: Research carried out under the project PID2019-105058GA-I00 funded by MCIN/ AEI /10.13039/501100011033

.. |Travis-CI Build Status| image:: https://travis-ci.org/MachineLearningBCAM/MRCpy.svg?branch=main
   :target: https://travis-ci.org/github/MachineLearningBCAM/MRCpy
.. |Code coverage| image:: https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy
   :target: https://codecov.io/gh/MachineLearningBCAM/MRCpy
