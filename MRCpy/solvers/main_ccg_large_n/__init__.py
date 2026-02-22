"""
Constraint Generation (CG) Solver for Large-Sample MRC.

This module implements an efficient Constraint Generation algorithm for
Minimax Risk Classifiers (MRC) specifically designed for problems with large
numbers of samples. The algorithm uses a dual formulation and iteratively adds
violated constraints to solve the MRC optimization problem efficiently.

Unlike the CCG variant, this solver focuses primarily on constraint generation
(adding rows) rather than both column and constraint generation. It is
particularly effective for problems where the number of samples is large but
the feature space is manageable.

The main entry point is the `main_large_n` function in the `main` module,
which coordinates the initialization and constraint generation iterations.

Key Components
--------------
- cg.py : Core constraint generation algorithm implementation
- cg_utils.py : Utility functions for constraint selection and model updates
- mrc_dual_lp.py : Gurobi dual LP model construction
- main.py : Main entry point and initialization logic

References
----------
The algorithm is based on constraint generation techniques for large-scale
optimization problems. See Mazumdar et al. for theoretical details.
"""