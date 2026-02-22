"""
Column and Constraint Generation (CCG) Solver for Large-Scale Multiclass MRC.

This module implements an efficient Column and Constraint Generation algorithm
for Minimax Risk Classifiers (MRC) specifically designed for multiclass
classification problems with large numbers of samples. The algorithm uses a
dual formulation and iteratively adds violated constraints to solve the MRC
optimization problem efficiently.

The main entry point is the `main_large_n_efficient_multiclass` function in
the `main` module, which coordinates the initialization and constraint
generation iterations.

Key Components
--------------
- cg.py : Core constraint generation algorithm implementation
- cg_utils.py : Utility functions for constraint generation and selection
- mrc_dual_lp.py : Gurobi dual LP model construction
- main.py : Main entry point and initialization logic

References
----------
The algorithm is based on constraint generation techniques for large-scale
multiclass optimization problems. See Mazumdar et al. for theoretical details.
"""