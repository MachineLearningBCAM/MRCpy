"""
Column and Constraint Generation (CCG) Solver for Large-Scale MRC.

This module implements an efficient Column and Constraint Generation algorithm
for Minimax Risk Classifiers (MRC) with large numbers of samples and features.
The algorithm iteratively adds both features (columns) and constraints (rows)
to solve the MRC optimization problem efficiently for high-dimensional data.

The main entry point is the `main_large_n_m` function in the `main` module,
which coordinates the initialization and CCG iterations.

Key Components
--------------
- ccg.py : Core CCG algorithm implementation
- ccg_utils.py : Utility functions for column/constraint generation
- mrc_lp_large_n_m.py : Gurobi LP model construction
- main.py : Main entry point and initialization logic

References
----------
The algorithm is based on column and constraint generation techniques for
large-scale optimization problems. See Mazumdar et al. for theoretical details.
"""