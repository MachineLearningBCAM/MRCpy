import cvxpy as cvx
import numpy as np

def try_solvers(objective, constraints, mu, cvx_solvers):
    '''
    Solves the MRC problem
    using different types of solvers available in CVXpy

    Parameters
    ----------
    objective : `cvxpy` variable of `float` value
        Minimization objective function of the
        problem of the MRC.

    constraints : `array`-like of shape (`n_constraints`)
        Constraints for the MRC optimization.

    mu : `cvxpy` array of shape (number of featuers in `phi`)
        Parameters used in the optimization problem

    cvx_solvers : List of alternate cvx solvers to use for
              solving the optimization in case of failure.

    Returns
    -------
    mu_ : `array`-like of shape (number of featuers in `phi`)
        Value of the parameters
        corresponding to the optimum value of the objective function.

    objective_value : `float`
        Optimized objective value.

    '''

    # Solve the problem
    prob = cvx.Problem(objective, constraints)
    try:
        prob.solve(solver=cvx_solvers[0], verbose=False)
        mu_ = mu.value
    except:
        print('Error occured while using ' + cvx_solvers[0] + ' solver.\n' + \
              'Trying with the following solvers ' + ' '.join(cvx_solvers[1:]))
        mu_ = None

    # if the solver could not find values of mu for the given solver
    if mu_ is None:

        # try with a different solver for solution
        for s in cvx_solvers[1:]:

            # Solve the problem
            try:
                prob.solve(solver=s, verbose=False)
                mu_ = mu.value
            except:
                print('Error occured while using ' + cvx_solvers[0] + ' solver.')
                mu_ = None

            # Check the values
            mu_ = mu.value

            # Break the loop once the solution is obtained
            if mu_ is not None:
                break

    # If no solution can be found for the optimization.
    if mu_ is None:
        raise ValueError('CVXpy solver couldn\'t find a solution .... ' +
                         'The problem is ', prob.status)

    objective_value = prob.value
    return mu_, objective_value