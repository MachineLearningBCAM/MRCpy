''' Stochastic gradient descent'''

import numpy as np

def SGD_optimization(tau_, lambda_, n, m, f_, g_, max_iters, stepsize, mini_batch_size=1, return_fn_vals=False):
    '''
    Solution of the CMRC convex optimization(minimization)
    using SGD approach.

    Parameters
    ----------
    n : `int`
        Number of samples used for optimization
    m : `int`
        Length of the feature mapping vector
    f_ : a lambda function/ function of the form - `f_(mu)`
        It is expected to be a lambda function or a function
        calculating a part of the objective function
        depending on the type of loss function chosen
        by taking the parameters(mu) of the optimization as input.
    g_ : a lambda function of the form - `g_(mu, idx)`
        It is expected to be a lambda function
        calculating the part of the subgradient of the objective function
        depending on the type of the loss function chosen.
        It takes as input -
        parameters (mu) of the optimization and
        the indices corresponding to the maximum value of subobjective
        for a given subset of Y (set of labels).

    Return
    ------
    new_params_ : `dict`
        Dictionary containing optimized values: mu and w_k (`array`-like,
        shape (`m`,)) - parameters corresponding to the last iteration,
        best_value (`float` - optimized value of the
        function in consideration.
    '''

    # Initial values for points
    # Infer number of classes from tau_ shape
    if len(tau_.shape) > 1:
        n_classes = tau_.shape[0]
        w_k = np.zeros((n_classes, tau_.shape[1]), dtype=np.float64)
    else:
        w_k = np.zeros(tau_.shape[0], dtype=np.float64)

    w_k_sum = w_k

    # Setting the initial indices for the batch
    batch_start_sample_id = 0
    batch_end_sample_id = batch_start_sample_id + mini_batch_size
    epoch_id = 0

    for k in range(1, (max_iters + 1)):

        g_0 = lambda_ * np.sign(w_k) - tau_ + g_(w_k,
                                                           batch_start_sample_id,
                                                           batch_end_sample_id,
                                                           n)

        if stepsize == 'decay':
            stepsize_ = 0.01 * (1 / 1 + 0.01 * epoch_id)
        elif type(stepsize) == float:
            stepsize_ = stepsize
        else:
            raise ValueError('Unexpected stepsize ... ')

        w_k = w_k - stepsize_ * g_0
        w_k_sum = w_k_sum + w_k

        batch_start_sample_id = batch_end_sample_id % n
        batch_end_sample_id = batch_start_sample_id + mini_batch_size
        if batch_end_sample_id > n:
            batch_end_sample_id = n
            epoch_id += 1

    w_k = w_k_sum / k
    psi, _ = f_(w_k)
    f_value = lambda_ @ np.abs(w_k) - tau_ @ w_k + psi
    mu = w_k

    # Return the optimized values in a dictionary
    new_params_ = {'w_k': w_k,
                   'mu': mu,
                   'best_value': f_value  # actually last value
                   }

    return new_params_