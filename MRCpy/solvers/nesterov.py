import numpy as np

def nesterov_optimization_mrc(tau_, lambda_, m, f_, g_, max_iters):
    '''
    Solution of the MRC convex optimization (minimization)
    using the Nesterov accelerated approach.
    .. seealso:: [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                        The Strength of Nesterov’s Extrapolation in
                        the Individual Convergence of Nonsmooth
                        Optimization. IEEE transactions on
                        neural networks and learning systems,
                        31(7), 2557-2568.
                        <https://ieeexplore.ieee.org/document/8822632>`_
    Parameters
    ----------
    m : `int`
        Length of the feature mapping vector
    f_ : a lambda function of the form - f_(mu)
        Lambda function
        calculating a part of the objective function
        depending on the type of loss function chosen
        by taking the parameters (mu) of the optimization as input.
    g_ : a lambda function of the form - g_(mu, idx)
        Lambda function
        calculating the part of the subgradient of the objective function
        depending on the type of the loss function chosen.
        It takes the as input: parameters (mu) of the optimization and
        the index corresponding to the maximum value of data matrix
        obtained from the instances.
    Returns
    -------
    new_params_ : `dict`
        Dictionary that stores the optimal points
        (`w_k`: `array-like` shape (`m`,), `w_k_prev`: `array-like`
         shape (`m`,)) where `m`is the length of the feature
        mapping vector and best value
        for the upper bound (`best_value`: `float`) of the function and
        the parameters corresponding to the optimized function value
        (`mu`: `array-like` shape (`m`,),
        `nu`: `float`).
    '''

    # Initial values for the parameters
    theta_k = 1
    theta_k_prev = 1

    # Initial values for points
    y_k = np.zeros(m, dtype=np.float64)
    w_k = np.zeros(m, dtype=np.float64)
    w_k_prev = np.zeros(m, dtype=np.float64)

    # Setting initial values for the objective function and other results
    v = f_(y_k)
    mnu = np.max(v)
    f_best_value = lambda_ @ np.abs(y_k) - tau_ @ y_k + mnu
    mu = y_k
    nu = -1 * mnu

    # Iteration for finding the optimal values
    # using Nesterov's extrapolation
    for k in range(1, (max_iters + 1)):
        y_k = w_k + theta_k * ((1 / theta_k_prev) - 1) * (w_k - w_k_prev)

        # Calculating the subgradient of the objective function at y_k
        v = f_(y_k)
        idx = np.argmax(v)
        g_0 = lambda_ * np.sign(y_k) - tau_ + g_(y_k, idx)

        # Update the parameters
        theta_k_prev = theta_k
        theta_k = 2 / (k + 1)
        alpha_k = 1 / (np.power((k + 1), (3 / 2)))

        # Calculate the new points
        w_k_prev = w_k
        w_k = y_k - alpha_k * g_0

        # Check if there is an improvement
        # in the value of the objective function
        mnu = v[idx]
        f_value = lambda_ @ np.abs(y_k) - tau_ @ y_k + mnu
        if f_value < f_best_value:
            f_best_value = f_value
            mu = y_k
            nu = -1 * mnu

    # Check for possible improvement of the objective value
    # for the last generated value of w_k
    v = f_(w_k)
    mnu = np.max(v)
    f_value = lambda_ @ np.abs(w_k) - tau_ @ w_k + mnu

    if f_value < f_best_value:
        f_best_value = f_value
        mu = w_k
        nu = -1 * mnu

    # Return the optimized values in a dictionary
    new_params_ = {'w_k': w_k,
                   'w_k_prev': w_k_prev,
                   'mu': mu,
                   'nu': nu,
                   'best_value': f_best_value,
                   }

    return new_params_

def nesterov_optimization_cmrc(tau_, lambda_, m, f_, g_, max_iters):
    '''
    Solution of the CMRC convex optimization
    using the Nesterov accelerated approach.

    .. seealso:: [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                        The Strength of Nesterov’s Extrapolation
                        in the Individual Convergence of Nonsmooth
                        Optimization. IEEE transactions on
                        neural networks and learning systems,
                        31(7), 2557-2568.
                        <https://ieeexplore.ieee.org/document/8822632>`_

    Parameters
    ----------
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
        It takes the as input -
        parameters (mu) of the optimization and
        the indices corresponding to the maximum value of subobjective
        for a given subset of Y (set of labels).

    Return
    ------
    new_params_ : `dict`
        Dictionary containing optimized values: mu (`array`-like,
        shape (`m`,)) - parameters corresponding to the optimized
        function value, f_best_value (`float` - optimized value of the
        function in consideration, w_k and w_k_prev (`array`-like,
        shape (`m`,)) - parameters corresponding to the last iteration.
    '''

    # Initial values for the parameters
    theta_k = 1
    theta_k_prev = 1

    y_k = np.zeros(m, dtype=np.float64)
    w_k = np.zeros(m, dtype=np.float64)
    w_k_prev = np.zeros(m, dtype=np.float64)

    # Setting initial values for the objective function and other results
    psi, _ = f_(y_k)
    f_best_value = lambda_ @ np.abs(y_k) - tau_ @ y_k + psi
    mu = y_k

    # Iteration for finding the optimal values
    # using Nesterov's extrapolation
    for k in range(1, (max_iters + 1)):
        y_k = w_k + theta_k * ((1 / theta_k_prev) - 1) * (w_k - w_k_prev)

        # Calculating the subgradient of the objective function at y_k
        psi, psi_grad = f_(y_k)
        g_0 = lambda_ * np.sign(y_k) - tau_ + psi_grad

        # Update the parameters
        theta_k_prev = theta_k
        theta_k = 2 / (k + 1)
        alpha_k = 1 / (np.power((k + 1), (3 / 2)))

        # Calculate the new points
        w_k_prev = w_k
        w_k = y_k - alpha_k * g_0

        # Check if there is an improvement
        # in the value of the objective function
        f_value = lambda_ @ np.abs(y_k) - tau_ @ y_k + psi
        if f_value < f_best_value:
            f_best_value = f_value
            mu = y_k

    # Check for possible improvement of the objective valu
    # for the last generated value of w_k
    psi, _ = f_(w_k)
    f_value = lambda_ @ np.abs(w_k) - tau_ @ w_k + psi

    if f_value < f_best_value:
        f_best_value = f_value
        mu = w_k

    # Return the optimized values in a dictionary
    new_params_ = {'w_k': w_k,
                   'w_k_prev': w_k_prev,
                   'mu': mu,
                   'best_value': f_best_value
                   }

    return new_params_

def nesterov_optimization_minimized_mrc(F, b, tau_, lambda_, max_iters):
    '''
    Solution of the MRC convex optimization (minimization)
    using an optimized version of the Nesterov accelerated approach.

    .. seealso::         [1] `Tao, W., Pan, Z., Wu, G., & Tao, Q. (2019).
                            The Strength of Nesterov’s Extrapolation in
                            the Individual Convergence of Nonsmooth
                            Optimization. IEEE transactions on
                            neural networks and learning systems,
                            31(7), 2557-2568.
                            <https://ieeexplore.ieee.org/document/8822632>`_

    Parameters
    ----------
    M : `array`-like of shape (:math:`m_1`, :math:`m_2`)
        Where :math:`m_1` is approximately
        :math:`(2^{\\textrm{n_classes}}-1) *
        \\textrm{min}(5000,\\textrm{len}(X))`,
        where the second factor is the number of training samples used for
        solving the optimization problem.

    h : `array`-like of shape (:math:`m_1`,)
        Where :math:`m_1` is approximately
        :math:`(2^{\\textrm{n_classes}}-1) *
        \\textrm{min}(5000,\\textrm{len}(X))`,
        where the second factor is the number of training samples used for
        solving the optimization problem.

    Returns
    ------
    new_params_ : `dict`
        Dictionary that stores the optimal points
        (`w_k`: `array-like` shape (`m`,), `w_k_prev`: `array-like`
         shape (`m`,)), where `m` is the length of the feature
        mapping vector, and best value
        for the upper bound (`best_value`: `float`) of the function and
        the parameters corresponding to the optimized function value
        (`mu`: `array-like` shape (`m`,),
        `nu`: `float`).
    '''
    b = np.reshape(b, (-1, 1))
    n, m = F.shape
    a = np.reshape(-tau_, (-1, 1))  # make it a column
    mu_k = np.zeros((m, 1))
    c_k = 1
    theta_k = 1
    nu_k = 0
    alpha = F @ a
    G = F @ F.transpose()
    H = 2 * F @ np.diag(lambda_)
    y_k = mu_k
    v_k = F @ mu_k + b
    w_k = v_k
    s_k = np.sign(mu_k)
    d_k = (1 / 2) * H @ s_k
    i_k = np.argmax(v_k)
    mu_star = mu_k
    v_star = -v_k[i_k]
    lambda_ = np.reshape(lambda_, (-1, 1))  # make it a column
    f_star = a.transpose() @ mu_k +\
        lambda_.transpose() @ np.abs(mu_k) + v_k[i_k]

    if n * n > (1024) ** 3:  # Large Dimension
        for k in range(1, max_iters + 1):
            g_k = a + lambda_ * s_k + F[[i_k], :].T
            y_k_next = mu_k - c_k * g_k
            mu_k_next = (1 + nu_k) * y_k_next - nu_k * y_k
            u_k = alpha + d_k + G[:, [i_k]]
            w_k_next = v_k - c_k * u_k
            v_k_next = (1 + nu_k) * w_k_next - nu_k * w_k
            i_k_next = np.argmax(v_k_next)
            s_k_next = np.sign(mu_k_next)
            delta_k = s_k_next - s_k

            d_k_next = d_k
            for i in range(m):
                if delta_k[i] == 2:
                    d_k_next = d_k_next + H[:, [i]]
                elif delta_k[i] == -2:
                    d_k_next = d_k_next - H[:, [i]]
                elif delta_k[i] == 1 or delta_k[i] == -1:
                    d_k_next = d_k_next + (1 / 2)\
                        * np.sign(delta_k[i]) * H[:, [i]]

            c_k_next = (k + 1) ** (-3 / 2)
            theta_k_next = 2 / (k + 1)
            nu_k_next = theta_k_next * ((1 / theta_k) - 1)
            f_k_next = a.transpose() @ mu_k_next +\
                lambda_.transpose() @ np.abs(mu_k_next) +\
                v_k_next[i_k_next]
            if f_k_next < f_star:
                f_star = f_k_next
                mu_star = mu_k_next
                v_star = -v_k_next[i_k_next]

            # Update variables
            mu_k = mu_k_next
            y_k = y_k_next
            nu_k = nu_k_next
            v_k = v_k_next
            w_k = w_k_next
            s_k = s_k_next
            d_k = d_k_next
            c_k = c_k_next
            i_k = i_k_next
            theta_k = theta_k_next

    else:  # Small Dimension

        MD = H / 2

        for k in range(1, max_iters + 1):
            g_k = a + lambda_ * s_k + F[[i_k], :].T
            y_k_next = mu_k - c_k * g_k
            mu_k_next = (1 + nu_k) * y_k_next - nu_k * y_k
            u_k = alpha + d_k + G[:, [i_k]]
            w_k_next = v_k - c_k * u_k
            v_k_next = (1 + nu_k) * w_k_next - nu_k * w_k
            i_k_next = np.argmax(v_k_next)
            s_k_next = np.sign(mu_k_next)
            delta_k = s_k_next - s_k

            index = np.where(delta_k != 0)[0]
            d_k_next = d_k + MD[:, index] @ delta_k[index]

            c_k_next = (k + 1) ** (-3 / 2)
            theta_k_next = 2 / (k + 1)
            nu_k_next = theta_k_next * ((1 / theta_k) - 1)
            f_k_next = a.transpose() @ mu_k_next +\
                lambda_.transpose() @ np.abs(mu_k_next) +\
                v_k_next[i_k_next]
            if f_k_next < f_star:
                f_star = f_k_next
                mu_star = mu_k_next
                v_star = -v_k_next[i_k_next]

            # Update variables
            mu_k = mu_k_next
            y_k = y_k_next
            nu_k = nu_k_next
            v_k = v_k_next
            w_k = w_k_next
            s_k = s_k_next
            d_k = d_k_next
            c_k = c_k_next
            i_k = i_k_next
            theta_k = theta_k_next

    new_params_ = {'w_k': w_k_next.flatten(),
                   'w_k_prev': w_k.flatten(),
                   'mu': mu_star.flatten(),
                   'nu': v_star,
                   'best_value': f_star[0][0],
                   }

    return new_params_
