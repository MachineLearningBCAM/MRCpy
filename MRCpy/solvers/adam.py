''' Gradient descent optimization with adam '''

import numpy as np

def adam(tau_, lambda_, n, m, f_, g_, max_iters, alpha, mini_batch_size=32, eps=1e-8):

    # Initial values for points
    w_k = np.zeros(m, dtype=np.float64)
    # initialize first and second moments
    m_ = np.zeros(m, dtype=np.float64)
    v_ = np.zeros(m, dtype=np.float64)

    # Hyperparameters
    beta1=0.9
    beta2=0.999

    # Setting the initial indices for the batch
    batch_start_sample_id = 0
    batch_end_sample_id = batch_start_sample_id + mini_batch_size
    epoch_id = 0

    # Run the gradient descent updates
    for t in range(max_iters):
        # Calculate gradient g(t)
        g_0 = lambda_ * np.sign(w_k) - tau_ + g_(w_k,
                                                 batch_start_sample_id,
                                                 batch_end_sample_id,
                                                 n)

        # Update the moments
        m_ = beta1 * m_ + (1.0 - beta1) * g_0
        v_ = beta2 * v_ + (1.0 - beta2) * g_0**2
        mhat = m_ / (1.0 - beta1**(t+1))
        vhat = v_ / (1.0 - beta2**(t+1))

        # Update the weight of MRC using the moments and learning rate
        w_k = w_k - alpha * mhat / (np.sqrt(vhat) + eps)

        # Update the batch indices
        batch_end_sample_id = batch_end_sample_id % n
        batch_start_sample_id = batch_end_sample_id
        batch_end_sample_id = batch_start_sample_id + mini_batch_size
        epoch_id += batch_end_sample_id // n

    psi, _ = f_(w_k)
    f_value = lambda_ @ np.abs(w_k) - tau_ @ w_k + psi
    mu = w_k

    # Return the optimized values in a dictionary
    new_params_ = {'w_k': w_k,
                   'mu': mu,
                   'best_value': f_value  # actually last value
                   }

    return new_params_