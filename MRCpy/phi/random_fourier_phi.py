''' Gaussian Kernel approximated using Random Features.'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping base class
from MRCpy.phi import BasePhi


class RandomFourierPhi(BasePhi):
    '''
    Fourier features

    Features obtained by approximating the rbf kernel by
    Random Fourier Feature map -

    .. math:: z(x) = \sqrt{2/D} *
                        [\cos(w_1^t * x), ..., \cos(w_D^t * x),
                         \sin(w_1^t * x), ..., \sin(w_D^t * x)]

    where w is a vector(dimension d) of random weights
    from gaussian distribution with mean 0 and variance
    :math:`1/\sigma` and
    D is the number of components in the resulting feature map.
    The parameter :math:`\sigma`
    in the variance is similar to the scaling parameter
    of the radial basis function kernel:

    .. math:: K(x, x\') = \exp{\\frac{-\| x-x\'\|^2}{2\sigma^2}}

    Note that when using Random Fourier feature mapping, training
    and testing instances are expected to be normalized.

    .. seealso:: For more information about Random Features check:

                    [1] **Random Features:** `Rahimi, A., & Recht, B. (2007).
                    Random Features for Large-Scale Kernel Machines. In NIPS
                    (Vol. 3, No. 4, p. 5).
                    <https://people.eecs.berkeley.edu/~brecht/
                    papers/07.rah.rec.nips.pdf>`_

                For more information about MRC, one can refer to the
                following resources:

                    [2] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
                    Minimax Classification with 0-1 Loss and Performance
                    Guarantees. Advances in Neural Information Processing
                    Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

                    [3] `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
                    Generalized Maximum Entropy for Supervised Classification.
                    arXiv preprint arXiv:2007.05447.
                    <https://arxiv.org/abs/2007.05447>`_

                    [4] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021).
                    MRCpy: A Library for Minimax Risk Classifiers.
                    arXiv preprint arXiv:2108.01952.
                    <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    n_classes : `int`
        Number of classes in the dataset.

    fit_intercept : `bool`, default = `True`
        Whether to calculate the intercept.
        If set to false, no intercept will be used in calculations
        (i.e. data is expected to be already centered).

    one_hot : `bool`, default = `False`
        Controls the method used for evaluating the features of the
        given instances in the binary case.
        Only applies in the binary case, namely, only when there are two
        classes. If set to true, one-hot-encoding will be used. If set to
        false a more efficient shorcut will be performed.

    sigma : `str` or `float`, default = 'scale'
        When given a string, it defines the type of heuristic to be used
        to calculate the scaling parameter `sigma` using the data.
        For comparison its relation with parameter `gamma` used in
        other methods is :math:`\gamma=1/(2\sigma^2)`.
        When given a float, it is the value for the scaling parameter.

        'scale'
            Approximates `sigma` by
            :math:`\sqrt{\\frac{\\textrm{n_features} * \\textrm{var}(X)}{2}}`
            so that `gamma` is
            :math:`\\frac{1}{\\textrm{n_features} * \\textrm{var}(X)}`
            where `var` is the variance function.

        'scale2'
            Approximates `sigma` by
            :math:`\sqrt{\\frac{\\textrm{n_features}}{2}}`
            so that `gamma` is
            :math:`\\frac{1}{\\textrm{n_features}}`
            where `var` is the variance function.

        'avg_ann_50'
            Approximates `sigma` by the average distance to the
            :math:`50^{\\textrm{th}}`
            nearest neighbour estimated from 1000 samples of the dataset using
            the function `rff_sigma`.

    n_components : `int`, default = `600`
        Number of features which the transformer transforms the input into.

    random_state : `int`, `RandomState` instance, default = None
        Random seed used to produce the `random_weights_`
        used for the approximation of the gaussian kernel.

    Attributes
    ----------
    random_weights_ : `array`-like of shape (`n_features`, `n_components`/2)
        Random weights applied to the training samples as a step for
        computing the random Fourier features.

    is_fitted_ : `bool`
        Whether the feature mappings has learned its hyperparameters (if any)
        and the length of the feature mapping is set.

    len_ : `int`
        Length of the feature mapping vector.

    '''

    def __init__(self, n_classes, fit_intercept=True, sigma='scale',
                 n_components=600, random_state=None, one_hot=False):

        # Call the base class init function.
        super().__init__(n_classes=n_classes, fit_intercept=fit_intercept,
                         one_hot=one_hot)

        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, Y=None):
        '''
        Learns the set of random weights for computing the features.
        Also, compute the scaling parameter if the value is not given.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances
            used to learn the feature configurations.

        Y : `array`-like of shape (`n_samples`,), default = `None`
            This argument will never be used in this case.
            It is present in the signature for consistency
            in the signature of the function among different feature mappings.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X = check_array(X, accept_sparse=True)

        d = X.shape[1]
        # Evaluate the sigma according to the sigma type given in self.sigma
        if self.sigma == 'scale':
            self.sigma_val = np.sqrt((d * X.var()) / 2)

        elif self.sigma == 'scale2':
            self.sigma_val = np.sqrt(d / 2)

        elif self.sigma == 'avg_ann_50':
            self.sigma_val = self.rff_sigma(X)

        elif type(self.sigma) != str:
            self.sigma_val = self.sigma

        else:
            raise ValueError('Unexpected value for sigma ...')

        # Obtain the random weight from a normal distribution.
        self.random_state_ = check_random_state(self.random_state)
        self.random_weights_ = \
            self.random_state_.normal(0, 1 / self.sigma_val,
                                      size=(d, int(self.n_components / 2)))

        # Sets the length of the feature mapping
        super().fit(X, Y)

        return self

    def transform(self, X):
        '''
        Compute the random Fourier features ((:math:`z(x)`)).

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled training instances.

        Returns
        -------
        X_feat : `array`-like of shape (`n_samples`, `n_features`)
            Transformed features from the given instances.

        '''

        check_is_fitted(self, ["random_weights_", "is_fitted_"])
        X = check_array(X, accept_sparse=True)

        X_trans = X @ self.random_weights_
        X_feat = (1 / np.sqrt(int(self.n_components / 2))) * \
            np.hstack((np.cos(X_trans), np.sin(X_trans)))

        return X_feat

    def rff_sigma(self, X):

        '''
        Computes the scaling parameter for the fourier features
        using the heuristic given in the paper "Compact Nonlinear Maps
        and Circulant Extensions" :ref:`[1] <refpf>`.

        The heuristic states that the scaling parameter is obtained as
        the average distance to the 50th nearest neighbour estimated
        from 1000 samples of the dataset.

        .. _refpf:
        .. seealso:: [1] `Yu, F. X., Kumar, S., Rowley, H., & Chang, S. F.
                        (2015). Compact nonlinear maps and circulant
                        extensions.
                        <https://arxiv.org/pdf/1503.03893.pdf>`_

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Unlabeled instances.

        Returns
        -------
        sigma : `float` value
            Scaling parameter computed using the heuristic.
        '''
        if X.shape[0] < 50:
            neighbour_ind = X.shape[0] - 2
        else:
            neighbour_ind = 50

        # Find the nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=(neighbour_ind + 1),
                                algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Compute the average distance to the 50th nearest neighbour
        sigma = np.average(distances[:, neighbour_ind])

        return sigma
