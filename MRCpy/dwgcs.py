'''Double-Weighting General Covariate Shift Adaptation using MRCs.'''

import numpy as np
import cvxpy as cvx
import sklearn as sk
from sklearn import preprocessing
from sklearn.base import check_X_y, check_array

# Import the DWGCS super class
from MRCpy import CMRC
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi

class DWGCS(CMRC):
    '''Double-Weighting General Covariate Shift

    The class DWGCS implements the method Double-Weighting for 
    General Covariate Shift (DWGCS) proposed in :ref:`[1] <ref1>`. 
    It is designed for supervised classification under covariate shift. 

    DW-GCS provides adaptatition to covariate shift when a set of training samples
    and a set of unlabeled samples from the testing distribution are available
    at learning, without any prior knowledge about the supports of the training
    and testing distribution.
    
    It implements 0-1 and log loss, and it can be used with multiple feature
    mappings.

    .. seealso:: For more information about DWGCS, one can refer to the
        following paper:

                    [1] `Segovia-Martín, J. I., Mazuelas, S., & Liu, A. (2023).
                    Double-Weighting for Covariate Shift Adaptation.
                    International Conference on Machine Learning (ICML) 2023.

                    @InProceedings{SegMazLiu:23,
                    title =     {Double-Weighting for Covariate Shift Adaptation},
                    author =    {Segovia-Mart{\'i}n, Jos{\'e} I. 
                                and Mazuelas, Santiago 
                                and Liu, Anqi},
                    booktitle = {Proceedings of the 40th 
                                International Conference on Machine Learning},
                    pages = 	{30439--30457},
                    year = 	    {2023},
                    volume = 	{202},
                    series = 	{Proceedings of Machine Learning Research},
                    month = 	{23--29 Jul},
                    publisher = {PMLR},
                    }
    Parameters
    ----------
    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization. 0-1 loss
        quantifies the probability of classification error at a certain example
        for a certain rule. Log-loss quantifies the minus log-likelihood at a
        certain example for a certain rule.

    deterministic : `bool`, default = `True`
       Whether the prediction of the labels
       should be done in a deterministic way (given a fixed `random_state`
       in the case of using random Fourier or random ReLU features).

    random_state : `int`, RandomState instance, default = `None`
        Random seed used when 'fourier' and 'relu' options for feature mappings
        are used to produce the random weights.

    fit_intercept : `bool`, default = `True`
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    D : `int`, default = 4
        Hyperparameter that balances the trade-off between error in
        expectation estimates and confidence of the classification.

    B : `int`, default = 1000
        Parameter that bound the maximum value of the weights
        beta associated to the training samples.

    solver : {‘cvx’, 'grad', 'adam'}, default = ’adam’
        Method to use in solving the optimization problem. 
        Default is ‘cvx’. To choose a solver,
        you might want to consider the following aspects:

        ’cvx’
            Solves the optimization problem using the CVXPY library.
            Obtains an accurate solution while requiring more time
            than the other methods. 
            Note that the library uses the GUROBI solver in CVXpy for which
            one might need to request for a license.
            A free license can be requested `here 
            <https://www.gurobi.com/academia/academic-program-and-licenses/>`_

        ’grad’
            Solves the optimization using stochastic gradient descent.
            The parameters `max_iters`, `stepsize` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.

        ’adam’
            Solves the optimization using
            stochastic gradient descent with adam (adam optimizer).
            The parameters `max_iters`, `alpha` and `mini_batch_size`
            determine the number of iterations, the learning rate and
            the batch size for gradient computation respectively.
            Note that the implementation uses nesterov's gradient descent
            in case of ReLU and threshold features, and the above parameters
            do no affect the optimization in this case.
    
    alpha : `float`, default = `0.001`
        Learning rate for ’adam’ solver.

    mini_batch_size : `int`, default = `1` or `32`
        The size of the batch to be used for computing the gradient
        in case of stochastic gradient descent and adam optimizer.
        In case of stochastic gradient descent, the default is 1, and
        in case of adam optimizer, the default is 32.

    max_iters : `int`, default = `100000` or `5000` or `2000`
        The maximum number of iterations to use in case of
        ’grad’ or ’adam’ solver.
        The default value is
        100000 for ’grad’ solver and
        5000 for ’adam’ solver and 
        2000 for nesterov's gradient descent.   

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier', 'relu', and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' feature mapping,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

        'relu'
            It uses Rectified Linear Unit (ReLU) features.
            See class `RandomReLUPhi`.

    **phi_kwargs : Additional parameters for feature mappings.
                Groups the multiple optional parameters
                for the corresponding feature mappings(`phi`).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument -
                `DWGCS(loss='log', phi='fourier', n_components=300)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.
    '''

    def __init__(self,
                 loss='0-1',
                 deterministic=True,
                 random_state=None,
                 fit_intercept=False,
                 D = 4,
                 B = 1000,
                 solver='adam',
                 alpha=0.01,
                 stepsize='decay',
                 mini_batch_size=None,
                 max_iters=None,
                 phi='linear',
                 **phi_kwargs):
        self.D = D
        self.B = B
        super().__init__(loss,
                         None,
                         deterministic,
                         random_state,
                         fit_intercept,
                         solver,
                         alpha,
                         stepsize,
                         mini_batch_size,
                         max_iters,
                         phi,
                         **phi_kwargs)
    
    def fit(self, xTr, yTr, xTe=None):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used in

            - Calculating the expectation estimates
              that constrain the uncertainty set
              for the minimax risk classification
            - Solving the minimax risk optimization problem.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        yTr : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        xTe : array-like of shape (`n_samples2`, `n_dimensions`), default = None
            These instances will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        Returns
        -------
        self :
            Fitted estimator
        '''

        xTr, yTr = check_X_y(xTr, yTr, accept_sparse=True)

        # Check if separate instances are given for the optimization
        if xTe is None:
            raise ValueError('Missing instances from testing distribution ... ')
        else:
            xTe = check_array(xTe, accept_sparse=True)

        # Obtaining the number of classes and mapping the labels to integers
        origY = yTr
        self.classes_ = np.unique(origY)
        n_classes = len(self.classes_)
        yTr = np.zeros(origY.shape[0], dtype=int)

        # Map the values of Y from 0 to n_classes-1
        for i, y in enumerate(self.classes_):
            yTr[origY == y] = i

        # Feature mappings
        if self.phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=n_classes,
                                        fit_intercept=self.fit_intercept,
                                        random_state=self.random_state,
                                        **self.phi_kwargs)
        elif self.phi == 'linear':
            self.phi = BasePhi(n_classes=n_classes,
                               fit_intercept=self.fit_intercept,
                               **self.phi_kwargs)
        elif self.phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=n_classes,
                                     fit_intercept=self.fit_intercept,
                                     random_state=self.random_state,
                                     **self.phi_kwargs)
        elif not isinstance(self.phi, BasePhi):
            raise ValueError('Unexpected feature mapping type ... ')

        # Fit the feature mappings
        self.phi.fit(xTr, yTr)
        
        # Compute weights alpha and beta
        self.DWKMM(xTr,xTe)

        # Compute the expectation estimates
        tau_ = self.compute_tau(xTr, yTr)
        lambda_ = self.compute_lambda(xTe, tau_, n_classes)

        # Fit the MRC classifier
        self.minimax_risk(xTe, tau_, lambda_, n_classes)

        return self
    
    def DWKMM(self,xTr,xTe):
        '''
        Obtain training and testing weights.

        Computes the weights associated to the
        training and testing samples solving the DW-KMM problem.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used in

            - Computing the training weights beta and testing weights alpha.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Testing instances used in

            - Computing the training weights beta and testing weights alpha.

            `n_samples` is the number of training samples and
            `n_dimensions` is the number of features.

        Returns
        -------
        self :
            Weights self.beta_ and self.alpha_
        '''

        n = xTr.shape[0]
        t = xTe.shape[0]
        x = np.concatenate((xTr, xTe), axis=0)
        epsilon_ = 1 - 1 / (np.sqrt(n))
        
        self.sigma_ = RandomFourierPhi(self.classes_).rff_sigma(preprocessing.StandardScaler().fit_transform(x))
        K = sk.metrics.pairwise.rbf_kernel(x, x, 1 / (2 * sigma_ ** 2))

        # Define the variables of the opt. problem
        beta_ = cvx.Variable((n, 1))
        alpha_ = cvx.Variable((t, 1))
        # Define the objetive function
        objective = cvx.Minimize(cvx.quad_form(cvx.vstack([beta_/n, -alpha_/t]), cvx.psd_wrap(K)))
        # Define the constraints
        constraints = [ 
            beta_ >= np.zeros((n, 1)),
            beta_ <= (self.B / np.sqrt(self.D)) * np.ones((n, 1)),
            alpha_ >= np.zeros((t, 1)),
            alpha_ <= np.ones((t, 1)),
            cvx.abs(cvx.sum(beta_) / n - cvx.sum(alpha_) / t) <= epsilon_,
            cvx.norm(alpha_ - np.ones((t, 1))) <= (1 - 1 / np.sqrt(self.D)) * np.sqrt(t)
        ]
        problem = cvx.Problem(objective,constraints)
        problem.solve()

        self.beta_ = beta_.value
        self.alpha_ = alpha_.value
        self.min_DWKMM = problem.value

        return self


    def compute_tau(self, xTr, yTr):
        '''
        Compute mean estimate tau using the given training instances.

        Parameters
        ----------
        xTr : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        yTr : `array`-like of shape (`n_samples`, 1), default = `None`
            Labels corresponding to the training instances
            used only to compute the expectation estimates.

        Returns
        -------
        tau_ :
            Mean expectation estimate
        '''

        phiMatrix = self.phi.eval_xy(xTr, yTr)
        phi_betaMatrix = np.multiply(self.beta_, phiMatrix)
        tau_ = np.mean(phi_betaMatrix, axis = 0)

        return tau_
    
    def compute_lambda(self, xTe, tau_, n_classes):
        '''
        Compute deviation in the mean estimate tau
        using the given testing instances.

        Parameters
        ----------
        xTe : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.
        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            The mean estimates
            for the expectations of feature mappings.
        n_classes : `int`
            Number of labels in the dataset.
        
        Returns
        -------
        lambda_ :
            Confidence vector
        '''
        
        d = self.phi.len_
        t = xTe.shape[0]
        delta_ = 1e-6 * np.ones(d)

        # Define the variables of the opt. problem
        lambda_ = cvx.Variable(d)
        p = cvx.Variable((t * n_classes,1))
        # Define the objetive function
        objective = cvx.Minimize(cvx.sum(lambda_))
        # Construct constraints
        phiMatrix = self.phi.eval_x(xTe)
        phiMatrix_2d = np.reshape(phiMatrix, (t * n_classes, d))
        alpha_rep = np.reshape(np.repeat(self.alpha_, n_classes),(t * n_classes, 1))
        phi_alphaMatrix = np.multiply(alpha_rep, phiMatrix_2d)
        # Define the constraints
        constraints = [
            cvx.reshape(tau_ - lambda_ + delta_, (d,1)) <= phi_alphaMatrix.T @ p,
            phi_alphaMatrix.T @ p <= cvx.reshape(tau_ + lambda_ - delta_, (d,1)),
            lambda_ >= np.zeros(d),
            cvx.sum(cvx.reshape(p, (n_classes, t)),axis=0) == np.ones(t) / t,
            p >= np.zeros((t * n_classes, 1))
        ]

        problem = cvx.Problem(objective, constraints)
        problem.solve()

        lambda_ = np.maximum(lambda_.value, 0)

        return lambda_
    
    def compute_phi(self, X):
        '''
        Compute the feature mapping corresponding to instances given
        for learning the classifiers and prediction.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Instances to be converted to features.

        Returns
        -------
        phi_alpha :
            Feature mapping weighted by alpha
        '''

        t = X.shape[0]
        d = self.phi.len_

        phiMatrix = self.phi.eval_x(X)
        alpha_rep = np.reshape(np.repeat(self.alpha_, d),(t, 1, d))
        phi_alpha = np.multiply(alpha_rep, phiMatrix)

        return phi_alpha
