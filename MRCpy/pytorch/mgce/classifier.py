"""
PyTorch-based minimax risk classification with alpha-losses.

Copyright (C) 2026 Kartheek Bondugula

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see https://www.gnu.org/licenses/.
"""

import torch
import numpy as np
import copy
from torch.utils.data import ConcatDataset
from tqdm.auto import trange
import os

# Import the loss function from the local loss module
from .loss import mgce_loss

# Import sklearn utilities for validation
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Import optional dependencies with fallback
try:
    from pycalib.metrics import conf_ECE, classwise_ECE
except ImportError:
    # Provide fallback or warning if pycalib is not available
    conf_ECE = None
    classwise_ECE = None

import logging
logger = logging.getLogger(__name__)


class mgce_clf():
    r"""
    PyTorch-based Minimax Generalized Cross-Entropy (MGCE).

    This class implements a PyTorch-based Minimax Risk Classifier (MRC)
    using :math:`\alpha`-losses (:math:`\ell_\alpha`), a convex surrogate loss 
    defined over classification margins.

    The classifier solves the minimax risk problem:

    .. math::

        \mathrm{h}^{\mathcal{U}_2} \in \arg\min_{\mathrm{h}}
        \max_{\mathrm{p} \in \mathcal{U}_2} \ell(\mathrm{h}, \mathrm{p})

    which finds the classifier :math:`\mathrm{h}` that minimizes the
    worst-case expected :math:`\ell`-loss over an uncertainty set
    :math:`\mathcal{U}_2` of distributions. This provides theoretical
    performance guarantees even when the true data distribution is
    unknown.

    The uncertainty set :math:`\mathcal{U}_2` is defined by fixing the
    marginal distribution of features to the empirical marginal and
    bounding the expectations of the feature mappings
    :math:`\Phi(x, y)`:

    .. math::

        \mathcal{U}_2 = \left\{ \mathrm{p} : \mathrm{p}_x =
        \hat{\mathrm{p}}_x, \;
        \left| \mathbb{E}_{\mathrm{p}}[\Phi(x,y)]
        - \boldsymbol{\tau} \right| \leq \boldsymbol{\lambda} \right\}

    where :math:`\hat{\mathrm{p}}_x` is the empirical marginal
    distribution of features, :math:`\boldsymbol{\tau}` are the empirical mean
    estimates, and :math:`\boldsymbol{\lambda}` controls the size of the uncertainty
    set based on the estimation accuracy.

    The :math:`\alpha`-loss family, indexed by :math:`\alpha \geq 1`, smoothly
    interpolates between well-known losses:

    - :math:`\alpha = 1`: the **0-1 loss** :math:`\ell_1(\mathrm{h}, y) = 1 - \mathrm{h}_y(x)`,
      which directly measures classification error but is non-smooth.
    - :math:`\alpha \to \infty`: the **log-loss**
      :math:`\ell_\infty(\mathrm{h}, y) = -\log \mathrm{h}_y(x)` (cross-entropy),
      which is smooth and easy to optimize.

    For intermediate :math:`\alpha > 1`:

    .. math::

        \ell_\alpha(\mathrm{h}(x), y) = \frac{\alpha}{\alpha - 1}
        \left(1 - \mathrm{h}_y(x)^{(\alpha-1)/\alpha}\right)

    Smaller :math:`\alpha` values stay closer to the 0-1 loss (more robust),
    while larger values behave more like log-loss (smoother optimization).

    See [1]_ for further details.
    Following [1]_, the implementation uses the parameter
    :math:`\beta = \alpha / (\alpha-1)`, so that :math:`\beta = 1`
    corresponds to the 0-1 loss (:math:`\alpha \to \infty`) and
    :math:`\beta \to \infty` corresponds to the log-loss
    (:math:`\alpha = 1`). The ``loss_parameter`` argument
    accepts :math:`\beta` values directly.

    Parameters
    ----------
    loss_parameter : float, default=1.4
        The :math:`\beta = \alpha / (\alpha-1)` parameter for the
        :math:`\alpha`-loss function. When beta=0 (alpha=1), it corresponds to
        0-1 loss. Larger beta values provide smoother approximations
        closer to log-loss (cross-entropy). Typical values are 
        in the range [1.0, 11.0].

    lambda_ : float, default=1e-5
        L1 regularization strength applied to model parameters.
        In the minimax framework, large lambda implies bigger uncertainty sets
        with better generalization. The implementation uses the same lambda
        for all the model parameters (features).

    deterministic : bool, default=True
        Whether predictions should be deterministic. When True, uses argmax
        for prediction. When False, samples from the predicted probability
        distribution.

    random_state : int or None, default=None
        Random seed for reproducible results. Used for weight initialization
        and stochastic operations.

    optimizer : torch.optim.Optimizer or None, default=None
        PyTorch optimizer instance for training. If None, must be provided
        during training or set as an attribute before calling fit().

    scheduler : torch.optim.lr_scheduler or None, default=None
        Learning rate scheduler for adaptive learning rate adjustment
        during training. Optional parameter.

    model : torch.nn.Module or None, default=None
        PyTorch neural network model to be trained. Must be provided
        either during initialization or before calling fit().

    device : str, default='cuda'
        Device for computation ('cuda' for GPU, 'cpu' for CPU).
        Automatically falls back to CPU if CUDA is not available.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the classifier has been fitted to training data.

    loss_parameter : str or float
        The :math:`\beta` parameter for the loss function.

    lambda_ : float
        L1 regularization strength.

    model : torch.nn.Module
        The neural network model being trained.

    optimizer : torch.optim.Optimizer
        Optimizer used for training.

    device : str
        Device used for computation.

    Examples
    --------
    Basic usage with a simple neural network:

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> from MRCpy.pytorch.mgce.classifier import mgce_clf
    >>>
    >>> # Create a simple model
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 64),
    ...     nn.ReLU(),
    ...     nn.Linear(64, 3)
    ... )
    >>>
    >>> # Create optimizer
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>>
    >>> # Initialize classifier with beta=1.4
    >>> clf = mgce_clf(
    ...     loss_parameter=1.4,
    ...     lambda_=1e-5,
    ...     model=model,
    ...     optimizer=optimizer,
    ...     device='cuda'
    ... )
    >>>
    >>> # Create dummy data
    >>> X = torch.randn(100, 10)
    >>> y = torch.randint(0, 3, (100,))
    >>> dataset = TensorDataset(X, y)
    >>> dataloader = DataLoader(dataset, batch_size=32)
    >>>
    >>> # Train the model
    >>> results = clf.fit(dataloader, n_epochs=10, verbose=True)
    >>> print(f"Final training accuracy: {results['train_acc'][-1]:.2f}%")

    Training with validation:

    >>> results = clf.fit(
    ...     train_loader,
    ...     validate=True,
    ...     val_dataloader=val_loader,
    ...     save_model_weights='best'
    ... )
    >>> print(f"Best validation accuracy: {max(results['val_acc']):.2f}%")

    Compute probabilities for test samples:

    >>> test_data = torch.randn(10, input_dim)
    >>> probabilities = clf.predict_proba(test_data)
    >>> print(f"Shape: {probabilities.shape}")  # (10, n_classes)
    >>> print(f"First sample probabilities: {probabilities[0]}")

    Deterministic predictions:

    >>> predictions = clf.predict(test_data)
    >>> print(f"Predicted classes: {predictions}")

    Stochastic predictions:

    >>> clf_stochastic = mgce_clf(deterministic=False, ...)
    >>> clf_stochastic.fit(train_loader)
    >>> pred1 = clf_stochastic.predict(test_data)
    >>> pred2 = clf_stochastic.predict(test_data)  # May differ from pred1

    See Also
    --------
    MRCpy.pytorch.mgce.loss.mgce_loss : :math:`\alpha`-loss function used by this classifier.

    References
    ----------
    .. [1] Bondugula, K., Mazuelas, S., Pérez, A., & Liu, A. (2026).
           Minimax Generalized Cross-Entropy. In Proceedings of the
           International Conference on Artificial Intelligence and
           Statistics (AISTATS).
    """

    def __init__(self,
                 loss_parameter='1.4',
                 lambda_=1e-5,
                 deterministic=True,
                 random_state=None,
                 optimizer=None,
                 scheduler=None,
                 model=None,
                 device='cuda'):
        self.loss_parameter = loss_parameter
        self.lambda_ = lambda_
        self.deterministic = deterministic
        self.random_state = random_state
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.device = device

    def fit(self, train_dataloader, pretrained=False, grad_bound=5.0, n_epochs=100,
            verbose=True, validate=False, val_dataloader=None,
            compute_ece=True, bins=15, save_model_weights='best', path="./"):
        """
        Fit the MRC model using the provided training data.

        This method trains the neural network using the minimax risk
        optimization corresponding with :math:`\alpha`-losses. It supports both training-only
        and training-with-validation modes.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader containing the training data. Each batch should return
            (inputs, labels) where inputs are feature tensors and labels are
            class indices.
            
        pretrained : bool, default=False
            Whether to use a pretrained model. If True, assumes the model
            has been pre-trained and may adjust training parameters accordingly.
            
        grad_bound : float, default=5.0
            Maximum gradient norm for gradient clipping. Helps prevent
            gradient explosion during training.
            
        n_epochs : int, default=100
            Number of training epochs to run.
            
        verbose : bool, default=True
            Whether to print training progress and metrics during training.
            
        validate : bool, default=False
            Whether to perform validation during training. If True,
            val_dataloader must be provided.
            
        val_dataloader : torch.utils.data.DataLoader or None, default=None
            DataLoader containing validation data. Required if validate=True.
            Should have the same format as train_dataloader.
            
        compute_ece : bool, default=True
            Whether to compute Expected Calibration Error (ECE) during
            validation. Only used if validate=True.
            
        bins : int, default=15
            Number of bins to use for ECE computation. Only used if
            compute_ece=True and validate=True.
            
        save_model_weights : {'best', 'last', 'None'}, default='best'
            Strategy for saving model weights:
            - 'best': Save weights from epoch with highest validation accuracy
            - 'last': Save weights from the final epoch
            - 'None': Don't save any weights
            
        path : str, default="./"
            Directory path where model weights should be saved (if applicable).

        Returns
        -------
        dict
            Dictionary containing training metrics:
            - 'train_loss': List of training losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_loss': List of validation losses per epoch (if validate=True)
            - 'val_acc': List of validation accuracies per epoch (if validate=True)

        Raises
        ------
        ValueError
            If invalid save_model_weights option is provided or if validation
            is enabled but val_dataloader is None.
            
        RuntimeError
            If model or optimizer are not properly initialized.

        Examples
        --------
        See class-level examples.
        """
        # Validate inputs
        if validate and val_dataloader is None:
            raise ValueError("val_dataloader must be provided when validate=True")
        
        if self.model is None:
            raise RuntimeError("Model must be provided before calling fit()")
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be provided before calling fit()")

        # Initialize the alpha-loss function with the number of classes and beta parameter
        loss_function = mgce_loss(len(train_dataloader.dataset.classes), self.loss_parameter)
        
        # Initialize arrays to store training metrics for each epoch
        train_loss_arr = []
        train_acc_arr = []

        # Initialize variables for model checkpointing and best model tracking
        best_epoch = n_epochs - 1
        best_val_acc = 0
        best_model_weights = None

        # Initialize validation metric arrays if validation is enabled
        if validate:
            val_loss_arr = []
            val_acc_arr = []
            val_ece_arr = []

        # Small epsilon value to prevent division by zero in regularization
        epsilon = 1e-6

        # Main training loop over epochs with optional progress bar
        for epoch in trange(n_epochs, disable=not verbose, desc="Training"):

            # Initialize training statistics for the current epoch
            train_loss = 0
            total_train_samples = 0
            correct_train = 0

            # Set model to training mode for proper batch normalization and dropout behavior
            self.model.train()
            
            # Iterate through training batches
            for inputs, labels in train_dataloader:
                # Move data to the specified device (GPU/CPU) with non-blocking transfer for efficiency
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Clear gradients from previous iteration
                self.model.zero_grad()
                self.optimizer.zero_grad()
               
                # Forward pass: compute model predictions
                logits = self.model(inputs)

                # Compute alpha-loss gradients and loss
                grad, loss = loss_function.get_gradient(logits, labels)
                
                # Backward pass: compute gradients using the custom alpha-loss gradients
                logits.backward(grad)

                # Apply L1 regularization manually since alpha-loss requires custom regularization
                # Select appropriate parameters based on whether using pretrained model
                if pretrained:
                    # For pretrained models, only regularize the final classification layer
                    model_parameters = self.model.fc.parameters()
                else:
                    # For models trained from scratch, regularize all parameters
                    model_parameters = self.model.parameters()

                # Initialize regularization value for this batch
                reg_val = 0
                
                # Apply L1 regularization to gradients and compute regularization loss
                for param in model_parameters:
                    if param.grad is not None:
                        # Add L1 regularization term to gradients
                        param.grad += self.lambda_ * param / (param.abs() + epsilon)
                    # Accumulate L1 regularization loss
                    reg_val += self.lambda_ * param.abs().sum()

                # Apply gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model_parameters, grad_bound)
                
                # Update model parameters using the optimizer
                self.optimizer.step()

                # Accumulate training statistics for this batch
                batch_size = logits.shape[0]
                train_loss += (loss.item() + reg_val) * batch_size
                total_train_samples += batch_size

                # Compute training accuracy for this batch
                _, predicted = logits.max(1)
                correct_train += predicted.eq(labels).sum().item()
            
            # Calculate average training metrics for this epoch
            avg_train_loss = train_loss / total_train_samples
            avg_train_acc = 100 * correct_train / total_train_samples

            # Store training metrics for this epoch
            train_loss_arr.append(avg_train_loss)
            train_acc_arr.append(avg_train_acc)

            # Log training progress if verbose mode is enabled
            if verbose:
                logger.info(
                    "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.2f%%",
                    epoch + 1,
                    n_epochs,
                    avg_train_loss,
                    avg_train_acc
                )

            # Perform validation if enabled
            if validate:

                # Initialize validation statistics for the current epoch
                val_loss = 0
                total_val_samples = 0
                correct_val = 0
                probs = []  # Store probability predictions for ECE computation
                y_true = []  # Store true labels for ECE computation

                # Set model to evaluation mode to disable dropout and batch norm updates
                self.model.eval()
                
                # Disable gradient computation for validation to save memory and computation
                with torch.no_grad():
                    # Compute regularization value from current model parameters
                    if pretrained:
                        # For pretrained models, only consider final layer parameters
                        model_parameters = self.model.fc.parameters()
                    else:
                        # For models trained from scratch, consider all parameters
                        model_parameters = self.model.parameters()

                    # Calculate total L1 regularization penalty
                    reg_val = sum(self.lambda_ * param.abs().sum() for param in model_parameters)

                    # Iterate through validation batches
                    for inputs, labels in val_dataloader:
                        # Move validation data to device
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
            
                        # Forward pass: compute model predictions
                        logits = self.model(inputs)
                        
                        # Compute validation loss and probability predictions using alpha-loss
                        loss, probs_batch_i = loss_function.get_loss_value(logits, labels, reg_val)
                        
                        # Store predictions and labels for ECE computation
                        probs.append(probs_batch_i)
                        y_true.append(labels)

                        # Accumulate validation loss (includes regularization from get_loss_value)
                        batch_size = inputs.size(0)
                        val_loss += (loss.item()) * batch_size
                        total_val_samples += batch_size

                        # Compute validation accuracy for this batch
                        _, predicted = logits.max(1)
                        correct_val += predicted.eq(labels).sum().item()

                # Calculate average validation metrics for this epoch
                avg_val_loss = val_loss / total_val_samples
                avg_val_acc = 100 * correct_val / total_val_samples

                # Store validation metrics
                val_loss_arr.append(avg_val_loss)
                val_acc_arr.append(avg_val_acc)

                # Compute Expected Calibration Error (ECE) if requested
                if compute_ece and conf_ECE is not None:
                    # Calculate ECE using concatenated predictions and true labels
                    conf_ece = conf_ECE(
                                        np.concatenate([y.cpu().numpy() for y in y_true]),
                                        np.concatenate([p.cpu().numpy() for p in probs]),
                                        bins=bins
                                    ) * 100  # Convert to percentage
                    val_ece_arr.append(conf_ece)
                else:
                    conf_ece = None

                # Log validation progress if verbose mode is enabled
                if verbose:
                    if compute_ece and conf_ece is not None:
                        logger.info(
                            "Epoch [%d/%d] | Val Loss: %.4f | Val Acc: %.2f%% | Val ECE: %.2f%%",
                            epoch + 1, n_epochs, avg_val_loss, avg_val_acc, conf_ece
                        )
                    else:
                        logger.info(
                            "Epoch [%d/%d] | Val Loss: %.4f | Val Acc: %.2f%%",
                            epoch + 1, n_epochs, avg_val_loss, avg_val_acc
                        )

                # Update best model if current validation accuracy is better
                if save_model_weights == 'best' and avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    best_epoch = epoch
                    best_model_weights = copy.deepcopy(self.model.state_dict())

        # Handle model weight saving based on the specified strategy
        if save_model_weights == 'last':
            # Save weights from the current (last) epoch
            best_model_weights = copy.deepcopy(self.model.state_dict())
        elif save_model_weights == 'best' and validate is False:
            # Save weights from the current (last) epoch since validation set is not available
            print('Validation set is not provided for best model weights. \
                   Saving the last model weights')
            best_model_weights = copy.deepcopy(self.model_state.dict())
        elif save_model_weights not in ['best', 'last', 'None', None]:
            # Raise error for invalid save_model_weights options
            raise ValueError(f"Invalid value for saving model weights: {save_model_weights}. "
                            "Valid options are: 'best', 'last', 'None'")

        # Save model weights to file if a saving strategy is specified
        if save_model_weights in ['best', 'last'] and best_model_weights is not None:
            # Create output directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Generate filename with epoch information
            save_file = os.path.join(path, f"mrc_model_epoch_{best_epoch + 1}.pt")
            
            # Save the model state dictionary to file
            torch.save(best_model_weights, save_file)
            
            # Log the save location if verbose mode is enabled
            if verbose:
                logger.info("Model weights saved at: %s", save_file)

        # Mark classifier as fitted
        self.is_fitted_ = True

        # Prepare results dictionary with training metrics
        results = {
            "train_loss": train_loss_arr,
            "train_acc": train_acc_arr
        }
        
        # Add validation metrics to results if validation was performed
        if validate:
            results.update({
                "val_loss": val_loss_arr,
                "val_acc": val_acc_arr,
                "val_ece": val_ece_arr
            })

        return results
        
    def predict_proba(self, X):
        """
        Compute class probabilities for the given input samples.

        This method computes the conditional probabilities p(y|x) for each
        class given the input features. The probabilities are computed using
        the trained model and the minimax risk framework with :math:`\alpha`-losses
        via the loss function's get_probs method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to compute class probabilities.
            Should be preprocessed in the same way as the training data.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities for each input sample. Each row sums to 1.0
            and represents the probability distribution over all classes
            for the corresponding input sample, computed using the :math:`\alpha`-loss
            framework.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet (i.e., fit() has not
            been called).
            
        ValueError
            If the input data format is invalid or incompatible with the
            trained model.

        Examples
        --------
        See class-level examples.

        Notes
        -----
        This method uses the :math:`\alpha`-loss framework's get_probs function to compute
        probabilities, ensuring consistency with the training loss function
        rather than using standard softmax normalization.
        """
        # Validate input data format and check if model has been fitted
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Convert to tensor if needed and move to device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        
        # Compute probabilities without gradient computation
        with torch.no_grad():
            logits = self.model(X)
            
            # Initialize the alpha-loss function to use get_probs
            # We need to determine the number of classes from the logits
            num_classes = logits.shape[1]
            loss_function = mgce_loss(num_classes, self.loss_parameter)
            
            # Use alpha-loss framework to compute probabilities
            probabilities = loss_function.get_probs(logits)
            
        # Convert back to numpy and return
        return probabilities.cpu().numpy()

    def get_worst_case_probs(self, X):
        r"""
        Compute worst-case class probabilities.

        This method computes the worst-case probability distribution
        :math:`\mathrm{p}^*` corresponding to the minimax solution.
        Given the prediction probabilities :math:`\mathrm{h}(x)` from
        :meth:`predict_proba`, the worst-case probabilities are:

        .. math::

            \mathrm{p}^*_y(x) = \frac{\mathrm{h}_y(x)^{(\beta-1)/\beta}}
            {\sum_{y'} \mathrm{h}_{y'}(x)^{(\beta-1)/\beta}}

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to compute worst-case probabilities.
            Should be preprocessed in the same way as the training data.

        Returns
        -------
        worst_case_probs : ndarray of shape (n_samples, n_classes)
            Worst-case class probabilities for each input sample.
            Each row sums to 1.0.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet.

        ValueError
            If the input data format is invalid or incompatible with the
            trained model.

        See Also
        --------
        predict_proba : Compute prediction probabilities :math:`\mathrm{h}(x)`.

        Notes
        -----
        The figure below illustrates the relationship between worst-case
        and prediction probabilities for different :math:`\beta` values.

        .. figure:: /_static/worst_case_probs_relation.png
           :alt: Relationship between worst-case and prediction probabilities
           :width: 400px
           :align: center
        """
        # Get prediction probabilities
        h = self.predict_proba(X)
        h = torch.tensor(h, dtype=torch.float32)

        # Compute h^{(beta-1)/beta}
        exponent = (self.loss_parameter - 1) / self.loss_parameter
        h_pow = h.pow(exponent)

        # Normalize
        worst_case_probs = h_pow / h_pow.sum(dim=1, keepdim=True)

        return worst_case_probs.numpy()


    def predict(self, X):
        """
        Predict class labels for the given input samples.

        This method predicts the most likely class for each input sample
        using the trained model. The prediction behavior depends on the
        deterministic parameter set during initialization:
        
        - If deterministic=True: Returns the class with highest logit value
        - If deterministic=False: Samples from the predicted probability distribution

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict class labels.
            Should be preprocessed in the same way as the training data.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels for each input sample as integer indices
            in the range [0, num_classes-1].

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet (i.e., fit() has not
            been called).
            
        ValueError
            If the input data format is invalid or incompatible with the
            trained model.

        Examples
        --------
        See class-level examples.

        See Also
        --------
        predict_proba : Get class probabilities instead of hard predictions
        """
        if self.deterministic:
            # Validate input data format and check if model has been fitted
            X = check_array(X, accept_sparse=True)
            check_is_fitted(self, "is_fitted_")

            # Convert to tensor if needed and move to device
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Compute predictions without gradient computation
            with torch.no_grad():
                # Get the logits and predict the label directly
                logits = self.model(X)
                _, predictions = logits.max(1)
                
            # Convert back to numpy and return
            return predictions.cpu().numpy()
        else:
            # Get class probabilities (this handles input validation)
            probabilities = self.predict_proba(X)

            # Sample from the probability distribution
            predictions = []
            for prob in probabilities:
                predictions.append(np.random.choice(len(prob), p=prob))
            return np.array(predictions)

    def compute_phi(self, X):
        """
        Compute phi features for the minimax risk framework.

        This method computes the phi feature representation used in the
        minimax risk classification framework by performing a forward pass
        through the neural network model. The output logits serve as the
        phi features in the MRC optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to compute phi features.
            Should be preprocessed in the same way as the training data.

        Returns
        -------
        phi_features : ndarray of shape (n_samples, n_classes)
            Model logits for each input sample, used as phi features
            in the minimax risk classification framework.

        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet.

        ValueError
            If the input data format is invalid or incompatible with the
            trained model.

        Notes
        -----
        This method is primarily used internally by the MRC framework
        and may not be needed for typical usage scenarios.
        """
        # Validate input and check if fitted
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Convert to tensor if needed and move to device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        
        # Compute phi features (model logits in this case)
        with torch.no_grad():
            phi_features = self.model(X)
            
        # Convert back to numpy and return
        return phi_features.cpu().numpy()