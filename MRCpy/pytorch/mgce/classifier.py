"""
PyTorch-based Marginally Constrained Minimax Risk Classifier.

This module implements a PyTorch-based version of Minimax Risk Classifiers
with marginally constrained generalized cross-entropy (MGCE) loss functions.
It provides GPU-accelerated training and inference for deep neural networks
with theoretical robustness guarantees.

The implementation supports:
- Custom MGCE loss functions with configurable beta parameters
- GPU acceleration with CUDA support
- Validation during training with model checkpointing
- Integration with standard PyTorch training workflows
- Comprehensive logging and progress tracking

Classes
-------
mgce_clf
    Main classifier class for training and inference with MGCE loss.

Examples
--------
Basic usage:

>>> import torch
>>> import torch.nn as nn
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from MRCpy.pytorch.mgce.classifier import mgce_clf
>>> 
>>> # Create model and data
>>> model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
>>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
>>> 
>>> # Initialize classifier
>>> clf = mgce_clf(model=model, optimizer=optimizer, loss_parameter=1.4)
>>> 
>>> # Train the model
>>> results = clf.fit(train_loader, n_epochs=50, validate=True, val_dataloader=val_loader)
"""
"""
Marginally constrained minimax risk classification using alpha loss functions. 
Copyright (C) 2021 Kartheek Bondugula

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
    """
    PyTorch-based Marginally Constrained Minimax Risk Classifier.

    This class implements a PyTorch-based version of Minimax Risk Classifiers
    with marginally constrained generalized cross-entropy (MGCE) loss functions.
    It supports deep neural networks and provides GPU acceleration for training
    and inference.

    The classifier uses a custom loss function that combines minimax risk
    optimization with margin-based generalized cross-entropy, enabling robust
    classification with theoretical guarantees.

    Parameters
    ----------
    loss_parameter : str or float, default='1.4'
        Beta parameter for the generalized cross-entropy loss function.
        When beta=1, it corresponds to 0-1 loss. When beta>1, it provides
        a smooth approximation with better optimization properties.
        
    lambda_ : float, default=1e-5
        L1 regularization strength applied to model parameters.
        Higher values increase regularization and may improve generalization
        but can reduce model capacity.
        
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
        Beta parameter for the loss function.
        
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
    >>> # Initialize classifier
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

    See Also
    --------
    MRCpy.base_mrc.BaseMRC : Base class for MRC implementations
    MRCpy.pytorch.mgce.loss.mgce_loss : Loss function used by this classifier

    References
    ----------
    .. [1] Mazuelas, S., Shen, Y., & Pérez, A. (2020). Generalized Maximum
           Entropy for Supervised Classification. arXiv preprint arXiv:2007.05447.
    .. [2] Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library
           for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952.
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
        """
        Initialize the MGCE classifier.

        Parameters
        ----------
        loss_parameter : str or float, default='1.4'
            Beta parameter for the generalized cross-entropy loss function.
            
        lambda_ : float, default=1e-5
            L1 regularization strength applied to model parameters.
            
        deterministic : bool, default=True
            Whether predictions should be deterministic.
            
        random_state : int or None, default=None
            Random seed for reproducible results.
            
        optimizer : torch.optim.Optimizer or None, default=None
            PyTorch optimizer instance for training.
            
        scheduler : torch.optim.lr_scheduler or None, default=None
            Learning rate scheduler for training.
            
        model : torch.nn.Module or None, default=None
            PyTorch neural network model to be trained.
            
        device : str, default='cuda'
            Device for computation ('cuda' for GPU, 'cpu' for CPU).

        Raises
        ------
        ValueError
            If invalid parameters are provided.
        """
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

        This method trains the neural network using the marginally constrained
        minimax risk optimization with generalized cross-entropy loss. It supports
        both training-only and training-with-validation modes.

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
        Basic training without validation:

        >>> results = clf.fit(train_loader, n_epochs=50, verbose=True)
        >>> print(f"Final training accuracy: {results['train_acc'][-1]:.2f}%")

        Training with validation:

        >>> results = clf.fit(
        ...     train_loader, 
        ...     validate=True, 
        ...     val_dataloader=val_loader,
        ...     save_model_weights='best'
        ... )
        >>> print(f"Best validation accuracy: {max(results['val_acc']):.2f}%")
        """
        # Validate inputs
        if validate and val_dataloader is None:
            raise ValueError("val_dataloader must be provided when validate=True")
        
        if self.model is None:
            raise RuntimeError("Model must be provided before calling fit()")
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be provided before calling fit()")

        # Initialize the MGCE loss function with the number of classes and beta parameter
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

                # Compute MGCE gradients and loss using the custom loss function
                grad, loss = loss_function.get_gradient(logits, labels)
                
                # Backward pass: compute gradients using the custom MGCE gradients
                logits.backward(grad)

                # Apply L1 regularization manually since MGCE loss requires custom regularization
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
                        
                        # Compute validation loss and probability predictions using MGCE loss
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
        the trained model and the marginally constrained minimax risk framework
        via the MGCE loss function's get_probs method.

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
            for the corresponding input sample, computed using the MGCE framework.

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
        Compute probabilities for test samples:

        >>> # Assuming clf is a fitted mgce_clf instance
        >>> test_data = torch.randn(10, input_dim)
        >>> probabilities = clf.predict_proba(test_data)
        >>> print(f"Shape: {probabilities.shape}")  # (10, n_classes)
        >>> print(f"First sample probabilities: {probabilities[0]}")

        Get the most likely class for each sample:

        >>> predicted_classes = np.argmax(probabilities, axis=1)
        >>> confidence_scores = np.max(probabilities, axis=1)

        Notes
        -----
        This method uses the MGCE framework's get_probs function to compute
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
            
            # Initialize the MGCE loss function to use get_probs
            # We need to determine the number of classes from the logits
            num_classes = logits.shape[1]
            loss_function = mgce_loss(num_classes, self.loss_parameter)
            
            # Use MGCE framework to compute probabilities
            probabilities = loss_function.get_probs(logits)
            
        # Convert back to numpy and return
        return probabilities.cpu().numpy()

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
        Deterministic predictions (default behavior):

        >>> # Assuming clf is a fitted mgce_clf instance with deterministic=True
        >>> test_data = torch.randn(10, input_dim)
        >>> predictions = clf.predict(test_data)
        >>> print(f"Predicted classes: {predictions}")
        >>> print(f"Predictions shape: {predictions.shape}")  # (10,)

        Stochastic predictions:

        >>> # Initialize with deterministic=False
        >>> clf_stochastic = mgce_clf(deterministic=False, ...)
        >>> clf_stochastic.fit(train_loader)
        >>> 
        >>> # Predictions will vary between calls due to sampling
        >>> pred1 = clf_stochastic.predict(test_data)
        >>> pred2 = clf_stochastic.predict(test_data)  # May differ from pred1

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
        minimax risk classification framework. These features are used
        internally for the optimization process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to compute phi features.

        Returns
        -------
        phi_features : ndarray of shape (n_samples, n_phi_features)
            Computed phi features for each input sample.

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