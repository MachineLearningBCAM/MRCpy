"""
PyTorch-based alpha-loss for minimax risk classification.

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
import math
import torch
import torch.nn.functional as F

class mgce_loss():
    r"""
    Alpha-loss for Minimax Generalized Cross-Entropy (MGCE) Classification.
    See :class:`~MRCpy.pytorch.mgce.classifier.mgce_clf` for the full
    :math:`\alpha`-loss definition and the minimax framework.

    This class implements the :math:`\alpha`-loss (:math:`\ell_\beta`) used in the
    Minimax Generalized Cross-Entropy (MGCE) framework. The class provides a
    parametric family of loss functions indexed by :math:`\beta \geq 1` 
    such that :math:`\beta = \alpha / (\alpha-1)` generalize the standard cross-entropy 
    (log-loss):

    - :math:`\beta = 1`: the convex 0-1 loss (see [2]_)
    - :math:`\beta \to \infty`: the log-loss (cross-entropy)

    The name "generalized cross-entropy" reflects that the :math:`\alpha`-loss
    family extends the standard cross-entropy by allowing the practitioner
    to tune :math:`\beta` to control the trade-off between robustness
    (low :math:`\beta`) and smoothness (high :math:`\beta`).

    This class computes gradients and loss values that can be directly
    integrated with PyTorch training loops.

    Parameters
    ----------
    num_classes : int
        Number of classes in the classification problem. Must be >= 2.
        
    beta : float, default=1.02
        The :math:`\beta` parameter controlling the loss behavior.
        Values in the range :math:`[1, 11]` sufficiently interpolate
        between all possible loss functions:

        - ``beta = 1.0``: 0-1 loss (non-smooth).
        - ``beta ≈ 11``: approximately the log-loss (cross-entropy).
        - Intermediate values smoothly trade off between robustness
          (low :math:`\beta`) and smoothness (high :math:`\beta`).

    Attributes
    ----------
    num_classes : int
        Number of classes in the classification problem.
        
    beta : float
        The parameter for the loss function.

    Examples
    --------
    Basic usage in a training loop:

    >>> import torch
    >>> from MRCpy.pytorch.mgce.loss import mgce_loss
    >>> 
    >>> # Initialize with beta=1.4
    >>> loss_fn = mgce_loss(num_classes=3, beta=1.4)
    >>> 
    >>> # In training loop
    >>> logits = model(inputs)  # Shape: (batch_size, num_classes)
    >>> labels = targets       # Shape: (batch_size,)
    >>> 
    >>> # Compute gradients for backpropagation
    >>> gradients, loss_value = loss_fn.get_gradient(logits, labels)
    >>> 
    >>> # Backpropagate using computed gradients
    >>> logits.backward(gradients)

    For validation/inference:

    >>> # Compute loss and probabilities
    >>> loss_val, probabilities = loss_fn.get_loss_value(logits, labels, reg_val=0.0)
    >>> predicted_classes = torch.argmax(probabilities, dim=1)

    References
    ----------
    .. [1] Bondugula, K., Mazuelas, S., Pérez, A., & Liu, A. (2026).
           Minimax Generalized Cross-Entropy. In Proceedings of the
           International Conference on Artificial Intelligence and
           Statistics (AISTATS).
    .. [2] Mazuelas, S., Shen, Y., & Pérez, A. (2022). Generalized
           Maximum Entropy for Supervised Classification. IEEE Transactions
           on Information Theory, 68(4), 2530-2550.
    
    """
    def __init__(self, num_classes, beta=1.02):
        self.num_classes = num_classes
        self.beta = beta

    def get_gradient(self, logits, labels):
        """
        Compute gradients for the :math:`\alpha`-loss function.

        This method computes the gradients with respect to the logits for
        backpropagation. The gradients are computed using either the 0-1 loss
        formulation (when beta=1) or the smooth :math:`\alpha`-loss approximation.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch_size, num_classes)
            Raw model outputs (logits) before applying softmax or other
            activation functions.
            
        labels : torch.Tensor of shape (batch_size,)
            True class labels as integer indices in range [0, num_classes-1].

        Returns
        -------
        gradients : torch.Tensor of shape (batch_size, num_classes)
            Gradients with respect to the input logits. Can be used directly
            with logits.backward(gradients) for backpropagation.
            
        loss_value : torch.Tensor (scalar)
            The computed loss value for the current batch.

        Examples
        --------
        >>> loss_fn = mgce_loss(num_classes=3, beta=1.4)
        >>> logits = torch.randn(32, 3, requires_grad=True)
        >>> labels = torch.randint(0, 3, (32,))
        >>> 
        >>> gradients, loss = loss_fn.get_gradient(logits, labels)
        >>> logits.backward(gradients)  # Backpropagate
        """
        B = logits.shape[0]
        total_loss = 0

        # Add the tau and lambda part of the objective value in MRCs
        total_loss += - logits[torch.arange(B), labels].mean()
        if self.beta != 1:

            # Step 1: compute nu for the whole batch (using batched bisection solver)
            nu_batch, _ = self._bisection_solver_batched(logits)

            # Step 2: compute psi_beta with clamp
            psi_beta = F.relu((logits + nu_batch.unsqueeze(1)) / self.beta + 1.0)  # [B, C]
            psi_beta_pow = psi_beta.pow(self.beta - 1)
            
            # Step 3: sum across classes for normalization
            sum_psi_beta = psi_beta_pow.sum(dim=1, keepdim=True) 

            # Step 4: compute gradient
            phi_mu_grad = psi_beta_pow / sum_psi_beta

            # Step 5: subtract 1 for the correct class
            phi_mu_grad[torch.arange(B), labels] -= 1.0

            # Step 6: normalize by batch size
            phi_mu_grad = phi_mu_grad / B

            # Also compute the loss
            total_loss += - nu_batch.mean()

        else:

            # Step 1: compute psi values and used indices
            nu_batch, used_mask, sorted_indices = self._psi_batched_with_indices(logits)
            # used_mask: [B, C] in sorted order

            device = logits.device
            # Convert used_mask from sorted order to original indices
            # Scatter mask back to original logits order
            used_mask_original = torch.zeros_like(used_mask, dtype=torch.bool)
            arange_B = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.num_classes)
            used_mask_original[arange_B, sorted_indices] = used_mask


            # Step 2: initialize gradient
            phi_mu_grad = torch.zeros_like(logits, dtype=logits.dtype)
            
            # Step 3: subtract 1 for the correct class
            phi_mu_grad[torch.arange(B), labels] = -1.0
            
            # Step 4: add 1 / |C| for all classes in contributing set
            # Compute |C| per sample
            C_counts = used_mask_original.sum(dim=1).clamp(min=1)  # avoid divide by zero
            # Add 1 / |C| for contributing indices
            phi_mu_grad += used_mask_original.float() / C_counts.unsqueeze(1)
            
            # Step 5: normalize by batch size
            phi_mu_grad = phi_mu_grad / B

            # Also compute the loss
            total_loss += nu_batch.mean()

        return phi_mu_grad, total_loss

    def get_loss_value(self, logits, labels, reg_val):
        """
        Compute loss value and probability predictions.

        This method computes the :math:`\alpha`-loss value and the corresponding
        probability predictions. It is typically used during validation
        or when you need both the loss and the predicted probabilities.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch_size, num_classes)
            Raw model outputs (logits) before applying softmax.
            
        labels : torch.Tensor of shape (batch_size,)
            True class labels as integer indices in range [0, num_classes-1].
            
        reg_val : float or torch.Tensor (scalar)
            Regularization value to add to the loss. Typically the L1 or L2
            regularization term from the model parameters.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
            The computed loss value including regularization.
            
        probabilities : torch.Tensor of shape (batch_size, num_classes)
            Predicted class probabilities. Each row sums to 1.0 and represents
            the probability distribution over classes for the corresponding sample.

        Examples
        --------
        >>> loss_fn = mgce_loss(num_classes=3, beta=1.4)
        >>> logits = torch.randn(32, 3)
        >>> labels = torch.randint(0, 3, (32,))
        >>> reg_val = 0.01
        >>> 
        >>> loss, probs = loss_fn.get_loss_value(logits, labels, reg_val)
        >>> predicted_classes = torch.argmax(probs, dim=1)
        """
        n_samples = logits.shape[0]
        total_loss = 0

        # Add the tau and lambda part of the objective value in MRCs
        total_loss += - logits[torch.arange(n_samples), labels].mean() + reg_val

        if self.beta == 1:
            # Obtain the nu solution corresponding with 0-1 loss
            nu_batch, used_mask, sorted_indices = self._psi_batched_with_indices(logits)
            total_loss += nu_batch.mean()
            hy_x = torch.clamp(1.0 + logits + nu_batch.unsqueeze(1), min=0.0)
        else:
            # Obtain the nu solutions corresponding with beta loss using bisection method
            nu_batch, _ = self._bisection_solver_batched(logits)
            # Computation corresponding with beta loss
            total_loss += - nu_batch.mean()
            psi_beta = torch.clamp((logits + nu_batch.unsqueeze(1)) / self.beta + 1.0, min=0.0)
            # Raise to the power of beta
            hy_x = psi_beta.pow(self.beta)
            # Replace inf with 0 (for numerical stability)
            hy_x = torch.where(torch.isinf(hy_x), torch.zeros_like(hy_x), hy_x)

        # hy_x: [B, C], num_classes: int
        c = hy_x.sum(dim=1)                      # [B] sum per sample
        zeros = torch.isclose(c, torch.tensor(0., device=hy_x.device))  # [B] boolean mask

        # Handle zero sums
        c = torch.where(zeros, torch.ones_like(c), c)                     # avoid division by zero
        hy_x = torch.where(zeros.unsqueeze(1), torch.ones_like(hy_x) / self.num_classes, hy_x)

        # Normalize
        hy_x = hy_x / c.unsqueeze(1)

        return total_loss, hy_x
    
    def get_probs(self, logits):
        """
        Compute class probabilities using the :math:`\alpha`-loss framework.

        This method computes class probabilities for given logits without
        computing the loss value, making it efficient for inference.

        For beta=1 (0-1 loss case), uses the psi function with batched
        computation. For beta>1 (smooth approximation), uses the bisection
        solver to find optimal nu values.

        Parameters
        ----------
        logits : torch.Tensor of shape (batch_size, num_classes)
            Raw model outputs (logits) before applying softmax or other
            activation functions.

        Returns
        -------
        probabilities : torch.Tensor of shape (batch_size, num_classes)
            Predicted class probabilities for each input sample. Each row
            sums to 1.0. Probabilities are computed using the :math:`\alpha`-loss
            framework rather than standard softmax.

        Examples
        --------
        >>> loss_fn = mgce_loss(num_classes=3, beta=1.4)
        >>> logits = torch.randn(32, 3)
        >>> 
        >>> probabilities = loss_fn.get_probs(logits)
        >>> predicted_classes = torch.argmax(probabilities, dim=1)
        >>> confidence_scores = torch.max(probabilities, dim=1)[0]

        See Also
        --------
        get_loss_value : Compute both loss value and probabilities.
        get_gradient : Compute gradients for backpropagation.
        """
        n_samples = logits.shape[0]

        if self.beta == 1:
            # Obtain the nu solution corresponding with 0-1 loss
            nu_batch, used_mask, sorted_indices = self._psi_batched_with_indices(logits)
            hy_x = torch.clamp(1.0 + logits + nu_batch.unsqueeze(1), min=0.0)
        else:
            # Obtain the nu solutions corresponding with beta loss using bisection method
            nu_batch, _ = self._bisection_solver_batched(logits)
            # Computation corresponding with beta loss
            psi_beta = torch.clamp((logits + nu_batch.unsqueeze(1)) / self.beta + 1.0, min=0.0)
            # Raise to the power of beta
            hy_x = psi_beta.pow(self.beta)
            # Replace inf with 0 (for numerical stability)
            hy_x = torch.where(torch.isinf(hy_x), torch.zeros_like(hy_x), hy_x)

        # hy_x: [B, C], num_classes: int
        c = hy_x.sum(dim=1)                      # [B] sum per sample
        zeros = torch.isclose(c, torch.tensor(0., device=hy_x.device))  # [B] boolean mask

        # Handle zero sums
        c = torch.where(zeros, torch.ones_like(c), c)                     # avoid division by zero
        hy_x = torch.where(zeros.unsqueeze(1), torch.ones_like(hy_x) / self.num_classes, hy_x)

        # Normalize
        hy_x = hy_x / c.unsqueeze(1)

        return hy_x

    def _psi_batched_with_indices(self, phi_mu):
        """
        Batched psi function computation with index tracking.

        This method computes the psi function values for the 0-1 loss case
        (beta=1) in a batched manner. It also tracks which indices contribute
        to the computation, which is needed for gradient calculation.

        Parameters
        ----------
        phi_mu : torch.Tensor of shape (batch_size, num_classes)
            Input tensor where each row corresponds to a sample's phi_mu values.

        Returns
        -------
        psi_values : torch.Tensor of shape (batch_size,)
            Computed psi values for each sample in the batch.
            
        used_mask : torch.Tensor of shape (batch_size, num_classes), dtype=bool
            Boolean mask indicating which indices were used in the computation
            for each sample. True indicates the index contributed to the result.
            
        sorted_indices : torch.Tensor of shape (batch_size, num_classes)
            Indices that sort each row of phi_mu in descending order.

        Notes
        -----
        This method is used internally for the 0-1 loss case and implements
        the batched version of the psi function computation for efficiency.
        """
        B, n_classes = phi_mu.shape
        
        # Sort descending
        sorted_vals, sorted_indices = torch.sort(phi_mu, descending=True, dim=1)  # [B, C]
        
        # Initialize
        value = sorted_vals[:, 0] - 1.0                   # [B]
        used_mask = torch.zeros_like(sorted_vals, dtype=torch.bool)  # [B, C]
        used_mask[:, 0] = True
        
        # Iterate over remaining classes
        for k in range(1, n_classes):
            new_value = (k * value + sorted_vals[:, k]) / (k + 1.0)  # [B]
            mask = new_value >= value                                 # [B] which samples should update
            value = torch.where(mask, new_value, value)
            # Update used_mask: set True only where new_value >= value
            used_mask[:, k] = mask
        
        psi_values = value + 1.0
        return psi_values, used_mask, sorted_indices


    def _bisection_solver_batched(self, phi_mu, tolerance=1e-4, max_iters=1000):
        """
        Batched bisection method solver for alpha-loss optimization.

        This method solves the optimization problem for the alpha-loss case
        (when beta != 1) using a batched bisection method. It finds the
        optimal nu values for each sample in the batch simultaneously.

        Parameters
        ----------
        phi_mu : torch.Tensor of shape (batch_size, num_classes)
            Input tensor where each row corresponds to a sample's phi_mu values.
            
        tolerance : float, default=1e-4
            Convergence tolerance for the bisection method. The algorithm
            stops when the interval width is smaller than this value.
            
        max_iters : int, default=1000
            Maximum number of iterations for the bisection method to prevent
            infinite loops.

        Returns
        -------
        nu_values : torch.Tensor of shape (batch_size,)
            Optimal nu values for each sample in the batch.
            
        iterations : int
            Number of iterations performed by the bisection method.

        Raises
        ------
        ValueError
            If the bisection method fails to find a valid interval for
            some batch elements.

        Notes
        -----
        This method is used internally for the alpha-loss case (beta > 1)
        and implements a batched bisection algorithm for computational efficiency.
        The method solves the equation f(nu) = 0 where f is defined based on
        the alpha-loss formulation.
        """

        def f_batched(nu, phi_mu, beta):
            # nu can be scalar or shape [batch, 1]
            psi_beta = torch.clamp((phi_mu + nu.unsqueeze(1)) / beta + 1.0, min=0.0)
            return torch.sum(torch.pow(psi_beta, beta), dim=-1) - 1.0

        device = phi_mu.device
        beta = torch.tensor(self.beta, dtype=phi_mu.dtype, device=device)
        n_classes = torch.tensor(self.num_classes, dtype=phi_mu.dtype, device=device)

        B = phi_mu.shape[0]

        # Initial bounds
        base_val = beta * ((1.0 / n_classes) ** (1.0 / beta) - 1.0)
        nu_1 = base_val - phi_mu.max(dim=1).values   # [B]
        nu_2 = base_val - phi_mu.min(dim=1).values   # [B]

        # Function values at endpoints
        f1 = f_batched(nu_1, phi_mu, beta)
        f2 = f_batched(nu_2, phi_mu, beta)

        # Mask of valid intervals
        if (phi_mu == 0).all().item():
            return nu_1, 0
        elif torch.any(f1 * f2 > 0):
            raise ValueError("Invalid interval for some batch elements")

        # Track which elements are already solved
        done = torch.isclose(f1, torch.tensor(0., device=device)) | \
               torch.isclose(f2, torch.tensor(0., device=device))

        # For those, lock nu to the corresponding endpoint
        nu = torch.where(torch.isclose(f1, torch.tensor(0., device=device)), nu_1,
              torch.where(torch.isclose(f2, torch.tensor(0., device=device)), nu_2, 0.5 * (nu_1 + nu_2)))

        iters = 0
        while torch.any(~done & ((nu_2 - nu_1).abs() > tolerance)) and iters < max_iters:
            nu_mid = 0.5 * (nu_1 + nu_2)
            f_mid = f_batched(nu_mid, phi_mu, beta)
            f_left = f_batched(nu_1, phi_mu, beta)

            # Root check: if f_mid is close to 0, mark as done
            new_done = torch.isclose(f_mid, torch.tensor(0., device=device))
            nu = torch.where(new_done & ~done, nu_mid, nu)  # set solution for those
            done = done | new_done

            # For not-yet-done, update interval
            mask = (f_mid * f_left < 0) & ~done
            nu_2 = torch.where(mask, nu_mid, nu_2)
            nu_1 = torch.where(mask, nu_1, nu_mid)

            iters += 1

        # Final midpoint for unfinished ones
        nu = torch.where(done, nu, 0.5 * (nu_1 + nu_2))
        return nu, iters
