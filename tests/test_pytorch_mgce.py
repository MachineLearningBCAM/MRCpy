""" Unit tests for the PyTorch MGCE classifier """

import unittest
import os
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the PyTorch MGCE classifier and loss
from MRCpy.pytorch.mgce.classifier import mgce_clf
from MRCpy.pytorch.mgce.loss import mgce_loss


class TestPyTorchMGCE(unittest.TestCase):

    def setUp(self):
        """Set up test data and common parameters."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate synthetic classification data
        self.n_samples = 100
        self.n_features = 10
        self.n_classes = 3
        
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Convert to tensors
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        train_dataset.classes = list(range(self.n_classes))
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        test_dataset.classes = list(range(self.n_classes))
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create a temporary directory for model saving tests
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_model(self):
        """Create a simple neural network model."""
        return nn.Sequential(
            nn.Linear(self.n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_classes)
        )

    def create_classifier(self, beta=1.4, deterministic=True, device='cpu'):
        """Create an MGCE classifier with default parameters."""
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        return mgce_clf(
            loss_parameter=beta,
            lambda_=1e-5,
            deterministic=deterministic,
            model=model,
            optimizer=optimizer,
            device=device
        )

    def test_classifier_initialization(self):
        """Test classifier initialization with various parameters."""
        # Test default initialization
        clf = self.create_classifier()
        self.assertEqual(clf.loss_parameter, 1.4)
        self.assertEqual(clf.lambda_, 1e-5)
        self.assertTrue(clf.deterministic)
        self.assertEqual(clf.device, 'cpu')
        self.assertIsNotNone(clf.model)
        self.assertIsNotNone(clf.optimizer)

    def test_classifier_training_basic(self):
        """Test basic training functionality."""
        clf = self.create_classifier()
        
        # Train the classifier
        results = clf.fit(self.train_loader, n_epochs=5, verbose=False)
        
        # Check that classifier is fitted
        self.assertTrue(hasattr(clf, 'is_fitted_'))
        self.assertTrue(clf.is_fitted_)
        
        # Check results structure
        self.assertIn('train_loss', results)
        self.assertIn('train_acc', results)
        self.assertEqual(len(results['train_loss']), 5)
        self.assertEqual(len(results['train_acc']), 5)
        
        # Check that training progressed (loss should generally decrease)
        self.assertIsInstance(results['train_loss'][0], (int, float, torch.Tensor))
        self.assertIsInstance(results['train_acc'][0], (int, float, torch.Tensor))

    def test_classifier_training_with_validation(self):
        """Test training with validation."""
        clf = self.create_classifier()
        
        # Train with validation
        results = clf.fit(
            self.train_loader,
            n_epochs=3,
            verbose=False,
            validate=True,
            val_dataloader=self.test_loader
        )
        
        # Check validation results are included
        self.assertIn('val_loss', results)
        self.assertIn('val_acc', results)
        self.assertIn('val_ece', results)
        self.assertEqual(len(results['val_loss']), 3)
        self.assertEqual(len(results['val_acc']), 3)

    def test_prediction_methods(self):
        """Test prediction methods."""
        clf = self.create_classifier()
        clf.fit(self.train_loader, n_epochs=3, verbose=False)
        
        # Test predict_proba
        probabilities = clf.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape, (len(self.X_test), self.n_classes))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6))
        self.assertTrue(np.all(probabilities >= 0))
        
        # Test predict
        predictions = clf.predict(self.X_test)
        self.assertEqual(predictions.shape, (len(self.X_test),))
        self.assertTrue(np.all((predictions >= 0) & (predictions < self.n_classes)))
        
        # Test compute_phi
        phi_features = clf.compute_phi(self.X_test)
        self.assertEqual(phi_features.shape, (len(self.X_test), self.n_classes))

    def test_deterministic_vs_stochastic_predictions(self):
        """Test deterministic vs stochastic prediction modes."""
        # Test deterministic mode
        clf_det = self.create_classifier(deterministic=True)
        clf_det.fit(self.train_loader, n_epochs=3, verbose=False)
        
        pred1 = clf_det.predict(self.X_test[:10])
        pred2 = clf_det.predict(self.X_test[:10])
        np.testing.assert_array_equal(pred1, pred2)  # Should be identical
        
        # Test stochastic mode
        clf_stoch = self.create_classifier(deterministic=False)
        clf_stoch.fit(self.train_loader, n_epochs=3, verbose=False)
        
        # Note: Stochastic predictions may vary, but we can't guarantee they will
        # differ on small test data, so we just test that the method works
        pred_stoch = clf_stoch.predict(self.X_test[:10])
        self.assertEqual(len(pred_stoch), 10)

    def test_different_beta_values(self):
        """Test classifier with different beta values."""
        beta_values = [1.0, 1.2, 1.4, 2.0]
        
        for beta in beta_values:
            with self.subTest(beta=beta):
                clf = self.create_classifier(beta=beta)
                clf.fit(self.train_loader, n_epochs=2, verbose=False)
                
                # Test that predictions work
                probabilities = clf.predict_proba(self.X_test[:5])
                self.assertEqual(probabilities.shape, (5, self.n_classes))
                self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6))

    def test_model_weight_saving(self):
        """Test model weight saving functionality."""
        clf = self.create_classifier()
        
        # Test saving best weights
        clf.fit(
            self.train_loader,
            n_epochs=3,
            verbose=False,
            validate=True,
            val_dataloader=self.test_loader,
            save_model_weights='best',
            path=self.temp_dir
        )
        
        # Check that model file was created
        model_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.pt')]
        self.assertTrue(len(model_files) > 0)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        clf = self.create_classifier()
        
        # Test fitting without model
        clf_no_model = mgce_clf(model=None, optimizer=None)
        with self.assertRaises(RuntimeError):
            clf_no_model.fit(self.train_loader)
        
        # Test validation without val_dataloader
        with self.assertRaises(ValueError):
            clf.fit(self.train_loader, validate=True, val_dataloader=None)
        
        # Test invalid save_model_weights option
        with self.assertRaises(ValueError):
            clf.fit(self.train_loader, save_model_weights='invalid')
        
        # Test prediction before fitting
        clf_unfitted = self.create_classifier()
        with self.assertRaises(Exception):  # Should raise NotFittedError
            clf_unfitted.predict_proba(self.X_test)

    def test_mgce_loss_function(self):
        """Test the MGCE loss function directly."""
        loss_fn = mgce_loss(num_classes=self.n_classes, beta=1.4)
        
        # Create test logits and labels
        logits = torch.randn(16, self.n_classes)
        labels = torch.randint(0, self.n_classes, (16,))
        
        # Test get_gradient
        gradients, loss_value = loss_fn.get_gradient(logits, labels)
        self.assertEqual(gradients.shape, logits.shape)
        self.assertIsInstance(loss_value.item(), float)
        
        # Test get_loss_value
        reg_val = 0.01
        total_loss, probabilities = loss_fn.get_loss_value(logits, labels, reg_val)
        self.assertEqual(probabilities.shape, logits.shape)
        self.assertTrue(torch.allclose(torch.sum(probabilities, dim=1), torch.ones(16), atol=1e-6))
        
        # Test get_probs (new function)
        probs_only = loss_fn.get_probs(logits)
        self.assertEqual(probs_only.shape, logits.shape)
        self.assertTrue(torch.allclose(torch.sum(probs_only, dim=1), torch.ones(16), atol=1e-6))
        
        # Test consistency between get_loss_value and get_probs
        self.assertTrue(torch.allclose(probabilities, probs_only, atol=1e-6))

    def test_mgce_loss_different_beta(self):
        """Test MGCE loss with different beta values."""
        logits = torch.randn(8, self.n_classes)
        
        for beta in [1.0, 1.2, 2.0]:
            with self.subTest(beta=beta):
                loss_fn = mgce_loss(num_classes=self.n_classes, beta=beta)
                
                # Test get_probs works for different beta values
                probabilities = loss_fn.get_probs(logits)
                self.assertEqual(probabilities.shape, logits.shape)
                self.assertTrue(torch.allclose(torch.sum(probabilities, dim=1), torch.ones(8), atol=1e-6))
                self.assertTrue(torch.all(probabilities >= 0))

    def test_mgce_vs_softmax_difference(self):
        """Test that MGCE probabilities differ from softmax."""
        clf = self.create_classifier()
        clf.fit(self.train_loader, n_epochs=3, verbose=False)
        
        # Get MGCE probabilities
        mgce_probs = clf.predict_proba(self.X_test[:10])
        
        # Get softmax probabilities for comparison
        clf.model.eval()
        with torch.no_grad():
            logits = clf.model(self.X_test_tensor[:10])
            softmax_probs = torch.softmax(logits, dim=1).numpy()
        
        # They should be different (at least for some samples)
        max_diff = np.max(np.abs(mgce_probs - softmax_probs))
        self.assertGreater(max_diff, 1e-6)  # Should have meaningful differences

    def test_consistency_across_calls(self):
        """Test that repeated calls give consistent results."""
        clf = self.create_classifier()
        clf.fit(self.train_loader, n_epochs=3, verbose=False)
        
        # Test predict_proba consistency
        probs1 = clf.predict_proba(self.X_test[:5])
        probs2 = clf.predict_proba(self.X_test[:5])
        np.testing.assert_array_almost_equal(probs1, probs2)
        
        # Test deterministic predict consistency
        pred1 = clf.predict(self.X_test[:5])
        pred2 = clf.predict(self.X_test[:5])
        np.testing.assert_array_equal(pred1, pred2)
