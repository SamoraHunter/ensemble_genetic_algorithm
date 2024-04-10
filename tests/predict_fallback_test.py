import torch
import unittest
from unittest.mock import MagicMock
from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import predict_with_fallback

class TestPredictWithFallback(unittest.TestCase):

    def test_model_prediction_success(self):
        # Mock the model to return a specific output
        model = MagicMock()
        X_batch = torch.randn(3, 5)
        y_batch = torch.randint(2, (3,), device=X_batch.device)

        expected_output = torch.randn(3, 1)
        model.return_value = expected_output

        y_pred = predict_with_fallback(model, X_batch, y_batch)
        self.assertEqual(y_pred.shape, expected_output.shape)
        self.assertTrue(torch.allclose(y_pred, expected_output))

    def test_model_prediction_failure(self):
        # Mock the model to raise an exception
        model = MagicMock()
        X_batch = torch.randn(3, 5)
        y_batch = torch.randint(2, (3,), device=X_batch.device)

        model.side_effect = Exception("Model prediction failed")

        y_pred = predict_with_fallback(model, X_batch, y_batch)
        self.assertEqual(y_pred.shape, y_batch.shape)
        self.assertTrue(torch.all((y_pred == 0) | (y_pred == 1)))

if __name__ == '__main__':
    unittest.main()
