import unittest
from unittest import mock

import torch

from ml_grid.model_classes_ga.pytorchANNBinaryClassifier_model import (
    predict_with_fallback,
)


class TestPredictWithFallback(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 2)
        self.input_data = torch.rand(10, 10)
        self.target_data = torch.rand(10)

    def test_predict_with_fallback_handles_exception(self):
        # Mock the model to raise an exception
        mock_model = mock.Mock()
        mock_model.side_effect = Exception("test exception")

        # Call the function with the mock model and input data
        y_pred = predict_with_fallback(
            model=mock_model, X_batch=self.input_data, y_batch=self.target_data
        )

        # Verify that the fallback (zeros tensor) is returned
        expected_fallback = torch.zeros(
            self.target_data.unsqueeze(1).shape, dtype=self.target_data.dtype, device=self.target_data.device
        )
        self.assertTrue(torch.equal(y_pred, expected_fallback))

    def test_predict_with_fallback_returns_expected_output(self):
        # Mock the model to return a prediction
        mock_model = mock.Mock()
        mock_model.return_value = torch.rand(10, 1)

        # Call the function with the mock model and input data
        y_pred = predict_with_fallback(
            model=mock_model, X_batch=self.input_data, y_batch=self.target_data
        )

        # Verify that the output is correct
        self.assertTrue(torch.allclose(y_pred, mock_model.return_value))


if __name__ == "__main__":
    unittest.main()
