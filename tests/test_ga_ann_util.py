import unittest

import torch
import torch.nn as nn

from ml_grid.ga_functions.ga_ann_util import BinaryClassification


class TestBinaryClassification(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for tests."""
        self.column_length = 10
        self.hidden_size = 32
        self.deep_layers = 2
        self.dropout_val = 0.5
        # This is the actual batch size for training data
        self.batch_size = 64

        self.model = BinaryClassification(
            column_length=self.column_length,
            deep_layers_1=self.deep_layers,
            hidden_layer_size=self.hidden_size,
            dropout_val=self.dropout_val,
        )

    def test_initialization(self):
        """Test if the model layers are initialized correctly."""
        self.assertIsInstance(self.model.layer_1, nn.Linear)
        self.assertEqual(self.model.layer_1.in_features, self.column_length)
        self.assertEqual(self.model.layer_1.out_features, self.hidden_size)

        self.assertIsInstance(self.model.deep_layers, nn.Sequential)
        self.assertEqual(len(self.model.deep_layers), self.deep_layers)
        for layer in self.model.deep_layers:
            self.assertIsInstance(layer, nn.Linear)
            self.assertEqual(layer.in_features, self.hidden_size)
            self.assertEqual(layer.out_features, self.hidden_size)

        self.assertIsInstance(self.model.layer_out, nn.Linear)
        self.assertEqual(self.model.layer_out.in_features, self.hidden_size)
        self.assertEqual(self.model.layer_out.out_features, 1)

        self.assertEqual(self.model.dropout.p, self.dropout_val)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        input_tensor = torch.randn(self.batch_size, self.column_length)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_forward_pass_with_different_batch_size(self):
        """Test the forward pass with a different batch size."""
        custom_batch_size = 16
        input_tensor = torch.randn(custom_batch_size, self.column_length)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (custom_batch_size, 1))

    def test_training_step(self):
        """Test a single training step (forward, backward, optimizer step)."""
        input_tensor = torch.randn(self.batch_size, self.column_length)
        target_tensor = torch.randint(0, 2, (self.batch_size, 1)).float()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        try:
            optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"Training step failed with exception: {e}")

    def test_dropout_and_eval_mode(self):
        """Test the behavior of dropout in train vs. eval mode."""
        # Use a batch size > 1 for train
        # Use a batch size > 1 for train() mode because of BatchNorm1d
        input_tensor = torch.randn(4, self.column_length)

        self.model.train()
        output1_train = self.model(input_tensor)
        output2_train = self.model(input_tensor)
        self.assertFalse(
            torch.equal(output1_train, output2_train),
            "Outputs should differ in train mode due to dropout.",
        )

        self.model.eval()
        output1_eval = self.model(input_tensor)
        output2_eval = self.model(input_tensor)
        self.assertTrue(
            torch.equal(output1_eval, output2_eval),
            "Outputs should be identical in eval mode.",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward_pass_on_gpu(self):
        """Test the forward pass on a CUDA device, if available."""
        device = torch.device("cuda")
        model_gpu = self.model.to(device)
        input_tensor = torch.randn(self.batch_size, self.column_length).to(device)

        try:
            output = model_gpu(input_tensor)
            self.assertEqual(output.shape, (self.batch_size, 1))
            self.assertEqual(output.device.type, "cuda")
        except Exception as e:
            self.fail(f"Forward pass on GPU failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
