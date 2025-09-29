import unittest
import numpy as np
import pandas as pd
from ml_grid.pipeline.data_correlation_matrix import handle_correlation_matrix


class TestHandleCorrelationMatrix(unittest.TestCase):

    def setUp(self):
        # Create a DataFrame with three groups
        np.random.seed(0)
        data = {
            "Group": np.random.choice(
                ["Low Correlation", "Medium Correlation", "High Correlation"], size=100
            ),
            "A": np.random.normal(loc=0, scale=1, size=100),
            "B": np.random.normal(loc=0, scale=1, size=100),
            "C": np.random.normal(loc=0, scale=1, size=100),
        }
        self.df = pd.DataFrame(data)

        # Add correlation patterns
        self.df["B"] = self.df["A"] + np.random.normal(
            loc=0, scale=0.1, size=100
        )  # Making B correlated with A
        self.df["C"] = np.random.normal(
            loc=0, scale=1, size=100
        )  # Keeping C uncorrelated with A and B

        self.local_param_dict = {"corr": 0.5}
        self.drop_list = []

    def test_no_numeric_columns(self):
        result = handle_correlation_matrix(
            self.local_param_dict,
            self.drop_list,
            self.df.drop(columns=["A", "B", "C"]),
            chunk_size=1,
        )
        self.assertEqual(result, [])

    def test_empty_dataframe(self):
        """Test that an empty DataFrame is handled gracefully."""
        result = handle_correlation_matrix(
            self.local_param_dict, self.drop_list, pd.DataFrame(), chunk_size=1
        )
        self.assertEqual(result, [])

    def test_negative_correlation(self):
        """Test that high negative correlation is also detected."""
        self.df["C"] = -self.df["A"] + np.random.normal(loc=0, scale=0.1, size=100)
        result = handle_correlation_matrix(
            self.local_param_dict, self.drop_list, self.df, chunk_size=3
        )
        # Pairs found: (A,B), (A,C), (B,C)
        # Logic: drop B (from A,B), drop C (from A,C). B is already processed when we check (B,C).
        # The function now correctly identifies that 'B' and 'C' should be dropped.
        self.assertEqual(len(result), 2, f"Expected 2 items, but got {len(result)}: {result}")
        self.assertCountEqual(result, ["B", "C"])

    def test_all_columns_in_single_chunk(self):
        result = handle_correlation_matrix(
            self.local_param_dict, self.drop_list, self.df, chunk_size=3
        )

        # 'B' is highly correlated with 'A', so 'B' should be added to the drop list.
        self.assertEqual(len(result), 1, f"Expected 1 item, but got {len(result)}: {result}")
        self.assertCountEqual(result, ["B"])


if __name__ == "__main__":
    unittest.main()
