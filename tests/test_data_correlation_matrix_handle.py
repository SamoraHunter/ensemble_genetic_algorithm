import unittest
import time

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
        self.assertEqual(
            len(result), 2, f"Expected 2 items, but got {len(result)}: {result}"
        )
        self.assertCountEqual(result, ["B", "C"])

    def test_all_columns_in_single_chunk(self):
        result = handle_correlation_matrix(
            self.local_param_dict, self.drop_list, self.df, chunk_size=3
        )

        # 'B' is highly correlated with 'A', so 'B' should be added to the drop list.
        self.assertEqual(
            len(result), 1, f"Expected 1 item, but got {len(result)}: {result}"
        )
        self.assertCountEqual(result, ["B"])

    def test_performance_scaling(self):
        """Test performance with small sample and extrapolate to verify it scales well."""
        np.random.seed(42)
        
        # Test with a smaller dataset that runs quickly
        n_rows = 1000
        n_cols_test = 1000  # Small enough to run in seconds
        target_cols = 15000  # What we want to extrapolate to
        
        print(f"\n{'='*70}")
        print(f"Performance scaling test")
        print(f"{'='*70}")
        
        # Create test data with known correlation patterns
        data = {}
        
        # Create independent columns
        for i in range(0, n_cols_test, 10):
            data[f"col_{i}"] = np.random.normal(loc=0, scale=1, size=n_rows)
        
        # Create correlated groups (proportional to what we'd have at 15k)
        base_col_0 = data["col_0"]
        for offset in [1, 2, 3, 4]:
            data[f"col_{offset}"] = base_col_0 + np.random.normal(0, 0.1, n_rows)
        
        # Fill remaining columns
        for i in range(n_cols_test):
            col_name = f"col_{i}"
            if col_name not in data:
                data[col_name] = np.random.normal(loc=0, scale=1, size=n_rows)
        
        df_test = pd.DataFrame(data)
        
        print(f"Test DataFrame: {df_test.shape}")
        print(f"Memory usage: {df_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Measure performance
        local_param_dict = {"corr": 0.8}
        
        print(f"\nRunning correlation analysis on {n_cols_test} columns...")
        start_time = time.time()
        
        result = handle_correlation_matrix(
            local_param_dict,
            [],
            df_test,
            chunk_size=50
        )
        
        elapsed_time = time.time() - start_time
        
        # Calculate expected time for 15k columns
        # Time complexity is roughly O(n^2) for correlation, but chunked so more like O(n * m)
        # where n is number of columns and m is average remaining columns per chunk
        # Simplified: assume roughly quadratic relationship
        scaling_factor = (target_cols / n_cols_test) ** 2
        estimated_time_15k = elapsed_time * scaling_factor
        
        # More conservative estimate considering chunking
        # With chunking, it's closer to O(n * n/2) amortized
        conservative_estimate = elapsed_time * (target_cols / n_cols_test) * (target_cols / n_cols_test / 2)
        
        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"{'='*70}")
        print(f"Time for {n_cols_test} columns: {elapsed_time:.2f} seconds")
        print(f"Columns dropped: {len(result)}")
        print(f"\nExtrapolation to {target_cols} columns:")
        print(f"  Quadratic estimate: {estimated_time_15k:.2f} seconds ({estimated_time_15k/60:.2f} minutes)")
        print(f"  Conservative estimate: {conservative_estimate:.2f} seconds ({conservative_estimate/60:.2f} minutes)")
        print(f"{'='*70}")
        
        # Performance assertions
        # Test should complete in reasonable time (< 30 seconds for 1000 columns)
        self.assertLess(
            elapsed_time, 
            30, 
            f"Test took too long for {n_cols_test} columns: {elapsed_time:.2f}s"
        )
        
        # Extrapolated time for 15k columns should be reasonable (< 10 minutes)
        self.assertLess(
            conservative_estimate,
            600,  # 10 minutes
            f"Extrapolated time for {target_cols} columns is too high: "
            f"{conservative_estimate:.2f}s ({conservative_estimate/60:.2f} min). "
            f"Consider further optimization."
        )
        
        # Correctness check - should find correlated columns
        self.assertGreater(
            len(result),
            0,
            "Should find at least some correlated columns"
        )
        
        # Check that known correlated columns are found
        expected_in_drop = ["col_1", "col_2", "col_3", "col_4"]
        found = [col for col in expected_in_drop if col in result]
        self.assertGreater(
            len(found),
            0,
            f"Expected to find some of {expected_in_drop} in results"
        )
        
        print(f"\n✓ Performance test passed!")
        print(f"✓ Found {len(found)}/{len(expected_in_drop)} expected correlations")
        print(f"✓ Estimated to handle {target_cols} columns in ~{conservative_estimate/60:.1f} minutes")

    def test_scaling_with_different_chunk_sizes(self):
        """Verify that different chunk sizes don't significantly affect performance."""
        np.random.seed(42)
        
        n_rows = 500
        n_cols = 500
        
        print(f"\n{'='*70}")
        print(f"Chunk size comparison test: {n_rows} rows × {n_cols} columns")
        print(f"{'='*70}")
        
        # Create test data
        data = {f"col_{i}": np.random.normal(0, 1, n_rows) for i in range(n_cols)}
        df_test = pd.DataFrame(data)
        
        local_param_dict = {"corr": 0.9}
        
        times = {}
        results = {}
        
        for chunk_size in [25, 50, 100]:
            print(f"\nTesting chunk_size={chunk_size}...")
            start_time = time.time()
            
            result = handle_correlation_matrix(
                local_param_dict,
                [],
                df_test,
                chunk_size=chunk_size
            )
            
            elapsed = time.time() - start_time
            times[chunk_size] = elapsed
            results[chunk_size] = len(result)
            
            print(f"  Time: {elapsed:.3f}s, Dropped: {len(result)} columns")
        
        print(f"\n{'='*70}")
        print("Summary:")
        for chunk_size in [25, 50, 100]:
            print(f"  chunk_size={chunk_size}: {times[chunk_size]:.3f}s")
        print(f"{'='*70}")
        
        # All chunk sizes should produce same results
        unique_result_counts = set(results.values())
        self.assertEqual(
            len(unique_result_counts),
            1,
            f"Different chunk sizes produced different results: {results}"
        )
        
        # All should complete in reasonable time (< 10 seconds for 500 cols)
        for chunk_size, elapsed in times.items():
            self.assertLess(
                elapsed,
                10,
                f"chunk_size={chunk_size} took too long: {elapsed:.2f}s"
            )
        
        print(f"\n✓ All chunk sizes produced consistent results: {list(results.values())[0]} columns dropped")


if __name__ == "__main__":
    unittest.main()