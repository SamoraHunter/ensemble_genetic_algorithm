"""Tests for read_in module."""

import pandas as pd
from pytest import raises

from ml_grid.pipeline.read_in import read, read_sample


class TestReadClass:
    """Tests for the read class."""

    def test_read_csv_success(self, tmp_path):
        """Test successful CSV reading with pandas."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)

        reader = read(str(csv_file))
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 2

    def test_read_with_use_polars_success(self, tmp_path):
        """Test successful CSV reading with Polars fallback."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)

        reader = read(str(csv_file), use_polars=True)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 2

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a nonexistent file returns empty DataFrame."""
        nonexistent = tmp_path / "nonexistent.csv"

        reader = read(str(nonexistent))
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 0

    def test_read_nonexistent_file_with_polars(self, tmp_path):
        """Test reading a nonexistent file with Polars fallback."""
        nonexistent = tmp_path / "nonexistent.csv"

        reader = read(str(nonexistent), use_polars=True)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 0


class TestReadSampleClass:
    """Tests for the read_sample class."""

    def test_read_sample_success(self, tmp_path):
        """Test successful sampling of a CSV file."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame(
            {
                "outcome_var_1": list(range(10)),
                "age": list(range(10, 20)),
                "male": [0, 1] * 5,
                "other_col": list(range(20, 30)),
            }
        )
        data.to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=5, column_sample_n=3)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) <= 5
        assert "outcome_var_1" in reader.raw_input_data.columns

    def test_read_sample_with_all_rows(self, tmp_path):
        """Test reading all rows when test_sample_n is greater than data size."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=100, column_sample_n=0)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_sample_with_all_columns(self, tmp_path):
        """Test reading all columns when column_sample_n is greater than data size."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=0, column_sample_n=100)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data.columns) == 3

    def test_read_sample_missing_columns(self, tmp_path):
        """Test sampling without necessary columns (no crash)."""
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"x": [1], "y": [2]}).to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=1, column_sample_n=2)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_sample_invalid_file(self):
        """Test reading a nonexistent file returns empty DataFrame."""
        reader = read_sample(
            "/nonexistent/file.csv", test_sample_n=5, column_sample_n=3
        )
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 0

    def test_read_sample_exception(self):
        """Test exception handling in read_sample."""
        reader = read_sample(
            "/nonexistent/file.csv", test_sample_n=5, column_sample_n=3
        )
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_with_use_polars_false_and_invalid_file(self, tmp_path):
        """Test reading a nonexistent file with use_polars=False."""
        csv_file = tmp_path / "test.csv"

        reader = read(str(csv_file), use_polars=False)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 0

    def test_read_sample_row_sampling_with_skiprows(self, tmp_path):
        """Test row sampling works with skiprows when test_sample_n > 0."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame({"a": list(range(100)), "b": list(range(100, 200))})
        data.to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=50, column_sample_n=0)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) <= 50

    def test_read_sample_column_sampling_with_necessary_cols(self, tmp_path):
        """Test column sampling includes necessary columns and samples additional ones."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame(
            {
                "outcome_var_1": list(range(10)),
                "age": list(range(10, 20)),
                "male": [0, 1] * 5,
                "col_a": list(range(20, 30)),
                "col_b": list(range(30, 40)),
            }
        )
        data.to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=0, column_sample_n=4)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        # Should include outcome_var_1 and at least one additional column
        assert "outcome_var_1" in reader.raw_input_data.columns

    def test_read_sample_invalid_outcome_variable(self, tmp_path):
        """Test ValueError is raised when outcome_var_1 has fewer than 2 unique classes."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame(
            {
                "outcome_var_1": [1] * 10,  # Only one class
                "age": list(range(10)),
                "male": [0, 1] * 5,
            }
        )
        data.to_csv(csv_file, index=False)

        with raises(ValueError) as exc_info:
            read_sample(str(csv_file), test_sample_n=5, column_sample_n=3)
        assert "Outcome variable does not have both classes post sampling" in str(
            exc_info.value
        )

    def test_read_sample_column_sampling_with_necessary_cols_and_remaining(
        self, tmp_path
    ):
        """Test column sampling when there are both necessary and remaining columns."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame(
            {
                "outcome_var_1": list(range(10)),
                "age": list(range(10, 20)),
                "male": [0, 1] * 5,
                "col_a": list(range(20, 30)),
                "col_b": list(range(30, 40)),
                "col_c": list(range(40, 50)),
            }
        )
        data.to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=0, column_sample_n=5)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert "outcome_var_1" in reader.raw_input_data.columns

    def test_read_sample_row_sampling_error(self, tmp_path):
        """Test exception handling during row sampling (open fails)."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        data.to_csv(csv_file, index=False)

        # This tests the outer except block for row sampling
        reader = read_sample(str(csv_file), test_sample_n=1000000, column_sample_n=0)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_sample_column_sampling_with_single_row_file(self, tmp_path):
        """Test column sampling when file has only header row (single row)."""
        csv_file = tmp_path / "test.csv"
        data = pd.DataFrame(
            {
                "outcome_var_1": [0] * 5 + [1] * 5,
                "age": list(range(10)),
                "male": [0, 1] * 5,
                "extra_col": list(range(10, 20)),
            }
        )
        data.to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=5, column_sample_n=4)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_sample_empty_column_selection(self, tmp_path):
        """Test exception handling in column sampling when file is nearly empty."""
        csv_file = tmp_path / "test.csv"
        # Write only header row, no data
        pd.DataFrame({"a": [], "b": []}).to_csv(csv_file, index=False)

        reader = read_sample(str(csv_file), test_sample_n=0, column_sample_n=1)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_read_sample_binary_file_column_sampling_error(self, tmp_path):
        """Test exception handling in column sampling with binary file."""
        csv_file = tmp_path / "test.csv"
        # Write binary garbage that will cause UnicodeDecodeError
        csv_file.write_bytes(b"\x00\x01\x02\xff")

        reader = read_sample(str(csv_file), test_sample_n=0, column_sample_n=1)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
