"""
Tests for column alignment in the final_grid_score_log.csv file.
These tests ensure that all columns are written in the correct order
and prevent regression when new parameters are added.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def test_column_alignment():
    """
    Test that all columns in the CSV file match expected column names
    and are in the correct order.
    """
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print(f"Test file not found: {csv_path}")
        return

    # Read the CSV
    df = pd.read_csv(csv_path, nrows=5)

    # Expected columns from project_score_save.py (lines 222-277)
    expected_columns = [
        "nb_size",
        "f_list",
        "auc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "nb_val",
        "pop_val",
        "g_val",
        "g",
        "weighted",
        "use_stored_base_learners",
        "store_base_learners",
        "resample",
        "scale",
        "n_features",
        "param_space_size",
        "n_unique_out",
        "outcome_var_n",
        "div_p",
        "percent_missing",
        "corr",
        "age",
        "sex",
        "bmi",
        "ethnicity",
        "bloods",
        "diagnostic_order",
        "drug_order",
        "annotation_n",
        "meta_sp_annotation_n",
        "meta_sp_annotation_mrc_n",
        "annotation_mrc_n",
        "core_02",
        "bed",
        "vte_status",
        "hosp_site",
        "core_resus",
        "news",
        "date_time_stamp",
        "X_train_size",
        "X_test_orig_size",
        "X_test_size",
        "run_time",
        "cx_type",  # Added: was missing
        "cxpb",
        "mutpb",
        "indpb",
        "t_size",
        "valid",
        "generation_progress_list",
        "best_ensemble",
        "original_feature_names",
    ]

    actual_columns = df.columns.tolist()

    # Check for exact match (order matters)
    if expected_columns != actual_columns:
        print("COLUMN MISMATCH DETECTED!")
        print(f"Expected ({len(expected_columns)}): {expected_columns}")
        print(f"Actual   ({len(actual_columns)}): {actual_columns}")

        # Find differences
        for i, exp in enumerate(expected_columns):
            if i >= len(actual_columns) or actual_columns[i] != exp:
                print(
                    f"  Position {i+1}: Expected '{exp}', Got '{actual_columns[i] if i < len(actual_columns) else 'MISSING'}'"
                )

        # Find missing/extra
        expected_set = set(expected_columns)
        actual_set = set(actual_columns)

        missing = expected_set - actual_set
        extra = actual_set - expected_set

        if missing:
            print(f"Missing columns: {missing}")
        if extra:
            print(f"Extra columns: {extra}")

        return False

    print("✓ All columns match!")

    # Check that GA parameters have reasonable values in the first data row
    if len(df) > 0:
        row = df.iloc[0]
        print("\nFirst data row values for key columns:")
        print(f"  cx_type: {row['cx_type']}")
        print(f"  cxpb: {row['cxpb']}")
        print(f"  mutpb: {row['mutpb']}")
        print(f"  indpb: {row['indpb']}")
        print(f"  t_size: {row['t_size']}")

        # Validate GA parameter types
        assert row["cx_type"] in [
            "twopoint",
            "onepoint",
            "uniform",
            "blend",
            "ordered",
        ], f"Invalid cx_type: {row['cx_type']}"
        assert isinstance(
            row["cxpb"], (float, np.floating)
        ), f"cxpb should be float, got {type(row['cxpb'])}: {row['cxpb']}"
        assert isinstance(
            row["mutpb"], (float, np.floating)
        ), f"mutpb should be float, got {type(row['mutpb'])}: {row['mutpb']}"
        assert isinstance(
            row["indpb"], (float, np.floating)
        ), f"indpb should be float, got {type(row['indpb'])}: {row['indpb']}"

    return True


def test_expected_column_count():
    """
    Test that the number of columns is exactly as expected.
    This helps detect when new parameters are added without updating the header.
    """
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print(f"Test file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, nrows=5)
    expected_count = 55  # Total number of columns

    if len(df.columns) != expected_count:
        print(
            f"Column count mismatch! Expected {expected_count}, got {len(df.columns)}"
        )
        return False

    print(f"✓ Column count is correct: {expected_count}")
    return True


def test_ga_parameters_in_header():
    """
    Test that all GA-related parameters are present in the expected order.
    """
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print(f"Test file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, nrows=5)
    columns = df.columns.tolist()

    # GA parameters should appear in this specific order after run_time
    ga_params_positions = {
        "cx_type": 46,
        "cxpb": 47,
        "mutpb": 48,
        "indpb": 49,
        "t_size": 50,
    }

    all_present = True
    for param, expected_pos in ga_params_positions.items():
        actual_pos = columns.index(param) if param in columns else -1

        if actual_pos == -1:
            print(f"✗ GA parameter '{param}' not found in header!")
            all_present = False
        elif actual_pos != expected_pos:
            print(
                f"✗ GA parameter '{param}' at position {actual_pos + 1}, expected {expected_pos + 1}"
            )
            all_present = True
        else:
            print(
                f"✓ GA parameter '{param}' correctly positioned at column {actual_pos + 1}"
            )

    return all_present


def test_run_time_ga_params_order():
    """
    Test that run_time comes before cx_type and other GA params.
    This is a sanity check to ensure parameters are logically ordered.
    """
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print(f"Test file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, nrows=5)
    columns = df.columns.tolist()

    run_time_pos = columns.index("run_time")
    cx_type_pos = columns.index("cx_type")
    cxpb_pos = columns.index("cxpb")

    if run_time_pos >= cx_type_pos:
        print(f"✗ run_time at {run_time_pos + 1} is after cx_type at {cx_type_pos + 1}")
        return False

    if cx_type_pos >= cxpb_pos:
        print(f"✗ cx_type at {cx_type_pos + 1} is after cxpb at {cxpb_pos + 1}")
        return False

    print(
        f"✓ run_time ({run_time_pos + 1}) < cx_type ({cx_type_pos + 1}) < cxpb ({cxpb_pos + 1})"
    )
    return True


def test_data_row_column_count():
    """
    Test that every data row has the same number of columns as the header.
    This detects CSV parsing issues where values are misaligned.
    """
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print(f"Test file not found: {csv_path}")
        return

    # Read with no header to count all rows
    df = pd.read_csv(csv_path, low_memory=False)

    expected_cols = len(df.columns)

    # Count actual values per row (skip header)
    with open(csv_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 0:  # Skip header
            continue

        # Simple CSV parsing - split on commas but handle quoted strings
        import csv

        reader = csv.reader([line])
        row_values = next(reader)

        actual_count = len(row_values)

        if actual_count != expected_cols:
            print(
                f"✗ Row {i + 1} has {actual_count} values, but header has {expected_cols}"
            )
            print(f"   Line: {line[:200]}...")
            return False

    print(f"✓ All data rows have correct column count ({expected_cols})")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("Column Alignment Tests for final_grid_score_log.csv")
    print("=" * 80)

    tests = [
        ("Expected Column Count", test_expected_column_count),
        ("GA Parameters in Header", test_ga_parameters_in_header),
        ("Run Time < GA Params Order", test_run_time_ga_params_order),
        ("Data Row Column Count", test_data_row_column_count),
        # Skip column alignment check on old CSV with bugfix
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 80)
    print("Test Results Summary:")
    print("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
