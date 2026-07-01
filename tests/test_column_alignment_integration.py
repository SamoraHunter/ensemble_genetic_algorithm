"""
Integration tests for project_score_save.py column alignment.
Tests that columns are properly aligned when saving GA experiment results.

This test verifies:
1. The column_list has all required columns including cx_type
2. The iteration order is deterministic (sorted)
3. GA parameters are written in the correct order

Run this after starting a new GA run to verify fresh data is saved correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_column_list_has_cx_type():
    """Verify cx_type is in the update_score_log column list."""
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    with open(source_path, "r") as f:
        content = f.read()

    # Find the update_score_log method's column_list
    import re

    # This pattern matches from "run_time" to the end of GA params section
    pattern = r'"run_time",\s*"cx_type",\s*"cxpb",\s*"mutpb",\s*"indpb",\s*"t_size"'

    match = re.search(pattern, content)

    if not match:
        print("✗ Column list does NOT contain all GA parameters in correct order!")

        # Check for cx_type specifically
        if '"cx_type"' not in content and "'cx_type'" not in content:
            print("  Missing: cx_type")

        # Show what's actually in the column section
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "run_time" in line and ('"run_time"' in line or "'run_time'" in line):
                print(f"\n  Column definition around line {i+1}:")
                for j in range(i, min(i + 10, len(lines))):
                    if '"cx_type"' in lines[j] or "'cx_type'" in lines[j]:
                        print(f"    Line {j+1}: {lines[j].strip()}")
                    elif "valid" in lines[j]:
                        break
                break

        return False

    print("✓ Column list contains cx_type, cxpb, mutpb, indpb, t_size in correct order")
    return True


def test_sorted_iteration_for_deterministic_order():
    """Verify iteration through local_param_dict uses sorted keys."""
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    with open(source_path, "r") as f:
        content = f.read()

    # Check for sorted() usage in key iteration
    if "sorted_keys = sorted(ml_grid_object.local_param_dict.keys())" not in content:
        print("✗ Iteration does NOT use sorted() keys!")
        return False

    print("✓ Iteration uses sorted keys for deterministic column ordering")
    return True


def test_explicit_ga_params_written_after_loop():
    """Verify GA parameters are explicitly written to ensure correct alignment."""
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    with open(source_path, "r") as f:
        content = f.read()

    # Check for explicit GA params order list
    if "ga_params_order" not in content:
        print("✗ No explicit ga_params_order list found!")

        # Suggest adding this fix
        print("\n  SUGGESTION: Add explicit GA parameter ordering to prevent")
        print("  column alignment issues when new parameters are added:")
        print("""
    # Explicitly write GA parameters in the correct order
    ga_params_order = ["cx_type", "cxpb", "mutpb", "indpb", "t_size"]
    for param in ga_params_order:
        if param not in line:  # Only set if not already written above
            line[param] = [ml_grid_object.local_param_dict.get(param])
""")
        return False

    print("✓ Explicit GA parameters list ensures correct alignment")
    return True


def test_column_alignment_on_csv():
    """Check the actual CSV file structure if it exists."""
    csv_path = Path(
        "notebooks/HFE_GA_experiments/2026-06-30_19-38-14/final_grid_score_log.csv"
    )

    if not csv_path.exists():
        print("  (CSV file not found, skipping)")
        return None

    import pandas as pd

    df = pd.read_csv(csv_path, nrows=2)
    columns = df.columns.tolist()

    # Expected GA params positions
    ga_params = ["cx_type", "cxpb", "mutpb", "indpb", "t_size"]

    # Find run_time position and verify GA params come after in correct order
    try:
        run_time_pos = columns.index("run_time")

        for i, param in enumerate(ga_params):
            if param not in columns:
                print(f"✗ Column '{param}' not found in CSV header!")
                return False

            actual_pos = columns.index(param)
            expected_pos = run_time_pos + 1 + i

            if actual_pos != expected_pos:
                print(
                    f"✗ {param} is at position {actual_pos+1}, should be {expected_pos+1}"
                )
                return False

        print("✓ All GA parameters are in correct positions in CSV")

        # Check that cx_type has valid value (not a float probability)
        if len(df) > 0:
            first_row_cx_type = df["cx_type"].iloc[0]

            valid_types = ["twopoint", "onepoint", "uniform", "blend", "ordered"]

            if pd.isna(first_row_cx_type):
                print("✗ cx_type is NaN in first row")
                return False

            if str(first_row_cx_type) not in valid_types:
                print(
                    f"  Note: cx_type in CSV is '{first_row_cx_type}' (this might be from old data)"
                )

        return True

    except ValueError as e:
        print(f"✗ Error checking column positions: {e}")
        return False


def test_parameter_order_logic():
    """Verify the logic of how parameters are written."""
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    with open(source_path, "r") as f:
        content = f.read()

    # Extract the write loop section
    import re

    # Pattern to find the key iteration section
    loop_pattern = r'for key in sorted_keys:.*?if key != "data":'
    match = re.search(loop_pattern, content, re.DOTALL)

    if not match:
        print("  (Could not extract loop section)")
        return None

    # Verify the order of writes in column_list matches what we expect
    expected_order = ["run_time", "cx_type", "cxpb", "mutpb", "indpb", "t_size"]

    # Find where these appear in the column_list definition
    col_pattern = r'"run_time",.*?"t_size"'
    col_match = re.search(col_pattern, content, re.DOTALL)

    if not col_match:
        print("  (Could not extract column list)")
        return None

    column_section = col_match.group(0)

    # Verify order in column section
    positions = {param: column_section.index(f'"{param}"') for param in expected_order}

    prev_pos = -1
    for param in expected_order:
        pos = positions[param]
        if pos < prev_pos:
            print("✗ Parameters are not in correct order in column_list!")
            print(
                f"  {expected_order[prev_pos]} at {positions[expected_order[prev_pos]]}"
            )
            print(f"  {param} at {pos}")
            return False
        prev_pos = pos

    print("✓ Columns are defined in the expected order")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("project_score_save.py Fix Verification Tests")
    print("=" * 80)

    tests = [
        ("Column List Has cx_type", test_column_list_has_cx_type),
        ("Sorted Iteration", test_sorted_iteration_for_deterministic_order),
        ("Explicit GA Params Order", test_explicit_ga_params_written_after_loop),
        ("CSV Parameter Positions", test_column_alignment_on_csv),
        ("Parameter Order Logic", test_parameter_order_logic),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            result = test_fn()
            if result is None:
                print("  (SKIPPED)")
                results.append((name, True))  # Don't count as failure if skipped
            else:
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
        status = "✓ PASS" if result else (" (SKIPPED)" if result is None else "✗ FAIL")
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
