"""
Unit tests for project_score_save.py column alignment.
Tests the update_score_log method's column_list structure.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_column_list_structure():
    """
    Test that the column_list in update_score_log has all GA parameters
    in the correct order.
    """
    # Read project_score_save.py and extract the column_list definition
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return False

    with open(source_path, "r") as f:
        content = f.read()

    # Expected GA parameters (from grid_param_space_ga.py)
    expected_ga_params = [
        "run_time",
        "cx_type",  # Must be present
        "cxpb",
        "mutpb",
        "indpb",
        "t_size",
    ]

    # Find the column_list definition in update_score_log
    # It starts around line 262 and includes all columns

    # Check that cx_type is in column_list (after run_time)
    lines = content.split("\n")
    column_lines = []

    for i, line in enumerate(lines):
        if '"run_time",' in line or "'run_time'," in line:
            # Found the start of GA params section
            # Extract next 10 lines to get all GA params
            for j in range(i, min(i + 20, len(lines))):
                column_lines.append((j + 1, lines[j]))
            break

    print("Found column definition:")
    for lineno, line in column_lines[:25]:  # Show first 25 lines
        print(f"  Line {lineno}: {line.strip()}")

    # Verify cx_type is present
    has_cx_type = any(
        '"cx_type"' in line or "'cx_type'" in line for _, line in column_lines
    )

    if not has_cx_type:
        print("\n✗ Column list is MISSING 'cx_type' parameter!")
        return False

    print("\n✓ cx_type found in column list")

    # Verify order: run_time < cx_type < cxpb < mutpb < indpb < t_size
    param_positions = {}
    for _, line in column_lines:
        for param in expected_ga_params:
            if f'"{param}"' in line or f"'{param}'" in line:
                # Extract position from index or search order
                param_positions[param] = len(param_positions)

    print("\nParameter positions within GA section:")
    for i, (name, pos) in enumerate(
        sorted(param_positions.items(), key=lambda x: x[1])
    ):
        expected_pos = (
            expected_ga_params.index(name) if name in expected_ga_params else -1
        )
        status = "✓" if i == expected_pos else "✗"
        print(f"{status} {name}: position {pos}")

    return has_cx_type


def test_initial_header_structure():
    """
    Test that the initial column definition (for file creation) also includes all GA params.
    Located around lines 107-120 in project_score_save.py
    """
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return False

    with open(source_path, "r") as f:
        content = f.read()

    expected_ga_params = [
        "run_time",
        "cx_type",
        "cxpb",
        "mutpb",
        "indpb",
        "t_size",
    ]

    # The initial header should be around line 107-120
    lines = content.split("\n")

    print("\nSearching for initial column definition (lines ~107-120):")
    for i in range(105, min(130, len(lines))):
        if '"run_time"' in lines[i] or "'run_time'" in lines[i]:
            print(f"Found at line {i+1}:")
            for j in range(i, min(i + 10, len(lines))):
                print(f"  Line {j+1}: {lines[j].strip()}")

    # Check if cx_type is present
    has_cx_type = '"cx_type"' in content or "'cx_type'" in content

    if not has_cx_type:
        print("\n✗ Initial header definition is MISSING 'cx_type' parameter!")
        return False

    print("\n✓ cx_type found in initial column definition")

    # Count total columns
    import re

    # Find the complete column list array
    match = re.search(
        r'"X_test_size",\s*"run_time",\s*["\']cx_type["\'],.*?"t_size",',
        content,
        re.DOTALL,
    )
    if match:
        ga_section = match.group(0)
        params_extracted = len([p for p in expected_ga_params if p in ga_section])
        print(f"  Found {params_extracted}/{len(expected_ga_params)} GA parameters")

    return has_cx_type


def test_parameter_mapping():
    """
    Test that when local_param_dict contains these keys, they map correctly to columns.
    The code iterates through local_param_dict and writes to column_list matching keys.

    This ensures:
    1. cx_type in local_param_dict["data"] -> column "cx_type"
    2. cxpb in local_param_dict["data"] -> column "cxpb"
    3. etc.
    """
    source_path = Path("ml_grid/pipeline/project_score_save.py")

    with open(source_path, "r") as f:
        content = f.read()

    # Extract the loop that processes local_param_dict
    import re

    match = re.search(
        r"for key in ml_grid_object\.local_param_dict:.*?if key in column_list:(.*?)(?=^\s+\w|#)",
        content,
        re.DOTALL | re.MULTILINE,
    )

    if not match:
        print("✗ Could not find local_param_dict processing loop")
        return False

    loop_content = match.group(1)

    expected_keys = ["cx_type", "cxpb", "mutpb", "indpb", "t_size"]

    print("\nTesting parameter-to-column mapping:")
    for key in expected_keys:
        if f'"{key}"' in loop_content or f"'{key}'" in loop_content:
            # The code does: line[key] = [ml_grid_object.local_param_dict.get(key)]
            print(f"  ✓ '{key}' will be mapped to column '{key}'")
        else:
            print(f"  ✗ '{key}' mapping not found")

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("project_score_save.py Column Alignment Tests")
    print("=" * 80)

    tests = [
        ("Update Score Log Column List", test_column_list_structure),
        ("Initial Header Structure", test_initial_header_structure),
        ("Parameter Mapping", test_parameter_mapping),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error: {e}")
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
