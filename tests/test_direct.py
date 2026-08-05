import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct test - simulate exactly what update_score_log does

import pandas as pd

tmpdir = tempfile.mkdtemp()

# Step 1: Create the initial CSV (this is what project_score_save_class.__init__ does)
column_list = [
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
    "cx_type",
    "cxpb",
    "mutpb",
    "indpb",
    "t_size",
    "mutpb_adaptive_enabled",
    "niche_params_enabled",
    "valid",
    "generation_progress_list",
    "best_ensemble",
    "original_feature_names",
    "t_fits",
    "n_fits",
]

df = pd.DataFrame(data=None, columns=column_list)
csv_path = os.path.join(tmpdir, "final_grid_score_log.csv")
df.to_csv(csv_path, mode="w", header=True, index=False)

print("Step 1: Initial CSV created")
print("Columns:", list(df.columns))

# Step 2: Create a line to append (this simulates what update_score_log does)
line = pd.DataFrame(data=None, columns=column_list)

# Set values for GA params
line["cx_type"] = ["twopoint"]
line["cxpb"] = [0.5]
line["mutpb"] = [0.2]
line["indpb"] = [0.05]
line["t_size"] = [3]

# Set our new columns (simulating the code in update_score_log)
mutpb_adaptive_config = {"enabled": False}
niche_config = {"enabled": False}

line["mutpb_adaptive_enabled"] = [mutpb_adaptive_config.get("enabled", False)]
line["niche_params_enabled"] = [niche_config.get("enabled", False)]

# Write to CSV
df_existing = pd.read_csv(csv_path)
print("\nStep 2: Before appending new row")
print("Existing columns:", list(df_existing.columns))

# Now append - this is what line[column_list].to_csv does
line_filtered = line[column_list]
line_filtered.to_csv(csv_path, mode="a", header=False, index=False)

# Read back
df_result = pd.read_csv(csv_path)
print("\nStep 3: After appending new row")
print("Result columns:", list(df_result.columns))
print()
print("DataFrame:")
print(df_result.to_string())

# Check if our columns exist
if "mutpb_adaptive_enabled" in df_result.columns:
    print(
        f"\n✓ mutpb_adaptive_enabled exists with value: {df_result['mutpb_adaptive_enabled'].iloc[0]}"
    )
else:
    print("\n✗ mutpb_adaptive_enabled NOT FOUND")

if "niche_params_enabled" in df_result.columns:
    print(
        f"✓ niche_params_enabled exists with value: {df_result['niche_params_enabled'].iloc[0]}"
    )
else:
    print("✗ niche_params_enabled NOT FOUND")

shutil.rmtree(tmpdir)
