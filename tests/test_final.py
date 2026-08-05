import os
import sys

sys.path.insert(0, "/workspaces/ensemble_genetic_algorithm")


# Clear any cached modules
for mod_name in list(sys.modules.keys()):
    if "ml_grid" in mod_name:
        del sys.modules[mod_name]

tmpdir = "/workspaces/ensemble_genetic_algorithm/test_run"
os.makedirs(tmpdir, exist_ok=True)
