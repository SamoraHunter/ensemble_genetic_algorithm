import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Clear any cached modules
for mod_name in list(sys.modules.keys()):
    if "ml_grid" in mod_name:
        del sys.modules[mod_name]

tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_run")
os.makedirs(tmpdir, exist_ok=True)
