"""Run the 89 checks of the 9 functions in rome_in_a_day.ipynb from a terminal, without starting Jupyter.

    python scripts/run_checks.py

The toolbox finds the functions by name in the __main__ module (toolbox/_yours.py), so the notebook cells that
define them are run here first, in this script's own namespace.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, "rome_in_a_day.ipynb")
FUNCTIONS = ("tfidf_matrix", "top_k_neighbours", "match_descriptors", "query_expansion", "connected_components",
             "homography_dlt", "homography_ratio", "triangulate_points", "umeyama")
HELPERS = ("_transfer_error",)  # defined in the same cell as homography_ratio

sys.path.insert(0, REPO)
import numpy as np  # noqa: E402,F401  (the cells use np, sp and the toolbox names)
import scipy.sparse as sp  # noqa: E402,F401
from toolbox import *  # noqa: E402,F401,F403

with open(NOTEBOOK, encoding="utf-8") as f:
    cells = json.load(f)["cells"]
starts = tuple(f"def {name}(" for name in FUNCTIONS + HELPERS)
for cell in cells:
    source = "".join(cell["source"])
    if cell["cell_type"] == "code" and source.lstrip().startswith(starts):
        exec(compile(source, f"rome_in_a_day.ipynb:{cell.get('id', '?')}", "exec"))
missing = [name for name in FUNCTIONS if name not in globals()]
if missing:
    sys.exit(f"not found in {NOTEBOOK}: {', '.join(missing)}")

run_all_tests()  # noqa: F405
