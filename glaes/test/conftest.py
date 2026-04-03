import os
import sys

# Set matplotlib to non-interactive backend before any matplotlib imports.
os.environ.setdefault("MPLBACKEND", "Agg")


import matplotlib

matplotlib.use("Agg")
