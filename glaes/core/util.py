import json
import logging
import re
import sys
from collections import OrderedDict, namedtuple
from difflib import SequenceMatcher as SM
from glob import glob
from os.path import basename, dirname, isdir, join, splitext
from warnings import warn

import geokit as gk
import numpy as np
import pandas as pd

# Configure Logging
glaes_logger = logging.getLogger("GLAES")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class GlaesError(Exception):
    pass
