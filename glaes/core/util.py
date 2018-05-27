import geokit as gk
import re 
import numpy as np
from glob import glob 
from os.path import dirname, basename, join, isdir, splitext
from collections import namedtuple, OrderedDict
import json
from warnings import warn
from difflib import SequenceMatcher as SM
import pandas as pd

class GlaesError(Exception): pass