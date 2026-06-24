"""The Geospatial Land Availability for Energy Systems (GLAES) model is intended for land eligbility analysis in any context"""

__version__ = "1.3.0"

from collections import OrderedDict as _OrderedDict
from glob import glob as _glob
from os.path import basename as _basename
from os.path import dirname as _dirname
from os.path import join as _join

from .core import util
from .core.ExclusionCalculator import ExclusionCalculator
from .core.priors import Priors
from .core.WeightedCriterionCalculator import WeightedCriterionCalculator
from .predefinedExclusions import ExclusionSets

_test_data_ = _OrderedDict()

for f in _glob(_join(_dirname(__file__), "test", "data", "*")):
    _test_data_[_basename(f)] = f
