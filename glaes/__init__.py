#from .core import (GlaesError,
#                   indicators,
#                   ExclusionCalculator,
#                   mappers,
#                   WeightedCriterionCalculator)


from .core import util
from .core.priors import Priors, setPriorDirectory
from .core.ExclusionCalculator import ExclusionCalculator
from .core.WeightedCriterionCalculator import WeightedCriterionCalculator
from .predefinedExclusions import ExclusionSets

from os.path import join as _join, dirname as _dirname, basename as _basename
from collections import OrderedDict as _OrderedDict
from glob import glob as _glob
_test_data_ = _OrderedDict()

for f in _glob(_join(_dirname(__file__), "test", "data", "*")):
    _test_data_[_basename(f)] = f