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

from os.path import join as _join, dirname as _dirname
_test_data_ = _join(_dirname(_dirname(__file__)), "testing", "data")