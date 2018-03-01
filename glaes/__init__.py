#from .core import (GlaesError,
#                   indicators,
#                   ExclusionCalculator,
#                   mappers,
#                   WeightedCriterionCalculator)


from .core import util
from .core.priors import Priors
from .core.ExclusionCalculator import ExclusionCalculator
from .core.WeightedCriterionCalculator import WeightedCriterionCalculator
from .predefinedExclusions import ExclusionSets

from os.path import join, dirname
_test_data_ = join(dirname(dirname(__file__)), "testing", "data")