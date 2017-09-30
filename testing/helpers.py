import glaes as gl
import geokit as gk
import numpy as np
from os.path import join

aachenShape = join("data","aachenShapefile.shp")

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

