import glaes as gl
import geokit as gk
import numpy as np
import gdal
from os.path import join
import matplotlib.pyplot as plt
import warnings

aachenShape = join("data","aachenShapefile.shp")
clcRaster = join("data", "clc-aachen_clipped.tif")
priorSample = join("data", "roads_prior_clip.tif")
cddaVector = join("data","CDDA_aachenClipped.shp")

def compareF(v, expected, msg="None"):
    if not np.isclose(v, expected): 
        raise RuntimeError("Expected:%.8f,   Got:%.8f   ,MSG:%s"%(expected, v, msg))

def compare(v, expected, msg="None"):
    if not expected==v:
        raise RuntimeError("Expected:%s,   Got:%s   ,MSG:%s"%(str(expected), str(v), msg))