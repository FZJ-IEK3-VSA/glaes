#from: /fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/create_prior.py


#%%
import geokit as gk
import numpy as np
from os.path import join, isdir, isfile, basename, splitext
from os import mkdir
import sys
from multiprocessing import Pool
import time
from datetime import datetime as dt
from glob import glob
from collections import namedtuple, OrderedDict
from json import dumps


'''
Ziel: #generische function
#1 
funktion zum erkennen ob raster oder vector data
try to load raster, if not ...
if vector data ... if raster data

#2 
generische evaluate funktion 
'''





#%%
#1. define sources
#shape file
airportsSource = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/Examples/aachen_placement_areas.shp"
# tif file
clcSource = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/glaes/test/data/roads_prior_clip.tif"


#%%
#2. evaluation values
# Indicates distances too close exclusion criterion
EVALUATION_VALUES = { 
    "agriculture_proximity":
    # Indicates distances too close to aggriculture areas (m)
    [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
        2500,
        3000,
        4000,
        5000,
    ],
    "airport_proximity":
    # Indicates distances too close to airports (m)
    [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1250,
        1500,
        1750,
        2000,
        2250,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
        7000,
        8000,
        9000,
        10000,
        15000,
    ],
}
#%%
#3. evaluation function
# tif
def evaluate_AGRICULTURE(regSource, ftrID, tail):
    name = "agriculture_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=(12, 22), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )


#%%
#shape
def evaluate_AIRPORT(regSource, ftrID, tail):
    ######################
    ## Evaluate airports
    name = "airport_proximity"
    unit = "meters"
    description = (
        "Indicates pixels which are less-than or equal-to X meters from an airport"
    )
    source = "EUROSTAT, CLC"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances) * 1.25)

    ### Get airport regions
    airportMask = reg.indicateValues(clcSource, value=6, applyMask=False) > 0.5
    airportGeoms = gk.geom.convertMask(airportMask, bounds=reg.extent.xyXY, srs=reg.srs)
    if airportGeoms is None:
        airportGeoms = []

    ### define an airport/airfield shape matcher
    def airportShapes(points, minSize, defaultRadius, minDistance=2000):
        locatedGeoms = []

        # look for best geometry for each airport
        for pt in points:
            found = False

            # First look for containing geometries greater than the minimal area
            containingGeoms = filter(lambda x: x.Contains(pt), airportGeoms)
            for geom in containingGeoms:
                if geom.Area() > minSize:
                    locatedGeoms.append(geom.Clone())
                    found = True
                if found:
                    continue
            if found:
                continue

            # Next look for nearby geometries greater than the minimal area
            nearbyGeoms = filter(lambda x: pt.Distance(x) <= minDistance, airportGeoms)
            for geom in nearbyGeoms:
                if geom.Area() > minSize:
                    locatedGeoms.append(geom.Clone())
                    found = True
                if found:
                    continue
            if found:
                continue

            # if all else fails, apply a default distance
            locatedGeoms.append(pt.Buffer(defaultRadius))

        if len(locatedGeoms) == 0:
            return None
        else:
            return locatedGeoms

    ### Locate airports
    airportWhere = "AIRP_USE!=4 AND (AIRP_PASS=1 OR AIRP_PASS=2) AND AIRP_LAND='A'"
    airportCoords = [
        point.Clone()
        for point, i in gk.vector.extractFeatures(
            airportsSource, reg.extent.box, where=airportWhere
        )
    ]
    for pt in airportCoords:
        pt.TransformTo(reg.srs)

    geom = airportShapes(airportCoords, minSize=1e6, defaultRadius=3000)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )

    #####################
    ## Evaluate airfields
    name = "airfield_proximity"
    unit = "meters"
    description = (
        "Indicates pixels which are less-than or equal-to X meters from an airfield"
    )
    source = "EUROSTAT, CLC"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    ### Locate airports
    airfieldWhere = "AIRP_USE!=4 AND (AIRP_PASS=0 OR AIRP_PASS=9) AND AIRP_LAND='A'"
    airfieldCoords = [
        point.Clone()
        for point, i in gk.vector.extractFeatures(
            airportsSource, reg.extent.box, where=airfieldWhere
        )
    ]
    for pt in airfieldCoords:
        pt.TransformTo(reg.srs)

    geom = airportShapes(airfieldCoords, minSize=1e6, defaultRadius=800)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )

#%%
#4
##################################################################
## UTILITY FUNCTIONS
def edgesByProximity(reg, geom, distances):
    # make initial matrix
    mat = (
        np.ones(reg.mask.shape, dtype=np.uint8) * 255
    )  # Set all values to no data (255)
    mat[reg.mask] = 254  # Set all values in the region to untouched (254)

    # Only do growing if a geometry is available
    if not geom is None and len(geom) != 0:
        # make grow func
        def doGrow(geom, dist):
            if dist > 0:
                if isinstance(geom, list) or isinstance(geom, filter):
                    grown = [g.Buffer(dist) for g in geom]
                else:
                    grown = geom.Buffer(dist)
            else:
                grown = geom

            return grown

        # Do growing
        value = 0
        for dist in distances:
            grown = doGrow(geom, dist)
            try:
                tmpSource = gk.vector.createVector(
                    grown
                )  # Make a temporary vector file
            except Exception as e:
                print(len(grown), [g.GetGeometryName() for g in grown])
                raise e

            indicated = reg.indicateFeatures(tmpSource) > 0.5  # Map onto the RegionMask

            # apply onto matrix
            sel = np.logical_and(
                mat == 254, indicated
            )  # write onto pixels which are indicated and available
            mat[sel] = value
            value += 1

    # Done!
    return mat


def edgesByThreshold(reg, source, thresholds):
    # make initial matrix
    mat = (
        np.ones(reg.mask.shape, dtype=np.uint8) * 255
    )  # Set all values to no data (255)
    mat[reg.mask] = 254  # Set all values in the region to untouched (254)

    # Only do growing if a geometry is available
    value = 0
    for thresh in thresholds:
        indicated = reg.indicateValues(source, value=(None, thresh)) > 0.5

        # apply onto matrix
        sel = np.logical_and(
            mat == 254, indicated
        )  # write onto pixels which are indicated and available
        mat[sel] = value
        value += 1

    # Done!
    return mat


def writeEdgeFile(
    result, reg, ftrID, output_dir, name, tail, unit, description, source, values
):
    # make output
    output = "%s.%s_%05d.tif" % (name, tail, ftrID)
    if not isdir(output_dir):
        mkdir(output_dir)

    valueMap = OrderedDict()
    for i in range(len(values)):
        valueMap["%d" % i] = "<=%.2f" % values[i]
    valueMap["254"] = "untouched"
    valueMap["255"] = "noData"

    meta = OrderedDict()
    meta["GLAES_PRIOR"] = "YES"
    meta["DISPLAY_NAME"] = name
    meta["ALTERNATE_NAME"] = "NONE"
    meta["DESCRIPTION"] = description
    meta["UNIT"] = unit
    meta["SOURCE"] = source
    meta["VALUE_MAP"] = dumps(valueMap)

    print(output)

    d = reg.createRaster(
        output=join(output_dir, output),
        data=result,
        overwrite=True,
        noDataValue=255,
        dtype=1,
        meta=meta,
    )


def geomExtractor(extent, source, where=None, simplify=None):
    searchGeom = extent.box
    if isinstance(source, str):
        searchFiles = [
            source,
        ]
    else:
        searchFiles = list(extent.filterSources(join(source[0], source[1])))

    geoms = []
    for f in searchFiles:
        for geom, attr in gk.vector.extractFeatures(
            f, searchGeom, where=where, outputSRS=extent.srs
        ):
            geoms.append(geom.Clone())

    if not simplify is None:
        newGeoms = [g.SimplifyPreserveTopology(simplify) for g in geoms]
        for g, ng in zip(geoms, newGeoms):
            if "LINE" in ng.GetGeometryName():
                test = ng.Length() / g.Length()
            else:
                test = ng.Area() / g.Area()

            if test < 0.97:
                raise RuntimeError(
                    "ERROR: Simplified geometry is >3% different from the original"
                )
            elif test < 0.99:
                print(
                    "WARNING: simplified geometry is slightly different from the original"
                )

    if len(geoms) == 0:
        return None
    else:
        return geoms


###################################################################
## MAIN FUNCTIONALITY
if __name__ == "__main__":
    START = dt.now()
    tail = str(int(dt.now().timestamp()))
    print("RUN ID: ", tail)
    print("TIME START: ", START)

    # Choose the function
    func = globals()["evaluate_" + sys.argv[1]]

    # Choose the source
    if len(sys.argv) < 3:
        source = join("reg", "aachenShapefile.shp")
    else:
        source = sys.argv[2]

    # Arange workers
    if len(sys.argv) < 4:
        doMulti = False
    else:
        doMulti = True
        pool = Pool(int(sys.argv[3]))

    # submit jobs
    res = []
    count = -1
    for g, a in gk.vector.extractFeatures(source):
        count += 1
        # if count<1 : continue
        # if count == 2:break

        # Do the analysis
        if doMulti:
            res.append(pool.apply_async(func, (source, count, tail)))
        else:
            func(source, count, tail)

    if doMulti:
        # Check for errors
        for r, i in zip(res, range(len(res))):
            try:
                r.get()
            except Exception as e:
                print("EXCEPTION AT ID: " + str(i))
                raise e

        # Wait for jobs to finish
        pool.close()
        pool.join()

    # finished!
    END = dt.now()
    print("TIME END: ", END)
    print("CALC TIME: ", (END - START))

# %%
