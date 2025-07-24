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



#%%
#   1. define sources
# some tif file from severin ?? content unknown
tif_file_ryberg = "zena_data/CLC/g100_clc12_V18_5_SRS_FIX.tif"

#shape file
shape_file = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/Examples/aachen_placement_areas.shp"
# tif file
tif_file_complicated = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/glaes/test/data/roads_prior_clip.tif"
tif_file = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/glaes/test/data/roads_prior_clip.tif"


#2. evaluation values
# Indicates distances too close to exclusion criterion
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


#3. evaluation function
# tif
def evaluate_tif(regSource, ftrID, tail, tif_file_ryberg):
    """creates a prior dataset from a shape file.

    Args:
        regSource (str): sets the path to the region shape file (could be also the path to a region mask)
        ftrID (str, int; optional): The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
        tail (str): extention to the "name". final file name will be: f"{name}.{tail}_{ftrID}.tif"
        tif_file_ryberg(str): string to the raster file which is used for "indicate_values" = Indicates those pixels in the RegionMask which correspond to a particular
        value, or range of values, from a given raster datasource
    """
    name = "agriculture_proximity"  #TODO the following things will later be written as meta data (should probably be part of the function arguments)
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)  #output path is beeing set

    # Get distances
    distances = EVALUATION_VALUES[name] #distances from distances dict

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, where=ftrID, padExtent=max(distances)) 
    #regSource = shapefile path
    #where = attribute feature
    #padExtent = buffer

#####
# Diese Funktion wird wohl nicht überall angewandt ?! 
# nur bei clc Daten


    # Indicate values from regionmask which match given raster values and create a geomoetry from the result
    matrix = reg.indicateValues(tif_file_ryberg, #Indicates those pixels in the RegionMask 
                                                 # which correspond to a particular value, or range of values, from a given raster datasource
                                value=(12, 22),  # values that are accepted 
                                applyMask=False) > 0.5 # region mask will not be applied
                                                 #on ylvalues which are > 0.5, means matrix, where values match raster values (=1) or match more than half (>0.5)
    
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs) #convert the array to a geom with mask extent
######

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )

def evaluate_tif_simple(regSource, ftrID, tail):
    name = "dni_threshold"
    unit = "kWh/m2/day"
    description = "Indicates pixels in which the average daily direct-normal irrandiance (DNI) is less-than or equal-to X kWh/m2/day"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    result = edgesByThreshold(reg, tif_file, thresholds)

    # make result
    writeEdgeFile(
        result,
        reg,
        ftrID,
        output_dir,
        name,
        tail,
        unit,
        description,
        source,
        thresholds,
    )
def evaluate_shape_simple(regSource, ftrID, tail):
    name = "railway_proximity"
    unit = "meters"
    description = (
        "Indicates pixels which are less-than or equal-to X meters from a railway"
    )
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor(extent=reg.extent, source=shape_file, where=r"fclass = 'rail'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )



##################################################################
## UTILITY FUNCTIONS
def calculate_distances(geom, dist):
    """
    Buffers geometry or geometries by a specified distance.
    This function applies a positive buffer (or "grow") operation to either a single geometry 
    or a collection of geometries. If `dist` is zero or negative, the original geometry is returned unchanged.


    Args:
        geom (geokit geometrie object): a geokit geometrie object (point, line, polygon) that is used for bffering
        dist (int; float): one distance of distance dict (List[float or int]): List of increasing distance values from Evaluation_values dict that are used for buffering

    Returns:
        Geometry or list of Geometries: The buffered geometry/geometries.
    """
    if dist > 0:
        if isinstance(geom, list) or isinstance(geom, filter): #if multiply geoms
            buffered_geoms = [g.Buffer(dist) for g in geom]
        else:                                                  # if one geom
            buffered_geoms = geom.Buffer(dist)
    else:                                                      # in no distance, no buffer. Geom left as is
        buffered_geoms = geom

    return buffered_geoms

def edgesByProximity(reg, geom, distances):
    """Calculating a labeled matrix where pixels closer to the geometry get lower values, 
    pixels further away gethigher values. Buffering/ Distance calculation follows the list of "distances"

    Args:
        reg (Regionmask): Regionmask with spatial extent for further processing
        geom (gk geometrie object): geometrie (point, line, polygon) of area/areas that shall be used for distance calculation
        distances (List[float or int]): distance dict (List[float or int]): List of increasing distance values from Evaluation_values dict that are used for buffering


    Returns:
        np.ndarray: A labeled matrix where each cell indicates its proximity zone (0 = closest)
    """

    # make initial matrix
    proximity_matrix = (np.ones(reg.mask.shape, dtype=np.uint8) * 255)  # Set all values to no data (255)
    proximity_matrix[reg.mask] = 254  # Set all values in the region to untouched (254)

    # Only do growing if a geometry is available
    if geom is not None and len(geom) != 0:

        # Do distance calculation
        value = 0 #sets the value of the first threshold
        
        for dist in distances:  
            #loops over the distance list
            
            buffered = calculate_distances(geom, dist) #buffers the geom by distance
            try:
                tmp_buffered_vector = gk.vector.createVector(buffered)  # Make a temporary vector file
            except Exception as e:
                print(len(buffered), [g.GetGeometryName() for g in buffered])
                raise e
            
            # Map onto the RegionMask al cells with more than 0.5 fit
            buffered_region = reg.indicateFeatures(tmp_buffered_vector) > 0.5  
            
            # apply onto matrix
            #select pixels which are valid (254) and part of the buffered region
            selected_matrix = np.logical_and(proximity_matrix == 254, buffered_region)  
            #selected cells are assigned to a value 
            proximity_matrix[selected_matrix] = value
            
            #label for the next threshold
            value += 1


    return proximity_matrix # Return the final labeled matrix


def edgesByThreshold(reg, source, thresholds):
    """
    Segments a raster source into threshold-based classes within a RegionMask,
    producing a labeled matrix. If a raster cell of thre region mask (valid cells 254) 
    falls into a threshold it's marked with a value

    Args:
        reg (Regionmask): Regionmask with spatial extent for further processing
        source (str): Path to a rasterfile
        thresholds (List[float or int]): List of increasing threshold values from Evaluation_values dict (should be categories/values of the source file)
    
    Returns: 
        np.ndarray: A labeled matrix with classes per threshold.
    """
    
    # make initial matrix
    threshold_matrix = (np.ones(reg.mask.shape, # use the outer shape of the region mask to create an array.
                   dtype=np.uint8) * 255)   #Set all values to no data (255)
    
    threshold_matrix[reg.mask] = 254  # Set all values in the region mask to untouched (254)


    value = 0 #sets the value of the first threshold
    
    for thresh in thresholds:
        # marks all values inside the region mask (mat) which are =< thresh of the raster
        # returns float matrix >0.5–1.0
        indicated = reg.indicateValues(source, value=(None, thresh)) > 0.5 

        # apply onto matrix: select pixels that are valid (254) and satisfy thresholf (indicated) (at the same time)
        selected_matrix = np.logical_and(threshold_matrix == 254, indicated)  
        
        # Assign the current class label to the selected pixels
        threshold_matrix[selected_matrix] = value
        
        #label for the next threshold
        value += 1

    return threshold_matrix # Return the final labeled matrix


def writeEdgeFile(
    result, reg, ftrID, output_dir, name, tail, unit, description, source, values
):
    """
    Writes a classified result matrix to a GeoTIFF file with metadata and value labeling.

    Args:
        result (np.ndarray): A 2D classified matrix (e.g., from thresholding or proximity analysis).
        reg (RegionMask): A RegionMask object used to define spatial extent and raster creation.
        ftrID (int or str): current feature, used in the output file name.
        output_dir (str): path where output raster will be saved.
        name (str): Base name for the output 
        tail (str): Suffix used to distinguish output variants (e.g., 'proximity', 'threshold').
        unit (str): Unit of the classified values (e.g., meters, degrees).
        description (str): explanation of the dataset.
        source (str): Source dataset or method used to generate the result.
        values (List[float]): List of thresholds or distance values used in classification.

    Returns:
        None: The function writes the raster file to disk; it does not return a value.
    """
    # make output
    output =  f"{name}.{tail}_{ftrID}.tif"
    
    #create dir if not exist
    if not isdir(output_dir):
        mkdir(output_dir)
        
    #Meta data mapping
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
    """
    Create a geometry list from the osm files. Not used yet

    Args:
        extent (Extent): extent of the RegionMask which is the study region
        source (str): Path to a vector file (e.g. shapefile)
        where (str or int, optional):             
            If string -> An SQL-like where statement to apply to the source
            If int -> The feature's ID within the dataset
                * Feature attribute name do not need quotes
                * String values should be wrapped in 'single quoteDefaults to None.
        simplify (float, optional): Distance tolerance for geometry simplification using 
            `SimplifyPreserveTopology`. Useful for reducing geometry complexity.


    Returns:
       list of geometries or None: A list of OGR geometry objects, or None if no features found.
    """
    
    # Get the bounds (box) of the extent to use as spatial filter
    searchGeom = extent.box
    # Prepare list of source files
    if isinstance(source, str): #sinle file
        searchFiles = [
            source,
        ]
    else:   #multiple
        searchFiles = list(extent.filterSources(join(source[0], source[1])))

    geoms = []
    # Iterate through each source file 
    # and extract features within search geometry
    for f in searchFiles:
        for geom, attr in gk.vector.extractFeatures(
            f, searchGeom, where=where, outputSRS=extent.srs
        ):
            geoms.append(geom.Clone())
            
     # If simplification
    if simplify is not None:
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
    # Return list or None if empty
    return geoms if geoms else None


# ###################################################################
# ## MAIN FUNCTIONALITY
# if __name__ == "__main__":
#     START = dt.now()
#     tail = str(int(dt.now().timestamp()))
#     print("RUN ID: ", tail)
#     print("TIME START: ", START)

#     # Choose the function
#     func = globals()["evaluate_" + sys.argv[1]]

#     # Choose the source
#     if len(sys.argv) < 3:
#         source = join("reg", "aachenShapefile.shp")
#     else:
#         source = sys.argv[2]

#     # Arange workers
#     if len(sys.argv) < 4:
#         doMulti = False
#     else:
#         doMulti = True
#         pool = Pool(int(sys.argv[3]))

#     # submit jobs
#     res = []
#     count = -1
#     for g, a in gk.vector.extractFeatures(source):
#         count += 1
#         # if count<1 : continue
#         # if count == 2:break

#         # Do the analysis
#         if doMulti:
#             res.append(pool.apply_async(func, (source, count, tail)))
#         else:
#             func(source, count, tail)

#     if doMulti:
#         # Check for errors
#         for r, i in zip(res, range(len(res))):
#             try:
#                 r.get()
#             except Exception as e:
#                 print("EXCEPTION AT ID: " + str(i))
#                 raise e

#         # Wait for jobs to finish
#         pool.close()
#         pool.join()

#     # finished!
#     END = dt.now()
#     print("TIME END: ", END)
#     print("CALC TIME: ", (END - START))

# %%
