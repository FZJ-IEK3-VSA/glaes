
import geokit as gk
import numpy as np
from os.path import join, isdir
from os import mkdir
from multiprocessing import Pool
from datetime import datetime as dt
from collections import namedtuple, OrderedDict
from json import dumps
from pathlib import Path



## CREATE PRIORS ##
# there are two types of priors using different methods. 
# First: evaluate by Proximity
    # indicate pixels by their distance to the target area (pixels to close)
    # gets a geom object. The geom object is mostly created from a matrix but can also be from: geomExtractor()
    # proximity: list of buffer distances
    # for each distance a buffer around the target is created and marked. 
    # mark the distance from a natural habitat with 100m, 200m, 300m, etc.
    
# Second: evaluate by threshold
    # indicate values of a file  above a certain threshold
    # gets a tif file file path
    # thresholds: cut offs.
    # for each threshold values <= that threshold are marked
    # elevation <= 10, <= 12 etc.
    
    
# TODO
# make evaluate_tif_by_proximity_new and evaluate_tif_simple_by_threshold running. 
# Think about smart things to overhand as arguments. Thing about other smart things to make the functions better
# create one general function which can be applied to either shape or tif file. Can all files just be treated as region mask and problem slved?
########################################################################################

#   1. define sources
base_path = Path(__file__).resolve().parents[1]
shape_file = str(base_path/"glaes/test/data/Natura2000_aachenClipped.shp") # has to be a string bacause pathlib paths produce problems for geookit
tif_file_elevation = str(base_path/"glaes/test/data/elevation.tif")

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
    ],
    "windspeed_50m_threshold":
    # Indicates areas with average wind speed below X (m/s)
    np.linspace(0, 20, 81),
}

#3. evaluation function
def evaluate_tif_by_proximity(regSource, ftrID, tail, tif_file_ryberg, output_dir=None, raster_target_value=None):
    """creates a prior dataset from a tif file. Loads the evaluation values 
    and classifies cells around the target area by proximity.

    Args:
        regSource (str): Region mask path. Sets the outline
        ftrID (str, int; optional): The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            * e.g. where = "ISO='DEU' AND POP>1000"
        tail (str): extention to the "name". final file name will be: f"{name}.{tail}_{ftrID}.tif"
        tif_file_ryberg(str): string to the raster file which is used for "indicate_values" = Indicates those pixels in the RegionMask which correspond to a particular
        value, or range of values, from a given raster datasource
        raster_target_value : tuple, defines values of raster that will be selected for proximity calculation
        
    """
    #TODO the following things will later be written as meta data (should probably be part of the function arguments)
    name = "agriculture_proximity"  #name used for identifying the proximity values
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an agriculture area"
    source = "Some_source"

    output_dir = join(output_dir, name)  #output path is beeing set

    # Get distances
    distances = EVALUATION_VALUES[name] #distances from distances dict

    # 1. set the area
    # Make Region Mask
    reg = gk.RegionMask.load(region=regSource, where=ftrID, padExtent=max(distances)) 
    #regSource = shapefile path
    #where = attribute feature
    #padExtent = buffer
    

    # 2. prepare the target tif file
    # NOTE The following function is probably not used everywhere?! 
    # Only for clc data ?
    # TODO simply overand a geometry or tif file?
    # Indicate values from regionmask which match given raster values and create a geomoetry from the result
    matrix = reg.indicateValues(tif_file_ryberg, #Indicates those pixels in the RegionMask 
                                                 # which correspond to a particular value, or range of values, from a given raster datasource
                                value=raster_target_value,  # values that are accepted 
                                applyMask=False) > 0.5 # region mask will not be applied
                                                 #on ylvalues which are > 0.5, means matrix, where values match raster values (=1) or match more than half (>0.5)
    
    geom = gk.geom.polygonizeMatrix(matrix, bounds=reg.extent.xyXY, srs=reg.srs) #convert the array to a geom with mask extent


    # 3. create proximity matrix
    # region mask, gk_geom (like a polygon), distances from evaluation_values
    result = edgesByProximity(reg, geom, distances)  

    # 4. make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, tail, unit, description, source, distances
    )
def evaluate_tif_by_threshold(regSource, #area to analyze
                              ftrID, #set the polygon or cell values that define th area
                              tif_file, #tif file to analyze
                              tail,
                              output_dir=None,
                              ):
    name = "dni_threshold"
    unit = "kWh/m2/day"
    description = "Indicates pixels in which the average daily direct-normal irrandiance (DNI) is less-than or equal-to X kWh/m2/day"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, where=ftrID, padExtent=500)

    #TODO
    #is tif file warping neccessary??
    #tif_file= gk.raster.warp(tif_file)

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
def evaluate_shape_by_proximity(regSource, ftrID, tail):
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
def evaluate_shape_by_threshold(regSource, ftrID, tail, tif_file_ryberg, shape_target_value=(10,11,12,13,14,15)):
    print ("This is just for consistency. In fact, we want a proximity and a threshold function, regardless of the input type")


########################################################################################
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
    """ distances from a target area are beeing calculated. Each distance area gets a unique value. Output matrix

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
    Thresholds of a raster file are beeing marked  as a matrix

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
    result, reg, ftrID, output_dir, name, tail, unit, description, source, values,
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
        #output=output_dir,
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
########################################################################################
## TESTING
evaluate_tif_by_proximity(
    regSource=shape_file,
    ftrID= "SITENAME='Buchenwälder bei Zweifall'",
    tail="testing",
    tif_file_ryberg=tif_file_elevation,
    raster_target_value=(10,11,12,13,14,15), #hope that makes snese
    output_dir = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/deletme_create_prior/output"
)
evaluate_tif_by_threshold(
    regSource=shape_file,
    ftrID= "SITENAME='Buchenwälder bei Zweifall'",
    raster_target_value=(10,11,12,13,14,15),
    tail="testing",
    tif_file=tif_file_elevation,
    output_dir = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/deletme_create_prior/output"
)