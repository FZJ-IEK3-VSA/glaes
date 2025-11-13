import geokit as gk
import numpy as np
from os.path import join, isdir
import os
from multiprocessing import Pool
from datetime import datetime as dt
from collections import namedtuple, OrderedDict
from json import dumps
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path



## CREATE PRIORS OVERALL PLAN  ##
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
    

    
####################    
####    TODO    ####
####################

# NOTE
# column 138: evaluate_tif_by_proximity is now excepting tis and shapes as input. the conversion from tif to regionmask
#             needs to be improved and then testet. Afterwards:

# First: from those three or four proximity/threshold accepting either tifs of shapes in input, 
#           create one threshold and one proximity function. Accepting shape and or tif files as input

#Second: clean up the function. Create a dict for the thresholds, meta data etc. that is importet for the 
#            specific "name" 

# Third: now that both functions work for buffering tif file,
#       the same function should be created for working with shape files as well

########################################################################################

#   1. define sources
base_path = Path(__file__).resolve().parents[0]
shape_file_in = str(base_path/"/input/aachenShapefile.shp") # has to be a string bacause pathlib paths produce problems for geookit
shape_file_clip = str(base_path/"input/Natura2000_aachenClipped.shp") # has to be a string bacause pathlib paths produce problems for geookit
tif_file_elevation = str(base_path/"input/roads_prior_clip.tif")
output_dir = str(base_path/"output/intermediates/")

#2. evaluation values
# Indicates distances too close to exclusion criterion:
# These distances are used for buffering around the selected feature of interest/ values of interest.
# values should match the input extent (if in 100 m resolution distances should not be less (e.g. 10, 20,30,etc.))
# values refer to the unit used in the srs (meter or degree)
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
        1000,
        2000,
        3000,
        4000,
    ],
    "dni_threshold":
    # Indicates distances too close to airports (m)
    [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
    ],
    "windspeed_50m_threshold":
    # Indicates areas with average wind speed below X (m/s)
    np.linspace(0, 20, 81),
}

#3. evaluation function
def evaluate_tif_by_proximity(Area, ftrID, target_tif, output_dir=None, raster_target_value=None):
    """creates a prior dataset from a tif file. Loads the evaluation values 
    and classifies cells around the target area by proximity.

    Args:
        Area (str): Region mask path. Sets the outline
        ftrID (str, int; optional): The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            * e.g. where = "ISO='DEU' AND POP>1000"
        target_tif(str): string to the raster file which is used for "indicate_values" = Indicates those pixels in the RegionMask which correspond to a particular
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
    
    suffix = os.path.splitext(Area)[1].lower()

    if suffix == ".shp":
        # Make Region Mask
        print("loading region of interest")
        reg = gk.RegionMask.load(region=Area, where=ftrID, padExtent=max(distances)) 
        #Area = shapefile path
        #where = attribute feature
        #padExtent = buffer
        
    elif suffix in (".tif", ".tiff"):
        #raster to matrix
        #matrix to region mask
        print("TODO") # there needs to be a different way
        raster_1 = gk.raster.loadRaster(Area)
        raster_matrix = gk.raster.extractMatrix(raster_1)
        ras_boolean = gk.raster.createRasterLike(source=raster_1, data=raster_matrix == 1, noData=0)
        vector_2 = gk.raster.polygonizeRaster(ras_boolean)
        vector_2 = str(base_path/"input/natura_xxx.shp")
        vector = gk.vector.createVector(vector_2)
        reg_1 = gk.RegionMask.load(vector_2)

    else:
        raise TypeError("Area must be a string ending with .tiff/.tif/.shp.")
  
    #Debugging 1
    filename = os.path.join(output_dir, f"region_of_interest.tif") 
    example_data = reg.mask.astype(int)  
    gk.raster.createRaster(data=example_data, output=filename, bounds=reg.extent.xyXY, srs=reg.srs)

    # 2. prepare the target tif file
    # NOTE The following function is probably not used everywhere?! 
    # Only for clc data ?
    # TODO simply overand a geometry or tif file?
    # Indicate values from regionmask which match given raster values and create a geomoetry from the result
    print("setting region/feature of interest")
    matrix = reg.indicateValues(target_tif, #Indicates those pixels in the RegionMask 
                                                 # which correspond to a particular value, or range of values, from a given raster datasource
                                value=raster_target_value,  # values that are accepted 
                                applyMask=False) > 0.5 # region mask will not be applied
                                                 #on ylvalues which are > 0.5, means matrix, where values match raster values (=1) or match more than half (>0.5)
                                                 
    target_area = gk.geom.polygonizeMatrix(matrix, bounds=reg.extent.xyXY, srs=reg.srs) #convert the array to a geom with mask extent
    
    #Debugging 2
    filename = os.path.join(output_dir, f"feature_of_interest_proximity.tif")
    gk.raster.createRaster(data=matrix, output=filename, bounds=reg.extent.xyXY, srs=reg.srs)



    # 3. create proximity matrix
    # region mask, gk_geom (like a polygon), distances from evaluation_values
    result = edgesByProximity(reg, target_area, distances)  

    # 4. make result
    writeEdgeFile(
        result, reg, ftrID, output_dir, name, unit, description, source, distances
    )
    
def evaluate_tif_by_threshold(Area, #area to analyze
                              ftrID, #set the polygon or cell values that define th area
                              tif_file, #tif file to analyze
                              output_dir=None,
                              ):
    name = "dni_threshold"
    unit = "kWh/m2/day"
    description = "Indicates pixels in which the average daily direct-normal irrandiance (DNI) is less-than or equal-to X kWh/m2/day"
    source = ""

    output_dir = output_dir

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(Area, where=ftrID, padExtent=500)

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
        unit,
        description,
        source,
        thresholds,
    )
    
def evaluate_SHAPE_by_proximity(Area, target_shape_file, ftrID_1=None, ftrID_2=None, output_dir=None, ):
    """creates a prior dataset from a tif file. Loads the evaluation values 
    and classifies cells around the target area by proximity.

    Args:
        Area (str): Region mask path. Sets the outline
        ftrID (str, int; optional): The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            * e.g. where = "ISO='DEU' AND POP>1000"
        target_tif(str): string to the raster file which is used for "indicate_values" = Indicates those pixels in the RegionMask which correspond to a particular
        value, or range of values, from a given raster datasource

        
    """
    
    #TODO the following things will later be written as meta data (should probably be part of the function arguments)
    name = "agriculture_proximity"  #name used for identifying the proximity values
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an agriculture area"
    source = "Some_source"

    output_dir = join(output_dir, name)  #output path is beeing set

    # Get distances
    distances = EVALUATION_VALUES[name] #distances from distances dict

    # 1. load region of interest
    #target_target_shape = gk.RegionMask.load("path",bounds=reg.extent.xyXY, srs=reg.srs)
    # Make Region Mask
    print("loading region of interest")
    reg = gk.RegionMask.load(region=Area, where=ftrID_1, padExtent=max(distances)) 
    #Area = target_shapefile path
    #where = attribute feature
    #padExtent = buffer
    
    
    # 2. set the area
    if ftrID_2:
        target_area = gk.vector.extractFeatures(target_shape_file, srs=reg.srs, where = ftrID_2 )
    else:    
        target_area = gk.vector.extractFeatures(target_shape_file, srs=reg.srs)
    



    # 3. create proximity matrix
    # region mask, gk_geom (like a polygon), distances from evaluation_values
    result = edgesByProximity(reg, target_area, distances)  
    
    
    # debugging
    name = name + "by_shape"
    # 4. make result
    writeEdgeFile(
        result, reg, ftrID_1, output_dir, name, unit, description, source, distances
    )

########################################################################################
## UTILITY FUNCTIONS
def calculate_distances(region_df_clipped, dist):
    """
    Buffers geometry of a df or geometries by a specified distance.
    This function applies a positive buffer (or "grow") operation to either a single geometry 
    or a collection of geometries. If `dist` is zero or negative, the original geometry is returned unchanged.


    Args:
        region_df_clipped (pandas df with geometry column): a geokit geometrie object (point, line, polygon) that is used for bffering
        dist (int; float): one distance of distance dict (List[float or int]): List of increasing distance values from Evaluation_values dict that are used for buffering

    Returns:
        Geometry or list of Geometries: The buffered geometry/geometries.
    """
    #print(type(geom))
    if dist > 0:
        # if isinstance(geom, list) or isinstance(geom, filter):
        if len(region_df_clipped) > 0:
            buffered_geom = [g.Buffer(dist) for g in region_df_clipped.geom]
            #buffered_geom = [g.Buffer(float(dist)) for g in geom]
        else:
            buffered_geom = region_df_clipped.Buffer(dist)
            
    else:
        buffered_geom = region_df_clipped

    return buffered_geom

def  edgesByProximity(reg, target_area, distances):
    """ distances from a target area are beeing calculated. Each distance area gets a unique value. Output matrix

    Args:
        reg (Regionmask): Regionmask with spatial extent for further processing
        target_area (pandas df with geometry column): 
        distances (List[float or int]): distance dict (List[float or int]): List of increasing distance values from Evaluation_values dict that are used for buffering


    Returns:
        np.ndarray: A labeled matrix where each cell indicates its proximity zone (0 = closest)
    """

    # make initial matrix
    proximity_matrix = (np.ones(reg.mask.shape, dtype=np.uint8) * 255)  # Set all values to no data (255)
    proximity_matrix[reg.mask] = 254  # Set all values in the region to untouched (254)

    # Only do growing if a geometry is available
    if target_area is not None and len(target_area) != 0:

        # Do distance calculation
        #value = 0 #sets the value of the first threshold
        
        for dist in distances:
            print(f"calculating proximity for {dist} around the feature of interest in the region of interest")  
            #loops over the distance list
            # instead of value = 0, value as distance
            value = dist
            buffered = calculate_distances(target_area, dist) #buffers the geom by distance
            try:
                tmp_buffered_vector = gk.vector.createVector(buffered)  # Make a temporary vector file
            except Exception as e:
                print(len(buffered), [g.GetGeometryName() for g in buffered])
                raise e
            
            # Map onto the RegionMask al cells with more than 0.5 fit
            buffered_region = reg.indicateFeatures(tmp_buffered_vector) > 0.5  

            #Debugging 3
            axh = reg.drawImage(
                    buffered_region,
                    figsize=(5, 6),
                    cmap="Reds",
                )
            
            output_dir = str(base_path/"output/intermediates/")
            filename = os.path.join(output_dir, f"buffered_region_of_interest_{dist}_SHAPE.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.show()
            
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



    
    for thresh in thresholds:
        print(f"setting the threshold: {thresh} around the feature of interest in the region of interest") 
       
        value = thresh
        # marks all values inside the region mask (mat) which are =< thresh of the raster
        # returns float matrix >0.5–1.0
        indicated = reg.indicateValues(source, value=(None, thresh)) > 0.5 
        
        #Debugging 3
        axh = reg.drawImage(
                indicated,
                figsize=(5, 6),
                cmap="Reds",
            )
            
        output_dir = str(base_path/"output/intermediates/")
        filename = os.path.join(output_dir, f"threshold_region_of_interest_{thresh}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        # apply onto matrix: select pixels that are valid (254) and satisfy thresholf (indicated) (at the same time)
        selected_matrix = np.logical_and(threshold_matrix == 254, indicated)  
        
        # Assign the current class label to the selected pixels
        threshold_matrix[selected_matrix] = value
        
        #label for the next threshold


    return threshold_matrix # Return the final labeled matrix

def writeEdgeFile(
    result, reg, ftrID, output_dir, name, unit, description, source, values,
):
    """
    Writes a classified result matrix to a GeoTIFF file with metadata and value labeling.

    Args:
        result (np.ndarray): A 2D classified matrix (e.g., from thresholding or proximity analysis).
        reg (RegionMask): A RegionMask object used to define spatial extent and raster creation.
        ftrID (int or str): current feature, used in the output file name.
        output_dir (str): path where output raster will be saved.
        name (str): Base name for the output 
        unit (str): Unit of the classified values (e.g., meters, degrees).
        description (str): explanation of the dataset.
        source (str): Source dataset or method used to generate the result.
        values (List[float]): List of thresholds or distance values used in classification.

    Returns:
        None: The function writes the raster file to disk; it does not return a value.
    """
    # make output

    output = f"{name}.tif"
    #create dir if not exist
    if not isdir(output_dir):
        os.mkdir(output_dir)
        
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
    
    print(f"saving proximity output {output}")
    
    d = reg.createRaster(
        output=join(output_dir, output),
        #output=output_dir,
        data=result,
        overwrite=True,
        noDataValue=255,
        dtype=1,
        meta=meta,
    )

########################################################################################
## TESTING
# evaluate_tif_by_proximity(          
#     Area=shape_file,
#     ftrID= "SITENAME='Kermeter'",
#     target_tif=tif_file_elevation,
#     raster_target_value=[3], 
#     output_dir = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/deletme_create_prior/output"
# )
# evaluate_tif_by_threshold(           
#     Area=shape_file_clip,
#     ftrID= "SITENAME='Fagnes du Nord-Est'",
#     tif_file=tif_file_elevation,
#     output_dir = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/deletme_create_prior/output"
# )
evaluate_SHAPE_by_proximity(           
    Area=shape_file_in,
    target_shape_file=shape_file_clip,
    output_dir = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/ethos-installation/ethos_suite_repositories/glaes/deletme_create_prior/output"
)