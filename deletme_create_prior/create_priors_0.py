import os

CONDA_ENV = "/fast/home/l-madeisky/.conda/envs/preprocessing_data"

os.environ["CONDA_PREFIX"] = CONDA_ENV
os.environ["PROJ_LIB"] = f"{CONDA_ENV}/share/proj"
os.environ["GDAL_DATA"] = f"{CONDA_ENV}/share/gdal"


import geokit as gk
import numpy as np
import os
import os
from multiprocessing import Pool
from datetime import datetime as dt
from collections import namedtuple, OrderedDict
from json import dumps
from pathlib import Path
import matplotlib.pyplot as plt



                            ## CREATE PRIORS  ##

# there are two types of priors using different methods. 
#                           evaluate by Proximity                               #
#################################################################################
    # indicate pixels by their distance to the target area (pixels to close)
    # gets a geom object
    # proximity: list of buffer distances
    # for each distance a buffer around the target is created and marked. 
    # e.g. mark the distance from a natural habitat with 100m, 200m, 300m, etc.


#                     Second: evaluate by threshold                             #
#################################################################################    

    # indicate values of a file  above a certain threshold
    # gets a tif file file path
    # thresholds: cut offs.
    # for each threshold values <= that threshold are marked
    # elevation <= 10, <= 12 etc.

########################################################################################


# due to issues with numpy version
if not hasattr(np, "bool"):
    np.bool = bool





#3. evaluation function
def evaluate_area_by_proximity(Area_path, target_tif, where=None, output_dir=None, raster_target_value=None, evaluation_name= None):
    """
    This function extracts all pixels in “target_tif” that match “raster_target_value,”
    generates buffer zones according to the values defined for “evaluation_name,” within "Area".
    

    Args:
        Area_path (shp path | tif path ): 
            The area boundary used for clipping the analysis. Can be a path to raster or a shapefile

        where ( str | optional an SQL-style filtering string):
            Feature ID used to select a specific polygon from "Area_path" if it is a vector dataset
            Ignored if "Area_path" is a raster

        target_tif (tif | str):
            Raster file containing values to buffer around. Usually represents
            infrastructure, land use features, or environmental constraints

        output_dir (str):
            Path where the output files will be stored

        raster_target_value (list[int | float]):
            Pixel value in "target_tif" that will be used as the target for buffering

        evaluation_name (str):
            Key referring to the corresponding metadata entry in the EVALUATIONS dictionary. 
            Determines thresholds, unit, and description for the evaluation process.
        
    """
    # gets data from the EVALUATIONS DICT
    eval_def = EVALUATIONS[evaluation_name]
    unit = eval_def["unit"]
    description = eval_def["description"]
    source = eval_def["source"]
    thresholds = eval_def["thresholds"]



    # sets output dir
    output_dir = os.path.join(output_dir, evaluation_name)  #output path is beeing set
    os.makedirs(output_dir, exist_ok=True)

    # get target raster information
    info = gk.raster.rasterInfo(target_tif)

    srs_raster = info.srs
    pixel_res = abs(info.pixelWidth)

    print("Target Raster SRS:")
    print(srs_raster.GetAttrValue("AUTHORITY", 1))
    print("Target Raster Pixel Size:")
    print(info.pixelWidth)
    print(info.pixelHeight)




    # 1. load the target area
    suffix = os.path.splitext(Area_path)[1].lower()

    if suffix == ".shp":
        # Make Region Mask
        print("loading area shape file")
        reg = gk.RegionMask.fromVector(Area_path, 
                                 srs=srs_raster, #srs from target raster
                                 pixelRes=pixel_res,
                                 where=None,
                                 Limitone=False, 
                                 padExtent=max(thresholds)
                                 ) 

        print(f"Target Raster SRS {srs_raster} and pixe size {pixel_res} is set for area shapes region mask")
        #Area = shapefile path
        #where = attribute feature
        #padExtent = buffer


 

    elif suffix in (".tif", ".tiff"):
        print("loading area tif file")

        raster_1 = gk.raster.loadRaster(Area_path)
        # Raster-Informationen laden
        info = gk.raster.rasterInfo(Area_path)

        # EPSG-Code
        #print(info.srs.GetAttrValue("AUTHORITY", 1))
        # Pixelgröße
        #print(info.pixelWidth)
        #print(info.pixelHeight)

        raster_matrix = gk.raster.extractMatrix(raster_1)
        # boolean raster
        ras_boolean = gk.raster.createRasterLike(source=raster_1, data=raster_matrix == 1, noData=0)

        # create vector from raster and then egionmask from vector
        vector_1 = gk.raster.polygonizeRaster(ras_boolean)
        vector_2 = gk.vector.createVector(vector_1)
        reg = gk.RegionMask.fromVector(vector_2, 
                                    srs=srs_raster, 
                                    pixelRes=pixel_res,
                                    padExtent=max(thresholds), 
                                    Limitone=False,
                                    )
        print(f"Target Raster SRS {srs_raster} and pixe size {pixel_res} is set for area tif region mask")

    else:
        raise TypeError("Area path must be a string ending with .tiff/.tif/.shp.")

    # region mask Info
    srs_info = reg.srs.GetAttrValue("AUTHORITY", 1)
    print("Region masks srs:")
    print(f"region mask srs: EPSG:{srs_info}")
    print("Region masks extent:")
    print("reg extent:", reg.extent)
    print("Region masks shape:")
    print("reg shape:", reg.mask.shape)




    # 2. prepare the target for proximity analysis  (target tif file)
    print("setting region/feature of interest")
    matrix = reg.indicateValues(target_tif, #Indicates those pixels in the RegionMask 
                                                 # which correspond to a particular value, or range of values, from a given raster datasource
                                value=raster_target_value,  # values that are accepted 
                                applyMask=False) > 0.5 # > 0.5 means matrix, where values match raster values (=1) or match more than half (>0.5)
    # to polygon                                             
    target_area = gk.geom.polygonizeMatrix(matrix, bounds=reg.extent.xyXY, srs=reg.srs) #convert the array to a geom with mask extent
    

    
    # 3. safety check -> is there an overlap between areas
    # Raster-Infos
    info = gk.raster.rasterInfo(target_tif)

    print("=== TARGET RASTER ===")
    print("EPSG:", info.srs.GetAttrValue("AUTHORITY", 1))
    print("bounds:", info.bounds)
    print("pixel size:", info.pixelWidth, info.pixelHeight)

    print("\n=== REGION MASK ===")
    print("EPSG:", reg.srs.GetAttrValue("AUTHORITY", 1))
    print("extent:", reg.extent)
    print("shape:", reg.mask.shape)    

    r = info.bounds
    e = reg.extent

    overlap = not (
        e.xMax < r[0] or
        e.xMin > r[2] or
        e.yMax < r[1] or
        e.yMin > r[3]
    )

    print("Overlap:", overlap)



    # 4. create proximity matrix
    # region mask, gk_geom, distances from evaluation_values, utput dir 
    result = edgesByProximity(reg, target_area, thresholds, output_dir)  

    # 5. make result
    writeEdgeFile(
        result, reg, output_dir, evaluation_name, unit, description, source, thresholds
    )
    
def evaluate_area_by_threshold(Area, #area to analyze
                              tif_file, #tif file to analyze
                              where=None, #set the polygon or cell values that define th area
                              output_dir=None,
                              evaluation_name= None,
                              ):
    """This function marks cells which are below a certain threshold. Base for the threshold values is the "tif_file"
    within the "Area".

    Args:
        Area (shp path | tif path ): 
            The area boundary used for clipping the analysis. Can be a path to raster or a shapefile

        where ( str | optional an SQL-style filtering string):
            Feature ID used to select a specific polygon from "Area" if it is a vector dataset
            Ignored if "Area" is a raster

        target_tif (tif | str):
            Raster file containing values to which the trechhold is applied. Usually represents
            infrastructure, land use features, or environmental constraints

        raster_target_value (list[int | float]):
            Pixel value in "target_tif" that will be used as the target for threshold application

        output_dir (str):
            Path where the output files will be stored

        evaluation_name (str):
            Key referring to the corresponding metadata entry in the EVALUATIONS dictionary. 
            Determines thresholds, unit, and description for the evaluation process.

    """
    
    # gets data from the EVALUATIONS DICT
    eval_def = EVALUATIONS[evaluation_name]
    unit = eval_def["unit"]
    description = eval_def["description"]
    source = eval_def["source"]
    thresholds = eval_def["thresholds"]
    
    # output dir     
    output_dir = os.path.join(output_dir, evaluation_name)  #output path is beeing set
    os.makedirs(output_dir, exist_ok=True)

    # get target raster information
    info = gk.raster.rasterInfo(target_tif)

    srs_raster = info.srs
    pixel_res = abs(info.pixelWidth)

    print("Target Raster SRS:")
    print(srs_raster.GetAttrValue("AUTHORITY", 1))
    print("Target Raster Pixel Size:")
    print(info.pixelWidth)
    print(info.pixelHeight)


    # 1. set the area
    suffix = os.path.splitext(Area)[1].lower()

    if suffix == ".shp":
        # Make Region Mask
        print("loading area shape file")
        reg = gk.RegionMask.load(region=Area, where=where, padExtent=500) 
        #Area = shapefile path
        #where = attribute feature
        #padExtent = buffer
        
    elif suffix in (".tif", ".tiff"):
        print("loading area tif file")
        # load raster
        raster_1 = gk.raster.loadRaster(Area)
        raster_matrix = gk.raster.extractMatrix(raster_1)
        # boolean raster
        ras_boolean = gk.raster.createRasterLike(source=raster_1, data=raster_matrix == 1, noData=0)
        
        # create vector from raster and then egionmask from vector
        vector_1 = gk.raster.polygonizeRaster(ras_boolean)
        vector_2 = gk.vector.createVector(vector_1)
        reg = gk.RegionMask.fromVector(vector_2, limitOne=False, padExtent=500)

    else:
        raise TypeError("Area must be a string ending with .tiff/.tif/.shp.")

    # region mask Info
    srs_info = reg.srs.GetAttrValue("AUTHORITY", 1)
    print("Region masks srs:")
    print(f"region mask srs: EPSG:{srs_info}")
    print("Region masks extent:")
    print("reg extent:", reg.extent)
    print("Region masks shape:")
    print("reg shape:", reg.mask.shape)



    # 2. Calculate the threshold
    result = edgesByThreshold(reg, tif_file, thresholds, output_dir)

    # 3. safety check -> is there an overlap between areas
    # Raster-Infos
    info = gk.raster.rasterInfo(target_tif)

    print("=== TARGET RASTER ===")
    print("EPSG:", info.srs.GetAttrValue("AUTHORITY", 1))
    print("bounds:", info.bounds)
    print("pixel size:", info.pixelWidth, info.pixelHeight)

    print("\n=== REGION MASK ===")
    print("EPSG:", reg.srs.GetAttrValue("AUTHORITY", 1))
    print("extent:", reg.extent)
    print("shape:", reg.mask.shape)    

    r = info.bounds
    e = reg.extent

    overlap = not (
        e.xMax < r[0] or
        e.xMin > r[2] or
        e.yMax < r[1] or
        e.yMin > r[3]
    )

    print("Overlap:", overlap)

    # 4. make result
    writeEdgeFile(
        result,
        reg,
        output_dir,
        evaluation_name,
        unit,
        description,
        source,
        thresholds,
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

def edgesByProximity(reg, target_area, thresholds, output_dir):
    """ 
    within the regionmask ("Area") the target_area (ftrID from raster) is buffered with the threhold values (set in the EVALUATIONS dict). 
    Each distance/ buffer zone area gets a unique value. Output is a matrix

    Args:
        reg (Regionmask): Regionmask with spatial extent for further processing
        target_area (geom): 
        thresholds (List[float or int]): distance dict (List[float or int]): List of increasing distance values from Evaluation_values dict that are used for buffering

    Returns:
        np.ndarray: A labeled matrix where each cell indicates its proximity zone (0 = closest)
    """

    # make initial matrix with the extent of reg ("Area")
    proximity_matrix = (np.ones(reg.mask.shape, dtype=np.uint8) * 255)  # Set all values to no data (255)
    proximity_matrix[reg.mask] = 254  # Set all values in the target region to untouched (254)

    # Only do buffering if a geometry is available
    if target_area is not None and len(target_area) != 0:

        # Do distance calculation
        for dist in thresholds:
            print(f"calculating proximity for {dist} around the feature of interest in the region of interest")  
            #loops over the distance list
            value = dist 
            buffered = calculate_distances(target_area, dist) #buffers the geom by distance
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
            
    return proximity_matrix # Return the final labeled matrix

def edgesByThreshold(reg, source, thresholds, output_dir):
    """
    Thresholds of a raster file are beeing marked  as a matrix

    Args:
        reg (Regionmask): Regionmask with spatial extent for further processing
        source (tif file): Path to a rasterfile
        thresholds (List[float or int]): List of increasing threshold values from Evaluation_values dict 
    
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
            
        output_dir_plots = os.path.join(output_dir, "intermediates")
        os.makedirs(output_dir_plots, exist_ok=True)
        filename = os.path.join(output_dir_plots, f"threshold_region_of_interest_{thresh}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        # apply onto matrix: select pixels that are valid (254) and satisfy thresholf (indicated) (at the same time)
        selected_matrix = np.logical_and(threshold_matrix == 254, indicated)  
        
        # Assign the current class label to the selected pixels
        threshold_matrix[selected_matrix] = value
        
        #label for the next threshold


    return threshold_matrix # Return the final labeled matrix

def writeEdgeFile(
    result, reg, output_dir, evaluation_name, unit, description, source, values,
):
    """
    Writes a  result matrix to a GeoTIFF file with metadata and value labeling

    Args:
        result (np.ndarray): A 2D  matrix (e.g., from thresholding or proximity analysis)
        reg (RegionMask): A RegionMask object used to define spatial extent and raster creation
        output_dir (str): path where output raster will be saved
        name (str): Base name for the output 
        unit (str): Unit of the  values (e.g., meters, degrees)
        description (str): explanation of the dataset
        source (str): Source dataset or method used to generate the result
        values (List[float]): List of thresholds or distance values used in classification

    Returns:
        None: The function writes the raster file to disk; it does not return a value
    """
    # make output

    output = f"{evaluation_name}.tif"
    os.makedirs(output_dir, exist_ok=True)
        
    #Meta data mapping
    valueMap = OrderedDict()        #proximity values and their meaning is written to the meta
    for i in range(len(values)):
        valueMap["%d" % i] = "<=%.2f" % values[i]
    valueMap["254"] = "untouched"
    valueMap["255"] = "noData"

    meta = OrderedDict()
    meta["GLAES_PRIOR"] = "YES"
    meta["DISPLAY_NAME"] = evaluation_name
    meta["ALTERNATE_NAME"] = "NONE"
    meta["DESCRIPTION"] = description
    meta["UNIT"] = unit
    meta["SOURCE"] = source
    meta["VALUE_MAP"] = dumps(valueMap)
    
    print(f"saving proximity output {output}")
    
    d = reg.createRaster(
        output=os.path.join(output_dir, output),
        #output=output_dir,
        data=result,
        overwrite=True,
        noDataValue=255,
        dtype=1,
        meta=meta,
    )

########################################################################################


#   1. define input files
base_path = Path(__file__).resolve().parents[0]

Area_path = str(base_path/"input/aachenShapefile.shp")
Area_path_tif = str(base_path/"input/aachen_srs_3035.tif")
target_tif =str(base_path/"input/roads_prior_clip.tif")




#2. evaluation values
# Indicates distances too close to exclusion criterion:
# These distances are used for 
#   a. buffering around the selected feature of interest/ values of interest.
#   b. thresholds for cell / area identification
# values should match the input extent (if in 100 m resolution distances should not be less (e.g. 10, 20,30,etc.))
# values refer to the unit used in the srs (meter or degree)

EVALUATIONS = {
    # proximity
    "agriculture_proximity": {
        "unit": "m",
        "description": "Indicates distances too close to agriculture areas (m)",
        "source": "",
        "thresholds": [
            0, 100, 
            #200, 300, 400, 500
        ], # example values
    },
    # threshold
    "dni_threshold": {
        "unit": "kWh/m2/day",
        "description": "Indicates pixels in which the average daily direct-normal irradiance (DNI) is less-than or equal-to X kWh/m2/day",
        "source": "",
        "thresholds": [
            0, 5, 10, 20, 50, 100, 200, 300, 400,
            ],  # example values
    },
}

## TESTING

# Area_path = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/glaes_2026/glaes/deletme_create_prior/input/aachenShapefile.shp"
# Area_path_tif = "/fast/home/l-madeisky/models_IEK_3/IEK3_Models/glaes_2026/glaes/deletme_create_prior/input/aachen_srs_3035.tif"
# target_tif ="/fast/home/l-madeisky/models_IEK_3/IEK3_Models/glaes_2026/glaes/deletme_create_prior/input/roads_prior_clip.tif"

#with shape
# evaluate_area_by_proximity(          
#     Area_path=Area_path,
#     evaluation_name= "agriculture_proximity",
#     where=None,
#     target_tif=target_tif,
#     raster_target_value=[3], 
#     output_dir = str(base_path/"output/proximity/with_shape/")
# )


#with tif
# evaluate_area_by_proximity(          
#     Area_path=Area_path_tif,
#     evaluation_name= "agriculture_proximity",
#     where=None,
#     target_tif=target_tif,
#     raster_target_value=[3], 
#     output_dir = str(base_path/"output/proximity/with_tif/")
# )


# #wtih shape
evaluate_area_by_threshold(           
    Area=Area_path,
    evaluation_name="dni_threshold",
    where=None,
    tif_file=target_tif,
    output_dir = str(base_path/"output/threshold/with_shape/")
)

# # with tif
# evaluate_area_by_threshold(           
#     Area=Area_path_tif,
#     evaluation_name="dni_threshold",
#     where=None,
#     tif_file=target_tif,
#     output_dir = str(base_path/"output/threshold/with_tif/")
# )