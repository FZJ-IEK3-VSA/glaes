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

#################################################################
## DEFINE SOURCES
clcSource="/home/s.ryberg/data/zena_data/CLC/g100_clc12_V18_5_SRS_FIX.tif"
waterbodySource="/home/s.ryberg/data/COPERNICUS/20/l6_pwb_eur_20m_full01_100_fin06.tif"
urbanClustersSource="/home/s.ryberg/data/EUROSTAT/Urban_Clusters/URB_CLST_2011.tif"
airportsSource = "/home/s.ryberg/data/EUROSTAT/Airports/AIRP_PT_2013.shp"
osmRailwaysSource = "/home/s.ryberg/data/OSM/geofabrik/railways/","*gis.osm_railways*.shp"
osmRoadsSource = "/home/s.ryberg/data/OSM/geofabrik/roads/","*gis.osm_roads*.shp"
osmPowerlinesSource = "/home/s.ryberg/data/OSM/osm2shp/power_ln/power_ln_europe_clip.shp"
riverSegmentsSource = "/home/s.ryberg/data/EUROSTAT/rivers_and_catchments/data/","*Riversegments.shp"
hydroLakesSource = "/home/s.ryberg/data/WWF/HydroLAKES_polys_v10.shp"
wdpaSource = "/home/s.ryberg/data/protected/WDPA/WDPA_Apr2017-shapefile/clipped","*.shp"
demSource = "/home/s.ryberg/data/zena_data/DEM/eudem_dem_4258_europe.tif"
gwaSource = "/home/s.ryberg/data/global_wind_atlas/WS_%03dm_global_wgs84_mean_trimmed.tif"
dniSource = "/home/s.ryberg/data/global_solar_atlas/World_DNI_GISdata_LTAy_DailySum_GlobalSolarAtlas_GEOTIFF/DNI.tif"
ghiSource = "/home/s.ryberg/data/global_solar_atlas/World_GHI_GISdata_LTAy_DailySum_GlobalSolarAtlas_GEOTIFF/GHI.tif"

##################################################################
## DEFINE EDGES
EVALUATION_VALUES = { 
    "woodland_mixed_proximity":
        # Indicates distances too close to mixed-tree forests (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "woodland_coniferous_proximity":
         # Indicates distances too close to coniferous forests (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "woodland_deciduous_proximity":
        # Indicates distances too close to deciduous forests(m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "waterbody_proximity":
        # Indicates distances too close to lakes (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "lake_proximity":
        # Indicates distances too close to lakes (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "river_proximity":
        # Indicates distances too close to rivers (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "ocean_proximity":
        # Indicates distances too close to oceans (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000, 10000, 15000, 20000],

    "wetland_proximity":
        # Indicates distances too close to wetlands (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "elevation_threshold":
        # Indicates elevations above X (m)
        np.linspace(-1000, 3000, 41), 

    "slope_threshold":
        # Indicates slopes above X (degree)
        np.linspace(0, 30, 61), 

    "slope_north_facing_threshold":
        # Indicates north-facing slopes above X (degree)
        np.linspace(-20, 20, 81),

    "power_line_proximity":
        # Indicates distances too close to power-lines (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000,
         6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],

    "roads_main_proximity":
        # Indicates distances too close to main roads (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 
         3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],

    "roads_secondary_proximity":
        # Indicates distances too close to secondary roads (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 
         3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],

    "roads_proximity":
        # Indicates distances too close to all accissible roads (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2250, 2500, 2750, 3000, 
         3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],

    "railway_proximity":
        # Indicates distances too close to railways (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000,
         6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000],

    "settlement_urban_proximity":
        # Indicates distances too close to dense settlements (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
         2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000, 20000],

    "settlement_proximity":
        # Indicates distances too close to light settlements (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
         2200, 2400, 2600, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000, 20000],

    "industrial_proximity":
        # Indicates distances too close to industrial areas (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "mining_proximity":
        # Indicates distances too close to mines (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "agriculture_proximity":
         # Indicates distances too close to aggriculture areas (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "airport_proximity":
        # Indicates distances too close to airports (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 
         4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000],

    "airfield_proximity":
        # Indicates distances too close to airfields (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 
         4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 15000],

    "protected_park_proximity":
        # Indicates distances too close to protected parks (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_landscape_proximity":
        # Indicates distances too close to protected landscapes (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_natural_monument_proximity":
        # Indicates distances too close to protected natural-monuments (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_reserve_proximity":
        # Indicates distances too close to protected reserves (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_wilderness_proximity":
        # Indicates distances too close to protected wilderness (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_biosphere_proximity":
        # Indicates distances too close to protected biospheres (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_habitat_proximity":
        # Indicates distances too close to protected habitats (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "protected_bird_proximity":
        # Indicates distances too close to protected bird areas (m)
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000],

    "windspeed_50m_threshold":
        # Indicates areas with average wind speed below X (m/s)
        np.linspace(0, 20, 81),

    "windspeed_100m_threshold":
        # Indicates areas with average wind speed below X (m/s)
        np.linspace(0, 20, 81),

    "ghi_threshold":
        # Indicates areas with average total daily irradiance below X (kWh/m2/day)
        np.linspace(0, 20, 81), 

    "dni_threshold":
        # Indicates areas with average total daily irradiance below X (kWh/m2/day)
        np.linspace(0, 20, 81), 
    
    #"connection_distance": # Indicates distances too far from power grid (m)
    #   [50000, 45000, 40000, 35000, 30000, 25000, 20000, 18000, 16000,
    #    14000, 12000, 11000, 10000,  9000,  8000,  7000,  6000,  5000,
    #     4000,  3500,  3000,  2500,  2000,  1500,  1250,  1000,   750,
    #      500,   250,     0], 

    #"access_distance": # Indicates distances too far from roads (m)
    #   [20000, 19000, 18000, 17000, 16000, 15000, 14000, 13000, 12000,
    #    11000, 10000,  9000,  8000,  7000,  6000,  5000,  4000,  3500,
    #     3000,  2500,  2000,  1500,  1250,  1000,   750,   500,   250,     0], 
    
      }

#######################################################
## EVALUATION FUNCTIONS
def evaluate_OCEAN(regSource, ftrID, tail):
    name = "ocean_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an ocean"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=44, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WETLAND(regSource, ftrID, tail):
    name = "wetland_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a wetland area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=(35,39), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_INDUSTRIAL(regSource, ftrID, tail):
    name = "industrial_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an industrial area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=3, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_MINING(regSource, ftrID, tail):
    name = "mining_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a mining area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=7, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

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
    matrix = reg.indicateValues(clcSource, value=(12,22), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_AG_ARABLE(regSource, ftrID, tail):
    name = "agriculture_arable_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an arable agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES["agriculture_proximity"]
    
    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=(12,14), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_AG_PERMANENT(regSource, ftrID, tail):
    name = "agriculture_permanent_crop_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a permanent-crop agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES["agriculture_proximity"]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=(15,17), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_AG_PASTURE(regSource, ftrID, tail):
    name = "agriculture_pasture_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a pastural agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES["agriculture_proximity"]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=18, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_AG_HETEROGENEOUS(regSource, ftrID, tail):
    name = "agriculture_heterogeneous_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a heterogeneous agriculture area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES["agriculture_proximity"]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=(19,22), applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WOODLANDS_MIXED(regSource, ftrID, tail):
    name = "woodland_mixed_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a mixed-tree woodland area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=23, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WOODLANDS_CONIFEROUS(regSource, ftrID, tail):
    name = "woodland_coniferous_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a predominantly coniferous (needle leaved) woodland area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=24, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WOODLANDS_DECIDUOUS(regSource, ftrID, tail):
    name = "woodland_deciduous_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a predominantly deciduous (broad leaved) woodland area"
    source = "CLC12"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(clcSource, value=25, applyMask=False) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_ROADS(regSource, ftrID, tail):
    name = "roads_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from all accessible roadways"
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm filesi
    whereStmt =  r"fclass LIKE '%motorway%' OR fclass LIKE '%trunk%' OR fclass LIKE '%primary%' OR fclass LIKE '%secondary%' OR fclass LIKE '%tertiary%' OR fclass = 'service' OR fclass = 'unclassified'"
    geom = geomExtractor( reg.extent, osmRoadsSource, whereStmt) 

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_ROADS_MAIN(regSource, ftrID, tail):
    name = "roads_main_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from major roadways"
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, osmRoadsSource, r"fclass LIKE '%motorway%' OR fclass LIKE '%trunk%' OR fclass LIKE '%primary%'") 

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_ROADS_SECONDARY(regSource, ftrID, tail):
    name = "roads_secondary_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from secondary roadways"
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, osmRoadsSource, r"fclass LIKE '%secondary%' OR fclass LIKE '%tertiary%'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_POWER_LINE(regSource, ftrID, tail):
    name = "power_line_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a power line"
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, osmPowerlinesSource, r"power='line'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_RAILWAY(regSource, ftrID, tail):
    name = "railway_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a railway"
    source = "OSM"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, osmRailwaysSource, r"fclass = 'rail'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WATERBODY(regSource, ftrID, tail):
    name = "waterbody_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from permanent water bodies"
    source = "COPERNICUS HRL"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Indicate values and create a geomoetry from the result
    matrix = reg.indicateValues(waterbodySource, value=1, applyMask=False, resolutionDiv=5) > 0.5
    geom = gk.geom.convertMask(matrix, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_RIVER(regSource, ftrID, tail):
    name = "river_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from rivers and other running water bodies"
    source = "WWF hydroBASINS"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, riverSegmentsSource)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_LAKE(regSource, ftrID, tail):
    name = "lake_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from lakes and other stagnant water bodies"
    source = "WWF HydroLAKES"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, hydroLakesSource)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_PARK(regSource, ftrID, tail):
    name = "protected_park_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected park"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%park%' OR IUCN_CAT = 'II'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_LANDSCAPE(regSource, ftrID, tail):
    name = "protected_landscape_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected landscape"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%landscape%' OR IUCN_CAT = 'V'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_MONUMENT(regSource, ftrID, tail):
    name = "protected_natural_monument_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected natural monument"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%monument%' OR IUCN_CAT = 'III'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_RESERVE(regSource, ftrID, tail):
    name = "protected_reserve_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected reserve"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%reserve%' OR IUCN_CAT = 'Ia'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WILDERNESS(regSource, ftrID, tail):
    name = "protected_wilderness_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected wilderness"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    #geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%wilderness%' OR IUCN_CAT = 'Ib'", simplify=reg.pixelSize/5)
        
    matrix = None

    for f in reg.extent.filterSources( join(wdpaSource[0], wdpaSource[1])):
        tmp = reg.indicateFeatures(f, where=r"DESIG_ENG LIKE '%wilderness%' OR IUCN_CAT = 'Ib'", resolutionDiv=5, applyMask=False ) > 0.5
        
        if matrix is None: matrix = tmp
        else: np.logical_or(tmp, matrix, matrix)
    
    if matrix.any():
        geom = gk.geom.convertMask(matrix, reg.extent.xyXY, reg.srs)
    else: 
        geom = None

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_BIOSPHERE(regSource, ftrID, tail):
    name = "protected_biosphere_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected biosphere"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%bio%'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_HABITAT(regSource, ftrID, tail):
    name = "protected_habitat_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected habitat"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    #geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%habitat%' OR IUCN_CAT = 'IV'", simplify=reg.pixelSize/5)

    matrix = None

    for f in reg.extent.filterSources( join(wdpaSource[0], wdpaSource[1])):
        tmp = reg.indicateFeatures(f, where=r"DESIG_ENG LIKE '%habitat%' OR IUCN_CAT = 'IV'", resolutionDiv=5, applyMask=False ) > 0.5
        
        if matrix is None: matrix = tmp
        else: np.logical_or(tmp, matrix, matrix)
    
    if matrix.any():
        geom = gk.geom.convertMask(matrix, reg.extent.xyXY, reg.srs)
    else: 
        geom = None

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_BIRDS(regSource, ftrID, tail):
    name = "protected_bird_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from a protected bird area"
    source = "WDPA"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%bird%'")

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_URBAN(regSource, ftrID, tail):
    name = "settlement_urban_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from dense-urban and city settlements"
    source = "EUROSTAT"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    indicated = reg.indicateValues(urbanClustersSource, value=(5000, 2e7), applyMask=False) > 0.5
    geom = gk.geom.convertMask(indicated, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_SETTLEMENT(regSource, ftrID, tail):
    name = "settlement_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from any settlement area"
    source = "CLC"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    #eurostatUrban = reg.indicateValues(urbanClustersSource, value=(5000, 2e7), applyMask=False) > 0.5
    clcUrban = reg.indicateValues(clcSource, value=(1,2), applyMask=False) > 0.5

    geom = gk.geom.convertMask( clcUrban, bounds=reg.extent.xyXY, srs=reg.srs)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_AIRPORT(regSource, ftrID, tail):

    ######################
    ## Evaluate airports
    name = "airport_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an airport"
    source = "EUROSTAT, CLC"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=max(distances)*1.25)

    ### Get airport regions
    airportMask = reg.indicateValues(clcSource, value=6, applyMask=False) > 0.5
    airportGeoms = gk.geom.convertMask(airportMask, bounds=reg.extent.xyXY, srs=reg.srs)
    if airportGeoms is None: airportGeoms = []
    
    ### define an airport/airfield shape matcher
    def airportShapes( points, minSize, defaultRadius, minDistance=2000 ):
        locatedGeoms = []

        # look for best geometry for each airport
        for pt in points:
            found = False

            # First look for containing geometries greater than the minimal area
            containingGeoms = filter(lambda x: x.Contains(pt), airportGeoms)
            for geom in containingGeoms:
                if geom.Area() > minSize:
                    locatedGeoms.append( geom.Clone() )
                    found = True
                if found: continue
            if found: continue

            # Next look for nearby geometries greater than the minimal area
            nearbyGeoms = filter(lambda x: pt.Distance(x) <= minDistance, airportGeoms)
            for geom in nearbyGeoms:
                if geom.Area() > minSize:
                    locatedGeoms.append( geom.Clone() )
                    found = True
                if found: continue
            if found: continue

            # if all else fails, apply a default distance
            locatedGeoms.append( pt.Buffer(defaultRadius) )

        if len(locatedGeoms)==0: return None
        else: return locatedGeoms

    ### Locate airports
    airportWhere = "AIRP_USE!=4 AND (AIRP_PASS=1 OR AIRP_PASS=2) AND AIRP_LAND='A'"
    airportCoords = [point.Clone() for point,i in gk.vector.extractFeatures(airportsSource, reg.extent.box, where=airportWhere)]
    for pt in airportCoords: pt.TransformTo(reg.srs)

    geom = airportShapes(airportCoords, minSize=1e6, defaultRadius=3000)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

    #####################
    ## Evaluate airfields
    name = "airfield_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from an airfield"
    source = "EUROSTAT, CLC"

    output_dir = join("outputs", name)

    # Get distances
    distances = EVALUATION_VALUES[name]

    ### Locate airports
    airfieldWhere = "AIRP_USE!=4 AND (AIRP_PASS=0 OR AIRP_PASS=9) AND AIRP_LAND='A'"
    airfieldCoords = [point.Clone() for point,i in gk.vector.extractFeatures(airportsSource, reg.extent.box, where=airfieldWhere)]
    for pt in airfieldCoords: pt.TransformTo(reg.srs)

    geom = airportShapes(airfieldCoords, minSize=1e6, defaultRadius=800)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)

def evaluate_WINDSPEED50(regSource, ftrID, tail):
    name = "windspeed_50m_threshold"
    unit = "m/s"
    description = "Indicates pixels in which the average windspeed (measured at 50m) is less-than or equal-to X m/s"
    source = "Global Wind Atlas"

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    result = edgesByThreshold(reg, gwaSource%50, thresholds)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_WINDSPEED100(regSource, ftrID, tail):
    name = "windspeed_100m_threshold"
    unit = "m/s"
    description = "Indicates pixels in which the average windspeed (measured at 100m) is less-than or equal-to X m/s"
    source = "Global Wind Atlas"

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    result = edgesByThreshold(reg, gwaSource%100, thresholds)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_GHI(regSource, ftrID, tail):
    name = "ghi_threshold"
    unit = "kWh/m2/day"
    description = "Indicates pixels in which the average daily global-horizontal irrandiance (GHI) is less-than or equal-to X kWh/m2/day"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    result = edgesByThreshold(reg, ghiSource, thresholds)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_DNI(regSource, ftrID, tail):
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
    result = edgesByThreshold(reg, dniSource, thresholds)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_ELEVATION(regSource, ftrID, tail):
    name = "elevation_threshold"
    unit = "meters"
    description = "Indicates pixels in which the average elevation is less-than or equal-to X meters"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = EVALUATION_VALUES[name]

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    result = edgesByThreshold(reg, demSource, thresholds)

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_SLOPE(regSource, ftrID, tail):
    name = "slope_threshold"
    unit = "degrees"
    description = "Indicates pixels in which the average slope is less-than or equal-to X degrees"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = np.array(EVALUATION_VALUES[name])

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    demSourceClipped = reg.extent.clipRaster(demSource)
    gradientDS = gk.raster.gradient(demSourceClipped, mode="slope", factor="latlonToM")

    result = edgesByThreshold(reg, gradientDS, np.tan(thresholds*np.pi/180))

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

def evaluate_SLOPE_NORTH(regSource, ftrID, tail):
    name = "slope_north_facing_threshold"
    unit = "degrees"
    description = "Indicates pixels in which the average slope in the northern direction is less-than or equal-to X degrees"
    source = ""

    output_dir = join("outputs", name)

    # Get distances
    thresholds = np.array(EVALUATION_VALUES[name])

    # Make Region Mask
    reg = gk.RegionMask.load(regSource, select=ftrID, padExtent=500)

    # Create a geometry list from the osm files
    demSourceClipped = reg.extent.clipRaster(demSource)
    gradientDS = gk.raster.gradient(demSourceClipped, mode="north-south", factor="latlonToM")

    result = edgesByThreshold(reg, gradientDS, np.tan(thresholds*np.pi/180))

    # make result
    writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, thresholds)

##################################################################
## UTILITY FUNCTIONS
def edgesByProximity(reg, geom, distances):
    
    # make initial matrix
    mat = np.ones(reg.mask.shape, dtype=np.uint8)*255 # Set all values to no data (255)
    mat[reg.mask] = 254 # Set all values in the region to untouched (254)
    
    # Only do growing if a geometry is available
    if not geom is None and len(geom)!=0:
        # make grow func
        def doGrow(geom, dist):
            if dist > 0:
                if isinstance(geom,list) or isinstance(geom, filter):
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
                tmpSource = gk.vector.createVector(grown) # Make a temporary vector file
            except Exception as e:
                print(len(grown), [g.GetGeometryName() for g in grown])
                raise e
            
            indicated = reg.indicateFeatures(tmpSource) > 0.5 # Map onto the RegionMask
            
            # apply onto matrix
            sel = np.logical_and(mat==254, indicated) # write onto pixels which are indicated and available
            mat[sel] = value
            value += 1

    # Done!
    return mat

def edgesByThreshold(reg, source, thresholds):
    # make initial matrix
    mat = np.ones(reg.mask.shape, dtype=np.uint8)*255 # Set all values to no data (255)
    mat[reg.mask] = 254 # Set all values in the region to untouched (254)
    
    # Only do growing if a geometry is available
    value = 0
    for thresh in thresholds:
        indicated = reg.indicateValues(source, value=(None,thresh)) > 0.5

        # apply onto matrix
        sel = np.logical_and(mat==254, indicated) # write onto pixels which are indicated and available
        mat[sel] = value
        value += 1

    # Done!
    return mat

def writeEdgeFile( result, reg, ftrID, output_dir, name, tail, unit, description, source, values):
    # make output
    output = "%s.%s_%05d.tif"%(name,tail,ftrID)
    if not isdir(output_dir): mkdir(output_dir)

    valueMap = OrderedDict()
    for i in range(len(values)): valueMap["%d"%i]="<=%.2f"%values[i]
    valueMap["254"]="untouched"
    valueMap["255"]="noData"

    meta = OrderedDict()
    meta["GLAES_PRIOR"] = "YES"
    meta["DISPLAY_NAME"] = name
    meta["ALTERNATE_NAME"] = "NONE"
    meta["DESCRIPTION"] = description
    meta["UNIT"] = unit
    meta["SOURCE"] = source
    meta["VALUE_MAP"] = dumps(valueMap)

    print(output)

    d = reg.createRaster(output=join(output_dir,output), data=result, overwrite=True, noDataValue=255, dtype=1, meta=meta)

def geomExtractor( extent, source, where=None, simplify=None ): 
    searchGeom = extent.box
    if isinstance(source,str):
        searchFiles = [source,]
    else:
        searchFiles = list(extent.filterSources( join(source[0], source[1]) ))
    
    geoms = []
    for f in searchFiles:
        for geom, attr in gk.vector.extractFeatures(f, searchGeom, where=where, outputSRS=extent.srs):
            geoms.append( geom.Clone() )

    if not simplify is None: 
        newGeoms = [g.SimplifyPreserveTopology(simplify) for g in geoms]
        for g, ng in zip(geoms, newGeoms):
            if "LINE" in ng.GetGeometryName():
                test = ng.Length()/g.Length()
            else: 
                test = ng.Area()/g.Area()

            if test<0.97:
                raise RuntimeError("ERROR: Simplified geometry is >3% different from the original")
            elif test<0.99:
                print("WARNING: simplified geometry is slightly different from the original")

    
    if len(geoms) == 0:
        return None
    else:
        return geoms

###################################################################
## MAIN FUNCTIONALITY
if __name__== '__main__':
    START= dt.now()
    tail = str(int(dt.now().timestamp()))
    print( "RUN ID: ", tail)
    print( "TIME START: ", START)

    # Choose the function
    func = globals()["evaluate_"+sys.argv[1]]

    # Choose the source
    if len(sys.argv)<3:
        source = join("reg","aachenShapefile.shp")
    else:
        source = sys.argv[2]

    # Arange workers
    if len(sys.argv)<4:
        doMulti = False
    else:
        doMulti = True
        pool = Pool(int(sys.argv[3]))
    
    # submit jobs
    res = []
    count = -1
    for g,a in gk.vector.extractFeatures(source):
        count += 1
        #if count<1 : continue
        #if count == 2:break

        # Do the analysis
        if doMulti:
            res.append(pool.apply_async(func, (source, count, tail)))
        else:
            func(source, count, tail)
    
    if doMulti:
        # Check for errors
        for r,i in zip(res,range(len(res))):
            try:
                r.get()
            except Exception as e:
                print("EXCEPTION AT ID: "+str(i))
                raise e

        # Wait for jobs to finish
        pool.close()
        pool.join()

    # finished!
    END= dt.now()
    print( "TIME END: ", END)
    print( "CALC TIME: ", (END-START))

