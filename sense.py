import glaes as gl
import numpy as np
from os.path import join, isdir, isfile, basename, splitext
from os import mkdir
import sys
from multiprocessing import Pool
import time
from datetime import datetime as dt
from glob import glob
from collections import namedtuple

##################################################################
## STATIC VARS
outputDir = "outputs"
conRange = namedtuple('conRange',["low","median","high","steps"])

ranges = {
         "woodlands-mixed": conRange(0, 300, 1000, 10), # Indicates distances too close to mixed-tree forests (m)
    "woodlands-coniferous": conRange(0, 300, 1000, 10), # Indicates distances too close to coniferous forests (m)
     "woodlands-deciduous": conRange(0, 300, 1000, 10), # Indicates distances too close to deciduous forests(m)
                   "lakes": conRange(0, 300, 3000, 20), # Indicates distances too close to lakes (m)
                  "rivers": conRange(0, 400, 3000, 15), # Indicates distances too close to rivers (m)
                  "oceans": conRange(0, 300, 2000, 20), # Indicates distances too close to oceans (m)
                "wetlands": conRange(0, 200, 2000, 20), # Indicates distances too close to wetlands (m)
               "elevation": conRange(2000, 1750, 1500, 60), # Indicates elevations above X (m)
                   "slope": conRange(30, 11, 0, 60), # Indicates slopes above X (degree)
                  "nslope": conRange(15, 3, -15, 60), # Indicates north-facing slopes above X (degree)
                  "aspect": conRange(180, 0, -180, 60), # Indicates aspects in given range (degrees)
             "power-lines": conRange(0, 150, 500, 10), # Indicates distances too close to power-lines (m)
              "roads-main": conRange(0, 200, 500, 10), # Indicates distances too close to main roads (m)
         "roads-secondary": conRange(0, 100, 500, 10), # Indicates distances too close to secondary roads (m)
                "railways": conRange(0, 200, 500, 10), # Indicates distances too close to railways (m)
                   "urban": conRange(0, 1500, 3000, 30), # Indicates distances too close to dense settlements (m)
                   "rural": conRange(0,  750, 2000, 20), # Indicates distances too close to light settlements (m)
              "industrial": conRange(0, 300, 1000, 10), # Indicates distances too close to industrial areas (m)
                   "mines": conRange(0, 200, 1000, 10), # Indicates distances too close to mines (m)
             "agriculture": conRange(0, 100, 250, 5), # Indicates distances too close to aggriculture areas (m)
                "airports": conRange(0, 4000, 10000, 20), # Indicates distances too close to airports (m)
               "airfields": conRange(0, 3000,  5000, 10), # Indicates distances too close to airfields (m)
                   "parks": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected parks (m)
              "landscapes": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected landscapes (m)
       "natural-monuments": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected natural-monuments (m)
                "reserves": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected reserves (m)
              "wilderness": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected wilderness (m)
              "biospheres": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected biospheres (m)
                "habitats": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected habitats (m)
                   "birds": conRange(0, 1000, 3000, 15), # Indicates distances too close to protected bird areas (m)
           "resource-wind": conRange(0, 4, 10, 20), # Indicates areas with average wind speed below X (m/s)
            "resource-ghi": conRange(0, 200, 400, 20), # Indicates areas with average total daily irradiance below X (kWh/m2/day)
            "resource-dni": conRange(0, 200, 400, 20), # Indicates areas with average total daily irradiance below X (kWh/m2/day)
         "grid-connection": conRange(50000, 10000, 0, 25), # Indicates distances too far from power grid (m)
             "road-access": conRange(20000, 5000, 0, 20), # Indicates distances too far from roads (m)
    }

clcSource="/home/s.ryberg/data/zena_data/CLC/g100_clc12_V18_5_SRS_FIX.tif"
urbanClustersSource="/home/s.ryberg/data/EUROSTAT/Urban_Clusters/URB_CLST_2011.tif"
airportsSource = "/home/s.ryberg/data/EUROSTAT/Airports/AIRP_PT_2013.shp"
osmRailwaysSource = "/home/s.ryberg/data/OSM/geofabrik/railways/","*gis.osm_railways*.shp"
osmRoadsSource = "/home/s.ryberg/data/OSM/geofabrik/roads/","*gis.osm_roads*.shp"
osmPowerlinesSource = "/home/s.ryberg/data/OSM/osm2shp/power_ln/power_ln_europe_clip.shp"
riverSegmentsSource = "/home/s.ryberg/data/EUROSTAT/rivers_and_catchments/data/","*Riversegments.shp"
hydroLakesSource = "/home/s.ryberg/data/WWF/HydroLAKES_polys_v10.shp"
wdpaSource = "/home/s.ryberg/data/protected/WDPA/WDPA_Apr2017-shapefile/clipped","*.shp"
demSource = "/home/s.ryberg/data/zena_data/DEM/eudem_dem_4258_europe.tif"

goodISO = ["CZE","SVK","DNK","MKD","MNE","NLD","ISL","HRV","SVN","BIH","AUT","SWE","BGR",
           "BEL","FIN","ALB","PRT","LIE","LUX","SRB","EST","IRL","HUN","XKO","LTU",
           "LVA","DEU","FRA","ITA","AND","ROU","CHE","ESP","NOR","GBR","POL","GRC","na"]

#goodISO = ["BGR","SVK",]

def makeRegionMask(regionSource, ftrID):
    return gl.RegionMask.fromSourceFeature(regionSource, ftrID, padExtent=3000)

def geomExtractor( extent, source, where=None ): 
    searchGeom = extent.box
    if isinstance(source,str):
        searchFiles = [source,]
    else:
        searchFiles = list(extent.filterSourceDir(source[0], source[1]))
    
    geoms = []
    for f in searchFiles:
        for geom, attr in gl.vectorItems(f, searchGeom, where=where, outputSRS=extent.srs):
            geoms.append( geom.Clone() )

    if len(geoms) == 0:
        return None
    else:
        return geoms

##################################################################
## A General Growing and saving function
def growAndSaveGeoms(reg, ftrID, geom, output_dir, name, tail):
    # make initial matrix
    mat = np.ones(reg.mask.shape, dtype=np.uint8)*255 # Set all values to no data (255)
    mat[reg.mask] = 254 # Set all values in teh region to fully available (254)

    # Only do growing if a geometry is available
    if not geom is None:
        # make grow func
        def doGrow(geom, dist):
            if dist > 0:
                if isinstance(geom, list) or isinstance(geom, filter):
                    #grown = gl.flattenGeomList([g.Buffer(dist) for g in geom])
                    grown = [g.Buffer(dist) for g in geom]
                else:
                    grown = geom.Buffer(dist) # Grow original shape (should already be in EPSG3035)

            else:
                grown = geom

            return grown

        # Create distances
        distances = np.linspace(ranges[name].low, ranges[name].high, 1+ranges[name].steps)

        # Do growing 
        value = 0
        for totalDist in distances: # dont include the last step
            grown = doGrow(geom, totalDist)
            try:
                tmpSource = gl.createVector(grown) # Make a temporary vector file
            except Exception as e:
                print(len(grown), [g.GetGeometryName() for g in grown])
                raise e
            
            indicated = reg.indicateAreas(tmpSource, resolutionDiv=1) > 0.5 # Map onto the RegionMask

            # apply onto matrix
            sel = np.logical_and(mat==254, indicated) # write onto pixels which are indicated and available
            mat[sel] = value
            value += 1

    # make output
    if not isdir(output_dir): mkdir(output_dir)
    fName = "%s_%d-%d-%d_%s_%05d.tif"%(name,ranges[name].low,ranges[name].high,ranges[name].steps,tail,ftrID)
    print(fName)
    d = reg.createRaster(output=join(output_dir,fName), data=mat, overwrite=True, noDataValue=255, dtype=1)

#######################################################
## Buffereing Calculation functions
def sense_OCEANS(regSource, ftrID, topdir, tail):
    name = "oceans"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=44, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_WETLANDS(regSource, ftrID, topdir, tail):
    name = "wetlands"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueMin=35, valueMax=39, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_INDUSTRIAL(regSource, ftrID, topdir, tail):
    name = "industrial"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=3, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_MINES(regSource, ftrID, topdir, tail):
    name = "mines"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=7, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_AGRICULTURE(regSource, ftrID, topdir, tail):
    name = "agriculture"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueMin=12, valueMax=22, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_WOODLANDS_MIXED(regSource, ftrID, topdir, tail):
    name = "woodlands-mixed"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=23, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_WOODLANDS_CONIFEROUS(regSource, ftrID, topdir, tail):
    name = "woodlands-coniferous"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=24, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)

def sense_WOODLANDS_DECIDUOUS(regSource, ftrID, topdir, tail):
    name = "woodlands-deciduous"
    output_dir = join(topdir, name)

    reg = makeRegionMask(regSource, ftrID)
    matrix = reg.indicateValues(clcSource, valueEquals=25, applyMask=False) > 0.5
    geom = gl.maskToGeom(matrix, reg.extent.xyXY, reg.srs, flatten=True)

    growAndSaveGeoms(reg, ftrID, geom, output_dir, name=name, tail=tail)


def sense_ROADS_MAIN(regSource, ftrID, topdir, tail):
    name = "roads-main"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, osmRoadsSource, r"fclass LIKE '%motorway%' OR fclass LIKE '%trunk%' OR fclass LIKE '%primary%'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)

def sense_ROADS_SECONDARY(regSource, ftrID, topdir, tail):
    name = "roads-secondary"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, osmRoadsSource, r"fclass LIKE '%secondary%' OR fclass LIKE '%tertiary%'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)

def sense_POWER_LINES(regSource, ftrID, topdir, tail):
    name = "power-lines"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, osmPowerlinesSource, r"power='line'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)

def sense_RAILWAYS(regSource, ftrID, topdir, tail):
    name = "railways"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, osmRailwaysSource, r"fclass = 'rail'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_RIVERS(regSource, ftrID, topdir, tail):
    name = "rivers"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, riverSegmentsSource)

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_LAKES(regSource, ftrID, topdir, tail):
    name = "lakes"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, hydroLakesSource)

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)

def sense_PARKS(regSource, ftrID, topdir, tail):
    name = "parks"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%park%' OR IUCN_CAT = 'II'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)

def sense_LANDSCAPES(regSource, ftrID, topdir, tail):
    name = "landscapes"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%landscape%' OR IUCN_CAT = 'V'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_MONUMENTS(regSource, ftrID, topdir, tail):
    name = "natural-monuments"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%monument%' OR IUCN_CAT = 'III'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_RESERVES(regSource, ftrID, topdir, tail):
    name = "reserves"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%reserve%' OR IUCN_CAT = 'Ia'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_WILDERNESS(regSource, ftrID, topdir, tail):
    name = "wilderness"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%wilderness%' OR IUCN_CAT = 'Ib'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_BIOSPHERES(regSource, ftrID, topdir, tail):
    name = "biospheres"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%bio%'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_HABITATS(regSource, ftrID, topdir, tail):
    name = "habitats"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%habitat%' OR IUCN_CAT = 'IV'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_BIRDS(regSource, ftrID, topdir, tail):
    name = "birds"
    output_dir = join(topdir, name)
    reg = makeRegionMask(regSource, ftrID)

    geoms = geomExtractor( reg.extent, wdpaSource, where=r"DESIG_ENG LIKE '%bird%'")

    growAndSaveGeoms(reg, ftrID, geom=geoms, output_dir=output_dir, name=name, tail=tail)


def sense_DEM(regionSource, ftrID, topdir, tail):
    output_dir=join(topdir,"dem")

    # Open region
    if not isdir(output_dir): mkdir(output_dir)
    reg = makeRegionMask(regionSource, ftrID)

    # make an analyzing function
    def demAnalyzer(name, ds, corrector=(lambda x:x)):
        mat = np.zeros(reg.mask.shape, dtype=np.uint8)+255
        mat[reg.mask] = 254

        # Create slopes
        values = np.linspace(ranges[name].low, ranges[name].high, ranges[name].steps)

        value=1
        for v in values:
            v2 = corrector(v)
            tmp = reg.indicateValues(ds, valueMin=v2, applyMask=False, resampleAlg='average') > 0.5
            sel = np.logical_and(mat==254, tmp)
            mat[sel] = value
            value += 1

        fName = "%s_%d-%d-%d_%05d.tif"%(name,ranges[name].low,ranges[name].high,ranges[name].steps,ftrID)
        print(fName)
        d = reg.createRaster(output=join(output_dir,fName), data=mat, overwrite=True, noDataValue=255, dtype=1)


    # Get clipped dataset
    demSourceClipped = reg.extent.clipRaster(demSource)

    # do analyses
    analyzer("elevation", demSourceClipped)
    analyzer("slope", gl.rasterGradient(demSourceClipped, mode="slope", factor="latlonToM"), lambda x: np.tan(x*np.pi/180) )
    analyzer("nslope", gl.rasterGradient(demSourceClipped, mode="north-south", factor="latlonToM"), lambda x: np.tan(x*np.pi/180) )
    analyzer("aspect", gl.rasterGradient(demSourceClipped, mode="dir", factor="latlonToM"), lambda x: x*np.pi/180 )


def sense_SETTLEMENTS(regionSource, ftrID, topdir, tail):
    name = "settlements"
    output_dir = join(topdir, name)

    # Open region
    reg = makeRegionMask(regionSource, ftrID)
    
    # Indicate
    indications = {}
    indications["urban"] = reg.indicateValues(urbanClustersSource, valueMin=5000, valueMax=2e7, applyMask=False) > 0.5

    clc_urban = reg.indicateValues(clcSource, valueMin=1, valueMax=2, applyMask=False) > 0.5
    indications["rural"] = np.logical_and(clc_urban, ~indications["urban"])
    
    # Get areas as geometry
    geoms = {}
    for name, matrix in indications.items():
        try:
            geoms[name] = gl.RegionMask(matrix, reg.extent).geometry
        except RuntimeError:
            geoms[name] = None   

    # Grow and Save
    growAndSaveGeoms(reg, ftrID, geom=geoms["urban"], output_dir=output_dir, name="urban", tail=tail)
    growAndSaveGeoms(reg, ftrID, geom=geoms["rural"], output_dir=output_dir, name="rural", tail=tail)

def sense_AIRPORTS(regionSource, ftrID, topdir, tail):
    name = "airports"
    output_dir = join(topdir, name)
    
    ### Open region
    reg = makeRegionMask(regionSource, ftrID)
    searchGeom = reg.extent.box.Buffer(7000)

    ### Get airport regions
    airportMaskDS = reg.indicateValues(clcSource,valueEquals=6) > 0.5
    airportGeoms = gl.maskToGeom(airportMaskDS, bounds=reg.extent.xyXY, srs=reg.srs)

    ### Locate airports and airfields
    airportWhere = "AIRP_USE!=4 AND (AIRP_PASS=1 OR AIRP_PASS=2) AND AIRP_LAND='A'"
    airportCoords = [point.Clone() for point,i in gl.vectorItems(airportsSource, searchGeom, where=airportWhere)]
    for pt in airportCoords: pt.TransformTo(reg.srs)

    airfieldWhere = "AIRP_USE!=4 AND (AIRP_PASS=0 OR AIRP_PASS=9) AND AIRP_LAND='A'"
    airfieldCoords = [point.Clone() for point,i in gl.vectorItems(airportsSource, searchGeom, where=airfieldWhere)]
    for pt in airfieldCoords: pt.TransformTo(reg.srs)

    ### Coordinate locations and shapes
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

    geoms = {}
    geoms["airports-small"] = airportShapes(airfieldCoords, minSize=1e5, defaultRadius=800)
    geoms["airports-large"] = airportShapes(airportCoords, minSize=1e6, defaultRadius=3000)

    # Grow and Save
    growAndSaveGeoms(reg, ftrID, geom=geoms["airports-large"], output_dir=output_dir, name="airports", tail=tail)
    growAndSaveGeoms(reg, ftrID, geom=geoms["airports-small"], output_dir=output_dir, name="airfields", tail=tail)


###################################################################
## MAIN FUNCTIONALITY
if __name__== '__main__':
    START= dt.now()
    print( "TIME START: ", START)

    # Choose the function
    func = globals()["sense_"+sys.argv[1]]

    # Choose the source
    if len(sys.argv)<3:
        source = "reg\\aachenShapefile.shp"
    else:
        source = sys.argv[2]
    tail = splitext(basename(source))[0]

    # Arange workers
    if len(sys.argv)<4:
        doMulti = False
    else:
        doMulti = True
        pool = Pool(int(sys.argv[3]))
    
    # submit jobs
    res = []
    count = -1
    for g,a in gl.vectorItems(source):
        count += 1
        
        # Check if we should skip this region
        if not a["ISO"] in goodISO: continue

        # Do the analysis
        if doMulti:
            res.append(pool.apply_async(func, (source, count, outputDir, tail)))
        else:
            func(source, count, outputDir, tail)
    
    if doMulti:
        
        # Wait for jobs to finish
        pool.close()
        pool.join()

        # Check for errors
        for r,i in zip(res,range(len(res))):
            try:
                r.get()
            except Exception as e:
                print("EXCEPTION AT ID: "+str(i))

    # finished!
    END= dt.now()
    print( "TIME END: ", END)
    print( "CALC TIME: ", (END-START))

