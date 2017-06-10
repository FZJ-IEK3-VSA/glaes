import geokit as gk
import re 
import numpy as np
from glob import glob 
from os.path import dirname, basename, join, isdir
from collections import namedtuple, OrderedDict

# Sort out the data paths
priordir = join(dirname(__file__), "..", "..", "data")
priorgroups = filter(isdir, glob(join(priordir,"*")))

# Typical criteria
Criterion = namedtuple("Criteria","typicalExclusion unit excludeDirection evaluationName doc")
TypicalCriteria = {
    "mixedWoodlandProximity": Criterion(
        300, "meters", 'lessThan',
        "woodlands-mixed",
        "Precalculated pixels within X meters from mixed-forest woodlands"
        ),

    "coniferousWoodlandProximity": Criterion(
        300, "meters", 'lessThan',
        "woodlands-coniferous",
        "Precalculated pixels within X meters from coniferous (needle-leaved) forests"
        ),

    "deciduousWoodlandProximity": Criterion(
        300, "meters", 'lessThan',
        "woodlands-deciduous",
        "Precalculated pixels within X meters from deciduous (broad-leaved) forests"
        ),

    "lakeProximity": Criterion(
        300, "meters", 'lessThan',
        "lakes",
        "Precalculated pixels within X meters from lakes and other stagnant water bodies"
        ),

    "riverProximity": Criterion(
        400, "meters", 'lessThan',
        "rivers",
        "Precalculated pixels within X meters from rivers"
        ),

    "oceanProximity": Criterion(
        300, "meters", 'lessThan',
        "oceans",
        "Precalculated pixels within X meters from oceans"
        ),

    "wetlandProximity": Criterion(
        200, "meters", 'lessThan',
        "wetlands",
        "Precalculated pixels within X meters from wetlands"
        ),

    "elevationThreshold": Criterion(
        1750, "meters", 'greaterThan',
        "elevation",
        "Precalculated pixels where the elevation is above X meters"
        ),

    "slopeThreshold": Criterion(
        11, "degrees", 'greaterThan',
        "slope",
        "Precalculated pixels with an average terrain slope above X degrees"
        ),

    "northFacingSlopeThreshold": Criterion(
        3, "degrees", 'greaterThan',
        "nslope",
        "Precalculated pixels with an average north-facing slope above X degrees"
        ),

    "powerLineProximity": Criterion(
        150, "meters", 'lessThan',
        "power-lines",
        "Precalculated pixels within X meters from power lines"
    ),

    "mainRoadProximity": Criterion(
        200, "meters", 'lessThan',
        "roads-main",
        "Precalculated pixels within X meters from main roads"
    ),

    "secondaryRoadProximity": Criterion(
        100, "meters", 'lessThan',
        "roads-secondary",
        "Precalculated pixels within X meters from secondary roads"
    ),

    "railwayProximity": Criterion(
        200, "meters", 'lessThan',
        "railways",
        "Precalculated pixels within X meters from railways"
    ),

    "urbanProximity": Criterion(
        1500, "meters", 'lessThan',
        "urban",
        "Precalculated pixels within X meters from dense urban areas and cities"
    ),

    "ruralProximity": Criterion(
        700, "meters", 'lessThan',
        "rural",
        "Precalculated pixels within X meters from sparse urban areas and rural settlements"
    ),

    "industrialProximity": Criterion(
        300, "meters", 'lessThan',
        "industrial",
        "Precalculated pixels within X meters from industrial areas"
    ),

    "mineProximity": Criterion(
        200, "meters", 'lessThan',
        "mines",
        "Precalculated pixels within X meters from mining area"
    ),

    "agricultureProximity": Criterion(
        100, "meters", 'lessThan',
        "agriculture",
        "Precalculated pixels within X meters from agricultural areas"
    ),

    "airportProximity": Criterion(
        4000, "meters", 'lessThan',
        "airports",
        "Precalculated pixels within X meters from airports"
    ),

    "airfieldProximity": Criterion(
        3000, "meters", 'lessThan',
        "airfields",
        "Precalculated pixels within X meters from airfields"
    ),

    "parkProximity": Criterion(
        1000, "meters", 'lessThan',
        "parks",
        "Precalculated pixels within X meters from protected parks"
    ),

    "landscapeProximity": Criterion(
        1000, "meters", 'lessThan',
        "landscapes",
        "Precalculated pixels within X meters from protected landscapes"
    ),

    "naturalMonumentProximity": Criterion(
        1000, "meters", 'lessThan',
        "natural-monuments",
        "Precalculated pixels within X meters from protected natural monuments"
    ),

    "reserveProximity": Criterion(
        1000, "meters", 'lessThan',
        "reserves",
        "Precalculated pixels within X meters from protected reserves"
    ),

    "wildernessProximity": Criterion(
        1000, "meters", 'lessThan',
        "wilderness",
        "Precalculated pixels within X meters from protected wildernesses"
    ),

    "biosphereProximity": Criterion(
        1000, "meters", 'lessThan',
        "biospheres",
        "Precalculated pixels within X meters from protected biospheres"
    ),

    "habitatProximity": Criterion(
        1000, "meters", 'lessThan',
        "habitats",
        "Precalculated pixels within X meters from protected habitats"
    ),

    "birdProximity": Criterion(
        1000, "meters", 'lessThan',
        "birds",
        "Precalculated pixels within X meters from bird sanctuaries"
    ),

    "windspeedThreshold": Criterion(
        5, "m/s", 'lessThan',
        "resource-wind-050m",
        "Precalculated pixels with an annual average windspeed measured at 50m BELOW X m/s"
    ),

    "windspeedThresholdAt50m": Criterion(
        5, "m/s", 'lessThan',
        "resource-wind-050m",
        "Precalculated pixels with an annual average windspeed measured at 50m BELOW X m/s"
    ),

    "windspeedThresholdAt100m": Criterion(
        5, "m/s", 'lessThan',
        "resource-wind-100m",
        "Precalculated pixels with an annual average windspeed measured at 100m BELOW X m/s"
    ),

    "ghiThreshold": Criterion(
        4.5, "kWh/m2", 'lessThan',
        "resource-ghi",
        "Precalculated pixels with a mean global-horizontal irradiances (integrated over a day) below X kWh/m2"
    ),

    "dniThreshold": Criterion(
        4.5, "kWh/m2", 'lessThan',
        "resource-dni",
        "Precalculated pixels with a mean direct-normal irradiances (integrated over a day) below X kWh/m2"
    ),

    "connectionDistance": Criterion(
        10000, "meters", 'greaterThan',
        "connection",
        "Precalculated pixels which are further than X meters from the closest grid connection"
    ),

    "accessDistance": Criterion(
        5000, "meters", 'greaterThan',
        "access",
        "Precalculated pixels which are further than X meters from the closest road"
    ),

    }

# Prior datasource class
class PriorSource(object):
    class _LoadFail(Exception):pass

    def __init__(s, path):
        s.path = path

        # load values
        s.parseName(path)

        # Get the variable class
        found = False
        for critName, critInfo in TypicalCriteria.items():
            if s.name == critInfo.evaluationName:
                found = True
                break

        if not found:
            raise RuntimeError("Could not match source to a typical criterion:", path)

        # Set some working info
        s.displayName = critName
        s.typicalExclusion = critInfo.typicalExclusion
        s.unit = critInfo.unit
        s.excludeDirection = critInfo.excludeDirection
        
        doc = critInfo.doc
        doc+= "\n\nTYPICAL EXCLUSION: %f\n"%s.typicalExclusion
        doc+= "\nPRECALCULATED EDGES:\n"
        for i in range(len(s.edges)): 
            doc += "\t%d -> %.2f %s\n"%(i, s.edges[i], s.unit)
        s.__doc__ = doc

    # Create file parser
    rangeRE = re.compile("(?P<constraint>[a-zA-Z0-9-]+)_RS_(?P<low>[0-9-]+)_(?P<high>[0-9-]+)_(?P<steps>[0-9-]+).tif")
    valueRE = re.compile("(?P<constraint>[a-zA-Z0-9-]+)_VS_(?P<values>[0-9_.-]+).tif")

    def parseName(s, path):
        # get the base
        base = basename(path)

        ### Parse information
        good = False

        # First check for a range set
        match = s.rangeRE.match(base)
        if not match is None:
            good = True

            info = match.groupdict() 
            s.name = info["constraint"]
            low = int(info["low"])
            high = int(info["high"])
            steps = int(info["steps"])
            
            s.edges = np.linspace(low, high, steps+1)
            s.values = (s.edges[1:] + s.edges[:-1])/2

        # if that doesn't work, check for an explicit value set
        if not good:
            match = s.valueRE.match(base)
            if not match is None:
                good = True

                info = match.groupdict() 
                s.name = info["constraint"]
                edges = np.array([float(x) for x in info["values"].split("_")])
                
                s.edges = edges
                s.values = (s.edges[1:] + s.edges[:-1])/2 
        
        # If still not good, print warning and skip
        if not good:
            raise s._LoadFail()

    def outsideEdges(s, val):
        if s <= s.edges.max() and s >= s.edges.min(): return True
        else: return False

    #### Make a datasource generator
    def generateRaster(s, extent, untouchedValue='noData', noDataValue=99999999):
        
        # make better values
        values = [s.edges[0], ]
        values.extend( s.values.tolist() ) 

        # make a mutator function to make indexes to estimated values
        #indexToValue = np.vectorize(lambda i: s.values[i]) # TODO: test 'interp' vs 'vectorize'
        #def mutator(data):
        #    return indexToValue(data)
        def mutator(data):
            noData = data == 255 
            untouched = data == 254
            result = np.interp(data, range(len(values)), values)
            result[untouched] = noDataValue if untouchedValue =='noData' else untouchedValue
            result[noData] = noDataValue
            return result

        # mutate main source
        clipDS = extent.clipRaster(s.path)
        mutDS = gk.raster.mutateValues(clipDS, processor=mutator, noData=noDataValue)

        # return
        return mutDS

    #### Make a datasource generator
    def generateVectorFromEdge(s, extent, edgeIndex=-1):
        # Check edgeIndex
        if edgeIndex<0: edgeIndex+=len(s.edges)

        # extract a matrix
        extent = extent.castTo('europe_m').fit(100)
        dataMatrix = extent.extractMatrix(s.path)

        # make geometries
        geoms = []
        fields = dict(edge=[], index=[])

        mat = dataMatrix <= edgeIndex

        if not mat.any(): raise RuntimeError("Failed to find edge in the given extent")

        shape = gk.geom.convertMask(mat, bounds=extent, flat=True)

        geoms.append(shape)
        fields["edge"].append(s.edges[edgeIndex])
        fields["index"].append(edgeIndex)

        # create vector
        geoms = geoms[::-1] # opposite order so they can be visualized
        fields["edge"] = fields["edge"][::-1]
        fields["index"] = fields["index"][::-1]

        vecDS = gk.vector.createVector(geoms, fieldVals=fields)

        # return
        return vecDS

# Load priors
class _Priors(object):pass
class PriorSet(object):
    def __init__(s,path):
        s.path = path
        s._sources = OrderedDict()

        for f in glob(join(path,"*.tif")):
            if f == 'goodAreas.tif':continue
            
            try:
                p = PriorSource(f)
                setattr(s, p.displayName, p)
                s.sources[p.displayName] = p
            except PriorSource._LoadFail:
                print("WARNING: Could not parse file: %s - %s"%(basename(path), basename(f)))
                pass

    def regionIsOkay(s, region):
        # Check if region is okay
        goodPixels = region.indicateValues(join(s.path,"goodArea.tif"), value=1).sum()

        goodRatio = goodPixels/region.mask.sum()
        if goodRatio > 0.9999:
            # Evertyhing is okay
            return True
        elif goodRatio > 0.95:
            print("WARNING: A portion of the defined region is not included of the precalculated exclusion areas")
            return True
        else:
            return False

    @property
    def sources(s): 
        """An easily indexable/searchable list of the PriorSet's sources"""
        return s._sources


Priors = _Priors()
for group in priorgroups:
    # Create the set
    ps = PriorSet(group)

    # Save the set
    setattr(Priors,basename(group),ps) # set the group

