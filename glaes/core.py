import geokit as gk
from os.path import join, dirname, basename, isfile
from glob import glob
import re
import numpy as np
from collections import namedtuple, OrderedDict
from functools import wraps
from difflib import SequenceMatcher as SM
import types


# make an error
class GlaesError(Exception): pass

# Define default values
Criterion = namedtuple("Criteria","typicalExclusion unit evaluationName doc")
CRITERIA = {
    "mixedWoodlandProximity": Criterion(
        300, "meters",
        "woodlands-mixed",
        "pixels within X meters from mixed-forest woodlands"
        ),

    "coniferousWoodlandProximity": Criterion(
        300, "meters",
        "woodlands-coniferous",
        "pixels within X meters from coniferous (needle-leaved) forests"
        ),

    "deciduousWoodlandProximity": Criterion(
        300, "meters",
        "woodlands-deciduous",
        "pixels within X meters from deciduous (broad-leaved) forests"
        ),

    "lakeProximity": Criterion(
        300, "meters",
        "lakes",
        "pixels within X meters from lakes and other stagnant water bodies"
        ),

    "riverProximity": Criterion(
        400, "meters",
        "rivers",
        "pixels within X meters from rivers"
        ),

    "oceanProximity": Criterion(
        300, "meters",
        "oceans",
        "pixels within X meters from oceans"
        ),

    "wetlandProximity": Criterion(
        200, "meters",
        "wetlands",
        "pixels within X meters from wetlands"
        ),

    "elevationThreshold": Criterion(
        1750, "meters",
        "elevation",
        "pixels where the elevation is above X meters"
        ),

    "slopeThreshold": Criterion(
        11, "degrees",
        "slope",
        "pixels with an average terrain slope above X degrees"
        ),

    "northFacingSlopeThreshold": Criterion(
        3, "degrees",
        "nslope",
        "pixels with an average north-facing slope above X degrees"
        ),

    "powerLineProximity": Criterion(
        150, "meters",
        "power-lines",
        "pixels within X meters from power lines"
    ),

    "mainRoadProximity": Criterion(
        200, "meters",
        "roads-main",
        "pixels within X meters from main roads"
    ),

    "secondaryRoadProximity": Criterion(
        100, "meters",
        "roads-secondary",
        "pixels within X meters from secondary roads"
    ),

    "railwayProximity": Criterion(
        200, "meters",
        "railways",
        "pixels within X meters from railways"
    ),

    "urbanProximity": Criterion(
        1500, "meters",
        "urban",
        "pixels within X meters from dense urban areas and cities"
    ),

    "ruralProximity": Criterion(
        700, "meters",
        "rural",
        "pixels within X meters from sparse urban areas and rural settlements"
    ),

    "industrialProximity": Criterion(
        300, "meters",
        "industrial",
        "pixels within X meters from industrial areas"
    ),

    "mineProximity": Criterion(
        200, "meters",
        "mines",
        "pixels within X meters from mining area"
    ),

    "agricultureProximity": Criterion(
        100, "meters",
        "agriculture",
        "pixels within X meters from agricultural areas"
    ),

    "airportProximity": Criterion(
        4000, "meters",
        "airports",
        "pixels within X meters from airports"
    ),

    "airfieldProximity": Criterion(
        3000, "meters",
        "airfields",
        "pixels within X meters from airfields"
    ),

    "parkProximity": Criterion(
        1000, "meters",
        "parks",
        "pixels within X meters from protected parks"
    ),

    "landscapeProximity": Criterion(
        1000, "meters",
        "landscapes",
        "pixels within X meters from protected landscapes"
    ),

    "naturalMonumentProximity": Criterion(
        1000, "meters",
        "natural-monuments",
        "pixels within X meters from protected natural monuments"
    ),

    "reserveProximity": Criterion(
        1000, "meters",
        "reserves",
        "pixels within X meters from protected reserves"
    ),

    "wildernessProximity": Criterion(
        1000, "meters",
        "wilderness",
        "pixels within X meters from protected wildernesses"
    ),

    "biosphereProximity": Criterion(
        1000, "meters",
        "biospheres",
        "pixels within X meters from protected biospheres"
    ),

    "habitatProximity": Criterion(
        1000, "meters",
        "habitats",
        "pixels within X meters from protected habitats"
    ),

    "birdProximity": Criterion(
        1000, "meters",
        "birds",
        "pixels within X meters from bird sanctuaries"
    ),

    "windspeedThreshold": Criterion(
        5, "m/s",
        "resource-wind-050m",
        "pixels with an annual average windspeed measured at 50m BELOW X m/s"
    ),

    "windspeedThresholdAt50m": Criterion(
        5, "m/s",
        "resource-wind-050m",
        "pixels with an annual average windspeed measured at 50m BELOW X m/s"
    ),

    "windspeedThresholdAt100m": Criterion(
        5, "m/s",
        "resource-wind-100m",
        "pixels with an annual average windspeed measured at 100m BELOW X m/s"
    ),

    "ghiThreshold": Criterion(
        4.5, "kWh/m2",
        "resource-ghi",
        "pixels with a mean global-horizontal irradiances (integrated over a day) below X kWh/m2"
    ),

    "dniThreshold": Criterion(
        4.5, "kWh/m2",
        "resource-dni",
        "pixels with a mean direct-normal irradiances (integrated over a day) below X kWh/m2"
    ),

    "connectionDistance": Criterion(
        10000, "meters",
        "connection",
        "pixels which are further than X meters from the closest grid connection"
    ),

    #"accessDistance": Criterion("access", 5000,
    #    ),

    }

##########################3
## Create source library
# Make a file parser
glaesDataFileRE_RS = re.compile("(?P<constraint>[a-zA-Z0-9-]+)_RS_(?P<low>[0-9-]+)_(?P<high>[0-9-]+)_(?P<steps>[0-9-]+).tif")
glaesDataFileRE_VS = re.compile("(?P<constraint>[a-zA-Z0-9-]+)_VS_(?P<values>[0-9_.-]+).tif")

# Set data dir path
datapath = join(dirname(__file__), "..", "data")

# make the source library
evaluationLibrary = {}
Item = namedtuple('Item',"values low high path")

# add sources
goodAreasPath = join(datapath,"goodAreas.tif")

skipFiles = ["goodAreas.tif", ]
for f in glob(join(datapath,"*.tif")):
    baseF = basename(f)
    if baseF in skipFiles: continue
    ### Parse information
    good = False

    # First check for a range set
    match = glaesDataFileRE_RS.match(baseF)
    if not match is None:
        good = True

        info = match.groupdict() 
        name = info["constraint"]
        low = int(info["low"])
        high = int(info["high"])
        steps = int(info["steps"])
        
        values = np.linspace(low, high, steps+1)

    # if that doesn't work, check for an explicit value set
    if not good:
        match = glaesDataFileRE_VS.match(baseF)
        if not match is None:
            good = True

            info = match.groupdict() 
            name = info["constraint"]
            values = np.array([float(x) for x in info["values"].split("_")])
            
            values = values
    
    # If still not good, print warning and skip
    if not good:
        print("WARNING: Could not parse file -", baseF)
        continue

    # Otherwise, a constraint has been found!
    evaluationLibrary[name] = Item(values, min(values), max(values), f)

### Make an index fetcher
def valueToIndex(values, limit, constraint):
    if limit is None: return None

    # check the limit values to make sure they are representable
    if (limit > values.max()) or (limit < values.min()):
        raise GlaesError("The chosen limit for '%s' is beyond the precalculated bounds of %.1f - %.1f "%(
            constraint, values.min(), values.max()))

    # Calculate the indicators and warn for large differences
    limitIndex = np.argmin( np.abs(limit - values) )

    if abs(limit-values[limitIndex])>0.0001: # Check if the value is close to a known value
        # if not close to a known value, check how far it is from teh next value
        if limit < values[limitIndex]: nextIndex = limitIndex-1 #
        else: nextIndex = limitIndex+1

        # Warn the user if the chosen value is more than 5% from the closest value
        if abs((limit - values[limitIndex])/(values[nextIndex] - values[limitIndex])) > 0.05:
            print("WARNING: %s is reverting to the closest precalculated value of %.2f"%(constraint, values[limitIndex]))

    # done!
    return limitIndex

### Setup indicator functions
class _indicators(object):
    """A collection of criterion indicating functions"""
    indicatorList = list(CRITERIA.keys())

    def determineIndicator(s,name):
        scores = [SM(None, name, funcName).ratio() for funcName in s.indicatorList]

        return s.indicatorList[ np.argmax(scores) ]

def indicatorFunc(criterionName):
    criterionInfo = CRITERIA[criterionName]
    
    def wrapper(region, X=criterionInfo.typicalExclusion):
        # make sure reg is a geokit.Regionmask object
        region = gk.RegionMask.load(region)
        
        # fetch data
        path = evaluationLibrary[criterionInfo.evaluationName].path
        values = evaluationLibrary[criterionInfo.evaluationName].values
            
        # Get the indicators
        index = valueToIndex(values, X, criterionName)

        # Get the results as a boolean matrix
        indicated = region.indicateValues(path, value=(None,index))

        # return indicated areas
        return indicated
    
    doc = "Indicates "+criterionInfo.doc
    doc+= "\n\nPRECALCULATED VALUES:\n"
    for i in range(len(evaluationLibrary[criterionInfo.evaluationName].values)): 
        doc += "\t%d -> %.2f %s\n"%(i, evaluationLibrary[criterionInfo.evaluationName].values[i], criterionInfo.unit)
    
    wrapper.__doc__ = doc

    return wrapper

indicators = _indicators()
for c in indicators.indicatorList: setattr(indicators, c, indicatorFunc(c))

###############################
# Make an Exclusion Calculator
class ExclusionCalculator(object):
    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

        # Check if region is okay
        s.goodPixels = s.region.indicateValues(goodAreasPath, value=1).sum()

        goodRatio = s.goodPixels/s.maskPixels
        if goodRatio > 0.9999:
            # Evertyhing is okay
            pass
        elif goodRatio > 0.95:
            print("WARNING: A portion of the defined region is not included of the precalculated exclusion areas")
        else:
            raise GlaesError( "A significant portion of the defined region is not included in the precalculated exclusion areas")

        # Make the total availability matrix
        s._availability = np.array(s.region.mask)
    
    def saveAvailability(s, output, **kwargs):
        s.region.createRaster(output=output, data=s.availability, **kwargs)

    def drawAvailability(s, ax=None, dataScaling=None, geomSimplify=None, output=None):
        # import some things
        from matplotlib.colors import LinearSegmentedColormap
        
        # Do we need to make an axis?
        if ax is None:
            doShow = True
            # import some things
            import matplotlib.pyplot as plt

            # make a figure and axis
            plt.figure(figsize=(12,12))
            ax = plt.subplot(111)
        else: doShow=False

        # fix bad inputs
        if dataScaling: dataScaling = -1*abs(dataScaling)
        if geomSimplify: geomSimplify = abs(geomSimplify)

        # plot the region background
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc=(140/255,0,0), ec='None', zorder=0)

        # plot the availability
        a2b = LinearSegmentedColormap.from_list('alpha_to_blue',[(1,1,1,0),(0,91/255, 130/255, 1)])
        gk.raster.drawImage(s.availability, bounds=s.region.extent, ax=ax, scaling=dataScaling, cmap=a2b)

        # Draw the region boundaries
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc='None', ec='k', linewidth=3)

        # Done!
        if doShow:
            ax.set_aspect('equal')
            ax.autoscale(enable=True)
            if output: 
                plt.savefig(output, dpi=200)
                plt.close()
            else: 
                plt.show()
        else:
            return ax

    @property
    def availability(s): 
        """The pixelated areas left over after all applied exclusions"""
        return s._availability

    @property
    def percentAvailable(s): return 100*s.availability.sum()/s.region.mask.sum()

    @property
    def areaAvailable(s): return s.availability.sum()*s.region.pixelWidth*s.region.pixelHeight

    ## General excluding function
    def exclude(s, verbose=True, **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        for indicator, X in kwargs.items():
            if X==False: continue
            # try to find the right function as a sting
            try:
                func = getattr(indicators, indicator)
            except AttributeError:
                findName = indicators.determineIndicator(indicator)
                if verbose: print("Mapping %s -> %s"%(indicator, findName))
                indicator = findName

                func = getattr(indicators, indicator)

            if X==True: 
                if verbose: print("Using default value for %s (%.2f)"%(indicator, CRITERIA[indicator].typicalExclusion))
                areas = func(s.region)
            else: areas = func(s.region, X)

            # exclude the indicated area from the total availability
            s._availability = np.min([s._availability, 1-areas],0)

##############################
# Make score mapping functions
### Setup indicator functions
class _mappers(object):
    """A collection of criterion indicating functions"""
    mapperList = list(CRITERIA.keys())

    def determineMapper(s,name):
        scores = [SM(None, name, funcName).ratio() for funcName in s.mapperList]

        return s.mapperList[ np.argmax(scores) ]

def mapperFunc(criterionName):
    criterionInfo = CRITERIA[criterionName]

    values = evaluationLibrary[criterionInfo.evaluationName].values
    firstVal = values[0]
    midVal = criterionInfo.typicalExclusion
    lastVal = values[-1]
    
    def wrapper(region, knownValues=(firstVal,midVal,lastVal), knownScores=(0,0.5,1), noData=None, 
                untouchedData=None, normalized=False):
        # make sure known inputs are okay
        if knownValues[0] > knownValues[-1]: # if known values are given in DESCENDING order, flip both
            knownValues = knownValues[::-1]
            knownScores = knownScores[::-1]

        # make sure reg is a geokit.Regionmask object
        region = gk.RegionMask.load(region)
        
        # fetch data
        path = evaluationLibrary[criterionInfo.evaluationName].path
        values = evaluationLibrary[criterionInfo.evaluationName].values
        scores = np.interp(values, knownValues, knownScores)

        # make a mutator
        IndexToScore = np.vectorize( lambda i: scores[i]  )
        def mutator(data):
            data[data==255] = noData if noData else 0
            data[data==254] = untouchedData if untouchedData else len(scores)-1
            return IndexToScore(data)
            
        # Process the dataset
        clippedDS = region.extent.clipRaster(path)
        mutDS = gk.raster.mutateValues(clippedDS, processor=mutator)

        # Warp and extract the results
        scoredValues = region.warp(mutDS)#, resampleAlg='average') # choosing 'average' because the region's pixel size
                                                                 #   should always be less-than or equal-to the evaluated
                                                                 #   datasets

        # normalize, maybe
        if normalized: scoredValues /= max(knownScores)

        # return indicated areas
        return scoredValues
    
    doc = "Maps "+criterionInfo.doc
    doc+= "\n\nPRECALCULATED VALUES:\n"
    for i in range(len(evaluationLibrary[criterionInfo.evaluationName].values)): 
        doc += "\t%d -> %.2f %s\n"%(i, evaluationLibrary[criterionInfo.evaluationName].values[i], criterionInfo.unit)
    
    wrapper.__doc__ = doc

    return wrapper

mappers = _mappers()
for c in mappers.mapperList: setattr(mappers, c, mapperFunc(c))

###############################
# Make an Weighted Criterion Calculator
class WeightedCriterionCalculator(object):
    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

        # Check if region is okay
        s.goodPixels = s.region.indicateValues(goodAreasPath, value=1).sum()

        goodRatio = s.goodPixels/s.maskPixels
        if goodRatio > 0.9999:
            # Evertyhing is okay
            pass
        elif goodRatio > 0.95:
            print("WARNING: A portion of the defined region is not included of the precalculated exclusion areas")
        else:
            raise GlaesError( "A significant portion of the defined region is not included in the precalculated exclusion areas")

        # Make the total availability matrix
        s._unnormalizedWeights = OrderedDict()
        s._totalWeight = 0
        s._result = None
    
    def save(s, output, **kwargs):
        s.region.createRaster(output=output, data=s.result, **kwargs)

    def draw(s, ax=None, dataScaling=None, geomSimplify=None, output=None, method='local'):
        # import some things
        from matplotlib.colors import LinearSegmentedColormap
        
        # Do we need to make an axis?
        if ax is None:
            doShow = True
            # import some things
            import matplotlib.pyplot as plt

            # make a figure and axis
            plt.figure(figsize=(12,12))
            ax = plt.subplot(111)
        else: doShow=False

        # fix bad inputs
        if dataScaling: dataScaling = -1*abs(dataScaling)
        if geomSimplify: geomSimplify = abs(geomSimplify)

        # plot the region background
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc=(140/255,0,0), ec='None', zorder=0)

        # plot the result
        rbg = LinearSegmentedColormap.from_list('blue_green_red',[(130/255,0,0,1), (0,91/255, 130/255, 1), (0,130/255, 0, 1)])
        rbg.set_under(color='w',alpha=1)

        if method == 'local': result = s.resultLocal
        elif method == 'global': result = s.resultGlobal
        elif method == 'raw': result = s.resultRaw
        else: raise GlaesError("method not understood")

        h = gk.raster.drawImage(result, bounds=s.region.extent, ax=ax, scaling=dataScaling, cmap=rbg, vmin=0)

        # Draw the region boundaries
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc='None', ec='k', linewidth=3)

        # Done!
        if doShow:
            plt.colorbar(h)
            ax.set_aspect('equal')
            ax.autoscale(enable=True)
            if output: 
                plt.savefig(output, dpi=200)
                plt.close()
            else: 
                plt.show()
        else:
            return ax

    @property
    def resultLocal(s):
        """The pixelated areas left over after all weighted overlays"""
        minV = s.result[s.region.mask].min()
        maxV = s.result[s.region.mask].max()
        out = (s.result-minV)/(maxV-minV)
        return s.region.applyMask( out, noData=-1)

    @property
    def resultGlobal(s):
        """The pixelated areas left over after all weighted overlays"""
        out = s.result/s.totalWeight
        return s.region.applyMask( out, noData=-1)

    @property
    def resultRaw(s):
        """The pixelated areas left over after all weighted overlays"""
        return s.region.applyMask( s.result, noData=-1)

    @property
    def totalWeight(s): 
        """The pixelated areas left over after all applied exclusions"""
        return s._totalWeight

    @property
    def result(s): 
        """The pixelated areas left over after all applied exclusions"""
        if s._result is None: s.combine()
        return s._result

    ## General excluding function
    def addCriterion(s, name, weight=1, shortName=None, verbose=True, **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """

        # try to find the right function as a sting
        try:
            func = getattr(mappers, name)
        except AttributeError:
            findName = mappers.determineMapper(name)
            if verbose: print("Mapping %s -> %s"%(name, findName))
            mapper = findName

            func = getattr(mappers, mapper)

        # evaluate
        newWeights = weight*func(s.region, normalized=True, **kwargs)
        s._totalWeight += weight

        # append
        shortName = name if shortName is None else shortName
        s._unnormalizedWeights[shortName] = newWeights
    
    def addExternalCriterion(s, source, knownValues, name='external', knownScores=(0,1), weight=1, resampleAlg='cubic'):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        # make sure known inputs are okay
        if knownValues[0] > knownValues[-1]: # if known values are given in DESCENDING order, flip both
            knownValues = knownValues[::-1]
            knownScores = knownScores[::-1]

        # make a mutator
        def mutator(data):
            return np.interp(data, knownValues, knownScores)

        # mclip,mutate,andwarp the datasource
        clippedDS = s.region.extent.clipRaster(source)
        mutDS = gk.raster.mutateValues( clippedDS, processor=mutator)
        result = s.region.warp(mutDS, resampleAlg=resampleAlg)

        newWeights = weight*result
        s._totalWeight += weight

        # append
        s._unnormalizedWeights[name] = newWeights


    def combine(s, combiner='mult'):
        # Set combiner
        if combiner == 'sum':
            result = np.zeroes(s.region.mask.shape)
            for k,v in s._unnormalizedWeights.items(): result+=v
        
        elif combiner == 'mult':
            result = np.ones(s.region.mask.shape)
            for k,v in s._unnormalizedWeights.items(): result*=v
        
        else:
            result = combiner(s._unnormalizedWeights)

        # do combination
        s._result = result
'''
def aspect(s, degreeRange=(-45, 225)):
    # check for bad values
    degreeMin, degreeMax = degreeRange

    if degreeMin<-180 or degreeMin>180: raise GlaesError("degreeMin is outside the allowable limit of -180 to 180")
    if degreeMax<-180 or degreeMax>540: raise GlaesError("degreeMax is outside the allowable limit of -180 to 540")
    if degreeMax<=degreeMin: raise GlaesError("degreeMax must be greater than degreeMin")
    if degreeMax-degreeMin > 360: raise GlaesError("Allowable range spans more than a full circle")
    
    # get constraint info
    lowValue = s.evaluationLibrary["aspect"]["low"]
    highValue = s.evaluationLibrary["aspect"]["high"]
    steps = s.evaluationLibrary["aspect"]["steps"]
    path = s.evaluationLibrary["aspect"]["path"]

    steps = np.linspace(lowValue,highValue,steps+1)
    stepSize = steps[1]-steps[0]

    # evaluate -180 to 180
    if degreeMin < 180:
        minIndicator = np.argmin(np.abs(steps-degreeMin))

        tmpMax = degreeMax if degreeMax<180 else 180
        maxIndicator = np.argmin(np.abs(steps-tmpMax))

        loop1Areas = s.region.indicateValues(path,value=(maxIndicator,minIndicator))
    else: loop1Areas = np.zeros(s._availability.shape, dtype='bool')

    # evaluate 180 to 540
    degreeMin -= 360
    degreeMax -= 360

    if degreeMax > -180:
        tmpMin = degreeMin if degreeMin>-180 else -180
        minIndicator = np.argmin(np.abs(steps-tmpMin))
        maxIndicator = np.argmin(np.abs(steps-degreeMax))

        loop2Areas = s.region.indicateValues(path,value=(maxIndicator,minIndicator))
    else: loop2Areas = np.zeros(s._availability.shape, dtype='bool')

    # combine exclusion areas
    areas = loop1Areas+loop2Areas
    areas[areas>1]=1 # make sure overlapping areas add to a maximum of 1

    # exclude the indicated area from the total availability
    s._availability = np.min([s._availability, 1-areas],0)

    # return the indicated ares incase its useful
    return areas
'''
