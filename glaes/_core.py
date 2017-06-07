import geokit as gk
from os.path import join, dirname, basename, isfile
from glob import glob
import re
import numpy as np
from collections import namedtuple
from functools import wraps

# make an error
class GlaesError(Exception): pass

# Define default values
DEFAULTS = {
         "woodlands-mixed": 300, # Indicates distances too close to mixed-tree forests (m)
    "woodlands-coniferous": 300, # Indicates distances too close to coniferous forests (m)
     "woodlands-deciduous": 300, # Indicates distances too close to deciduous forests(m)
                   "lakes": 300, # Indicates distances too close to lakes (m)
                  "rivers": 400, # Indicates distances too close to rivers (m)
                  "oceans": 300, # Indicates distances too close to oceans (m)
                "wetlands": 200, # Indicates distances too close to wetlands (m)
               "elevation": 1750, # Indicates elevations above X (m)
                   "slope": 11, # Indicates slopes above X (degree)
                  "nslope": 3, # Indicates north-facing slopes above X (degree)
                  "aspect": 0, # Indicates aspects in given range (degrees)
             "power-lines": 150, # Indicates distances too close to power-lines (m)
              "roads-main": 200, # Indicates distances too close to main roads (m)
         "roads-secondary": 100, # Indicates distances too close to secondary roads (m)
                "railways": 200, # Indicates distances too close to railways (m)
                   "urban": 1500, # Indicates distances too close to dense settlements (m)
                   "rural":  700, # Indicates distances too close to light settlements (m)
              "industrial": 300, # Indicates distances too close to industrial areas (m)
                   "mines": 200, # Indicates distances too close to mines (m)
             "agriculture": 100, # Indicates distances too close to aggriculture areas (m)
                "airports": 4000, # Indicates distances too close to airports (m)
               "airfields": 3000, # Indicates distances too close to airfields (m)
                   "parks": 1000, # Indicates distances too close to protected parks (m)
              "landscapes": 1000, # Indicates distances too close to protected landscapes (m)
       "natural-monuments": 1000, # Indicates distances too close to protected natural-monuments (m)
                "reserves": 1000, # Indicates distances too close to protected reserves (m)
              "wilderness": 1000, # Indicates distances too close to protected wilderness (m)
              "biospheres": 1000, # Indicates distances too close to protected biospheres (m)
                "habitats": 1000, # Indicates distances too close to protected habitats (m)
                   "birds": 1000, # Indicates distances too close to protected bird areas (m)
           "resource-wind": 4, # Indicates areas with average wind speed below X (m/s)
            "resource-ghi": 200, # Indicates areas with average total daily irradiance below X (kWh/m2/day)
            "resource-dni": 200, # Indicates areas with average total daily irradiance below X (kWh/m2/day)
         "grid-connection": 10000, # Indicates distances too far from power grid (m)
             "road-access": 5000, # Indicates distances too far from roads (m)
    }


##########################3
## Create source library
# Make a file parser
glaesDataFileRE = re.compile("(?P<constraint>[a-zA-Z-]+)_(?P<low>[0-9-]+)_(?P<high>[0-9-]+)_(?P<steps>[0-9-]+).tif")

# Set data dir path
path = join(dirname(__file__), "..", "data")

# make the source library
constraintLibrary = {}

# add sources
constraint = namedtuple('constraint',"low high steps stepsize path")
for f in glob(join(path,"*_*_*_*.tif")):
    # Parse information
    glaesDataFile = glaesDataFileRE.match(basename(f))
    if glaesDataFile is None: 
        print("WARNING: Could not parse file -", basename(f))
        continue
    # add to library
    info = glaesDataFile.groupdict()

    low = int(info["low"])
    high = int(info["high"])
    steps = int(info["steps"])
    stepsize = (high-low)/steps
    path = f

    constraintLibrary[info["constraint"]] = constraint(low,high,steps,stepsize,path)

# Make Exclusion Calculator
class ExclusionCalculator(object):
    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region)

        # Make the total availability matrix
        s._availability = np.array(s.region.mask)

        # exclude keyword inputs
        for k,v in kwargs.items():
            # look for a value or use default if True
            if isinstance(v,bool):
                if v: s.exclude(k)
            else:
                s.exclude(k,v)
    
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
    def exclude(s, name, value=None):
        """Exclude areas as caluclated by one of the indicators.indicateXXX scripts where XXX is replaced by a constraint name

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's docstrings for more information)
        * name examples:
            - Urban 
            - Rural
            - Industrial
            - Agriculture
            - Airfields
            - Airports
            - Biospheres
            - Birds
            - Elevation
            - Lakes
            - Landscapes
            - Mines
            - Monuments
            - NSlope
            - Oceans
            - Parks
            - Reserves
            - Rivers
            - Slope
            - Wetlands
            - SecondaryRoads 
            - MainRoads 
            - PowerLines 
            - Railways 
            - Habitats 
            - Wilderness 
            - MixedWoodlands 
            - ConiferousWoodlands 
            - DeciduousWoodland
        """

        # get the indicated area
        try:
            func = globals()["indicate"+name]
        except KeyError:
            findName = "indicate"+name.lower().replace(" ","").replace("_","")

            # Try to find the right function
            found = False
            for funcName in filter(lambda x: 'indicate' in x, globals().keys()):
                if findName == funcName.lower():
                    found = True
                    func = globals()[funcName]
                    break

            if not found:
                raise GlaesError("Could not understand function call. See the docstring for this function for the appropriate options")


        if value is None: areas = func(s.region)
        else: areas = func(s.region, value)

        # exclude the indicated area from the total availability
        s._availability = np.min([s._availability, 1-areas],0)
 
## Helper functions
def _general_indicator(region, constraint, limit):
    # make sure reg is a geokit.Regionmask object
    region = gk.RegionMask.load(region)
    
    # fetch data
    lowValue = constraintLibrary[constraint].low
    highValue = constraintLibrary[constraint].high
    steps = constraintLibrary[constraint].steps
    path = constraintLibrary[constraint].path

    steps = np.linspace(lowValue,highValue,steps+1)
    stepSize = steps[1]-steps[0]

    # check the limit values to make sure they are representable
    if (limit > highValue) if stepSize>0 else (limit < highValue):
        raise GlaesError("The chosen limit for '%s' is beyond the precalculated high-exclusion value of %.1f"%(constraint, highValue))

    if (limit < lowValue) if stepSize>0 else (limit > lowValue):
        raise GlaesError("The chosen limit for '%s' is beyond the precalculated low-exclusion value of %.1f"%(constraint, lowValue))

    # Calculate the indicators and warn for large differences
    highIndicator = np.argmin( np.abs(limit - steps) )
    if abs((limit - steps[highIndicator])/stepSize) > 0.10:
        print("WARNING: %s is reverting to the closest precalculated value of %.1f"%(constraint, steps[highIndicator]))
    
    # Get the results as a boolean matrix
    indicated = region.indicateValues(path, value=(None,highIndicator))

    # return indicated areas
    return indicated

def bufferedDoc(name, detailed, *args):
    tmp = '''Indicates {0} in the regional context with an added security distance (m) 

    PRECALCULATED SECURITY DISTANCES:
        - Low indication: {1:d} m
        - High indication: {2:d} m
        - Stepsize: {3:.3f} m
    '''.format(detailed, constraintLibrary[name].low, constraintLibrary[name].high, constraintLibrary[name].stepsize)

    for a in args: tmp += a+'\n'

    return tmp

def thresholdDoc(name, detailed, unit, *args):
    tmp = '''Indicates {0} in the regional context above a threshold value ({4}) 

    PRECALCULATED SECURITY DISTANCES:
        - Low indication: {1:d} {4}
        - High indication: {2:d} {4}
        - Stepsize: {3:.3f} {4}
    '''.format(detailed, constraintLibrary[name].low, constraintLibrary[name].high, constraintLibrary[name].stepsize, unit)

    for a in args: tmp += a+'\n'

    return tmp

#######################################################################
##  Make Indicator Functions
def indicateUrban(region, securityDistance=DEFAULTS["urban"]): return _general_indicator(region, "urban", securityDistance)
indicateUrban.__doc__ = bufferedDoc( "urban", "dense urban areas and cities")

def indicateRural(region, securityDistance=DEFAULTS["rural"]): return _general_indicator(region, "rural", securityDistance)
indicateRural.__doc__ = bufferedDoc( "rural" , "sparse urban areas and rural settlements")

def indicateIndustrial(region, securityDistance=DEFAULTS["industrial"]): return _general_indicator(region, "industrial", securityDistance)
indicateIndustrial.__doc__ = bufferedDoc( "industrial" , "industrial areas")

def indicateAgriculture(region, securityDistance=DEFAULTS["agriculture"]): return _general_indicator(region, "agriculture", securityDistance)
indicateAgriculture.__doc__ = bufferedDoc( "agriculture" , "agriculture areas")

def indicateAirfields(region, securityDistance=DEFAULTS["airfields"]): return _general_indicator(region, "airfields", securityDistance)
indicateAirfields.__doc__ = bufferedDoc( "airfields" , "airfields")

def indicateAirports(region, securityDistance=DEFAULTS["airports"]): return _general_indicator(region, "airports", securityDistance)
indicateAirports.__doc__ = bufferedDoc( "airports" , "airports")

def indicateBiospheres(region, securityDistance=DEFAULTS["biospheres"]): return _general_indicator(region, "biospheres", securityDistance)
indicateBiospheres.__doc__ = bufferedDoc( "biospheres" , "protected biospheres")

def indicateBirds(region, securityDistance=DEFAULTS["birds"]): return _general_indicator(region, "birds", securityDistance)
indicateBirds.__doc__ = bufferedDoc( "birds" , "protected bird zones")

def indicateElevation(region, elevation=DEFAULTS["elevation"]): return _general_indicator(region, "elevation", elevation)
indicateElevation.__doc__ = thresholdDoc( "elevation" , "elevation", "m")

def indicateLakes(region, securityDistance=DEFAULTS["lakes"]): return _general_indicator(region, "lakes", securityDistance)
indicateLakes.__doc__ = bufferedDoc( "lakes" , "lakes and other stagnant water bodies")

def indicateLandscapes(region, securityDistance=DEFAULTS["landscapes"]): return _general_indicator(region, "landscapes", securityDistance)
indicateLandscapes.__doc__ = bufferedDoc( "landscapes" , "protected landscapes")

def indicateMines(region, securityDistance=DEFAULTS["mines"]): return _general_indicator(region, "mines", securityDistance)
indicateMines.__doc__ = bufferedDoc( "mines" , "mining areas")

def indicateMonuments(region, securityDistance=DEFAULTS["natural-monuments"]): return _general_indicator(region, "natural-monuments", securityDistance)
indicateMonuments.__doc__ = bufferedDoc( "natural-monuments" , "protected natural monuments")

def indicateNSlope(region, degreesAbove=DEFAULTS["nslope"]): return _general_indicator(region, "nslope", degreesAbove)
indicateNSlope.__doc__ = thresholdDoc( "nslope" , "north-facing slopes", "degrees")

def indicateOceans(region, securityDistance=DEFAULTS["oceans"]): return _general_indicator(region, "oceans", securityDistance)
indicateOceans.__doc__ = bufferedDoc( "oceans" , "oceans")

def indicateParks(region, securityDistance=DEFAULTS["parks"]): return _general_indicator(region, "parks", securityDistance)
indicateParks.__doc__ = bufferedDoc( "parks" , "protected parks")

def indicateReserves(region, securityDistance=DEFAULTS["reserves"]): return _general_indicator(region, "reserves", securityDistance)
indicateReserves.__doc__ = bufferedDoc( "reserves" , "protected reserves")

def indicateRivers(region, securityDistance=DEFAULTS["rivers"]): return _general_indicator(region, "rivers", securityDistance)
indicateReserves.__doc__ = bufferedDoc( "rivers" , "rivers")

def indicateSlope(region, degreesAbove=DEFAULTS["slope"]): return _general_indicator(region, "slope", degreesAbove)
indicateSlope.__doc__ = thresholdDoc( "slope" , "terrain slopes", "degrees")

def indicateWetlands(region, securityDistance=DEFAULTS["wetlands"]): return _general_indicator(region, "wetlands", securityDistance)
indicateWetlands.__doc__ = bufferedDoc( "wetlands" , "wetland areas")

def indicateSecondaryRoads( region, securityDistance=DEFAULTS["roads-secondary"] ): return _general_indicator(region, "roads-secondary", securityDistance)
indicateSecondaryRoads.__doc__ = bufferedDoc( "roads-secondary", "secondary roads")

def indicateMainRoads( region, securityDistance=DEFAULTS["roads-main"] ): return _general_indicator(region, "roads-main", securityDistance)
indicateMainRoads.__doc__ = bufferedDoc( "roads-main", "main roads")

def indicatePowerLines( region, securityDistance=DEFAULTS["power-lines"] ): return _general_indicator(region, "power-lines", securityDistance)
indicatePowerLines.__doc__ = bufferedDoc( "power-lines", "power lines")

def indicateRailways( region, securityDistance=DEFAULTS["railways"] ): return _general_indicator(region, "railways", securityDistance)
indicateRailways.__doc__ = bufferedDoc( "railways", "railways")

def indicateHabitats( region, securityDistance=DEFAULTS["habitats"] ): return _general_indicator(region, "habitats", securityDistance)
indicateHabitats.__doc__ = bufferedDoc( "habitats", "habitats")

def indicateWilderness( region, securityDistance=DEFAULTS["wilderness"] ): return _general_indicator(region, "wilderness", securityDistance)
indicateWilderness.__doc__ = bufferedDoc( "wilderness", "wildernesses")

def indicateMixedWoodlands():pass
#def indicateMixedWoodlands( region, securityDistance=DEFAULTS["woodlands-mixed"] ): return _general_indicator(region, "woodlands-mixed", securityDistance)
#indicateMixedWoodlands.__doc__ = bufferedDoc( "woodlands-mixed", "mixed woodland areas")

def indicateConiferousWoodlands( region, securityDistance=DEFAULTS["woodlands-coniferous"] ): return _general_indicator(region, "woodlands-coniferous", securityDistance)
indicateConiferousWoodlands.__doc__ = bufferedDoc( "woodlands-coniferous", "coniferous (needle-leaved) woodland areas")

def indicateDeciduousWoodlands( region, securityDistance=DEFAULTS["woodlands-deciduous"] ): return _general_indicator(region, "woodlands-deciduous", securityDistance)
indicateDeciduousWoodlands.__doc__ = bufferedDoc( "woodlands-deciduous", "deciduous (broad-leaved) woodland areas")


'''
def excludeAspect(s, degreeRange=(-45, 225)):
    # check for bad values
    degreeMin, degreeMax = degreeRange

    if degreeMin<-180 or degreeMin>180: raise GlaesError("degreeMin is outside the allowable limit of -180 to 180")
    if degreeMax<-180 or degreeMax>540: raise GlaesError("degreeMax is outside the allowable limit of -180 to 540")
    if degreeMax<=degreeMin: raise GlaesError("degreeMax must be greater than degreeMin")
    if degreeMax-degreeMin > 360: raise GlaesError("Allowable range spans more than a full circle")
    
    # get constraint info
    lowValue = s.constraintLibrary["aspect"]["low"]
    highValue = s.constraintLibrary["aspect"]["high"]
    steps = s.constraintLibrary["aspect"]["steps"]
    path = s.constraintLibrary["aspect"]["path"]

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