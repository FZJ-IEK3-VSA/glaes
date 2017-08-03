import geokit as gk
from os.path import join, dirname, basename, isfile
from glob import glob
import re
import numpy as np
from collections import namedtuple, OrderedDict

from .priors import Priors, PriorSource
from .util import GlaesError

Areas = namedtuple('Areas', "coordinates geoms")

###############################
# Make an Exclusion Calculator
class ExclusionCalculator(object):
    typicalExclusions = {
        "access_distance": (5000, None ),
        "agriculture_proximity": (None, 50 ),
        "agriculture_arable_proximity": (None, 50 ),
        "agriculture_pasture_proximity": (None, 50 ),
        "agriculture_permanent_crop_proximity": (None, 50 ),
        "agriculture_heterogeneous_proximity": (None, 50 ),
        "airfield_proximity": (None, 3000 ),
        "airport_proximity": (None, 5000 ),
        "connection_distance": (10000, None ),
        "dni_threshold": (None, 5.0 ),
        "elevation_threshold": (1800, None ),
        "ghi_threshold": (None, 5.0 ),
        "industrial_proximity": (None, 300 ),
        "lake_proximity": (None, 400 ),
        "mining_proximity": (None, 100 ),
        "ocean_proximity": (None, 1000 ),
        "power_line_proximity": (None, 200 ),
        "protected_biosphere_proximity": (None, 300 ),
        "protected_bird_proximity": (None, 1500 ),
        "protected_habitat_proximity": (None, 1500 ),
        "protected_landscape_proximity": (None, 500 ),
        "protected_natural_monument_proximity": (None, 1000 ),
        "protected_park_proximity": (None, 1000 ),
        "protected_reserve_proximity": (None, 500 ),
        "protected_wilderness_proximity": (None, 1000 ),
        "camping_proximity": (None, 1000), 
        "touristic_proximity": (None, 800),
        "leisure_proximity": (None, 1000),
        "railway_proximity": (None, 150 ),
        "river_proximity": (None, 200 ),
        "roads_proximity": (None, 150 ), 
        "roads_main_proximity": (None, 200 ),
        "roads_secondary_proximity": (None, 100 ),
        "sand_proximity": (None, 1000 ),
        "settlement_proximity": (None, 500 ),
        "settlement_urban_proximity": (None, 1000 ),
        "slope_threshold": (10, None ),
        "slope_north_facing_threshold": (3, None ),
        "wetland_proximity": (None, 1000 ),
        "waterbody_proximity": (None, 300 ),
        "windspeed_100m_threshold": (None, 4.5 ),
        "windspeed_50m_threshold": (None, 4.5 ),
        "woodland_proximity": (None, 300 ),
        "woodland_coniferous_proximity": (None, 300 ),
        "woodland_deciduous_proximity": (None, 300 ),
        "woodland_mixed_proximity": (None, 300 )}

    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._availability = np.array(s.region.mask)

        # Make a list of item coords
        s.itemCoords=None
    
    def save(s, output, **kwargs):
        s.region.createRaster(output=output, data=s.availability, **kwargs)

    def draw(s, ax=None, dataScaling=None, geomSimplify=None, output=None, noBorder=True, goodColor=(0,91/255, 130/255), excludedColor=(140/255,0,0)):
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
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc=excludedColor, ec='None', zorder=0)

        # plot the availability
        a2b = LinearSegmentedColormap.from_list('alpha_to_blue',[(1,1,1,0),goodColor])
        gk.raster.drawImage(s.availability, bounds=s.region.extent, ax=ax, scaling=dataScaling, cmap=a2b)

        # Draw the region boundaries
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc='None', ec='k', linewidth=3)

        # Draw Items?
        if not s.itemCoords is None:
            ax.plot(s.itemCoords[:,0], s.itemCoords[:,1], 'ok')

        # Done!
        if doShow:
            ax.set_aspect('equal')
            ax.autoscale(enable=True)

            if noBorder:
                plt.axis('off')

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
        return s._availability > 0.5

    @property
    def percentAvailable(s): return 100*s.availability.sum()/s.region.mask.sum()

    @property
    def areaAvailable(s): return s.availability.sum()*s.region.pixelWidth*s.region.pixelHeight

    ## General excluding functions
    def excludeRasterType(s, source, value, valueMin=None, valueMax=None, **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        # Indicate on the source
        if not (valueMin is None and valueMax is None): value = (valueMin,valueMax)
        areas = s.region.indicateValues(source, value, **kwargs)
        
        # exclude the indicated area from the total availability
        s._availability = np.min([s._availability, 1-areas],0)

    def excludeVectorType(s, source, where=None, **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        if isinstance(source, PriorSource):
            edgeI = kwargs.pop("edgeIndex", np.argwhere(source.edges==source.typicalExclusion))
            source = source.generateVectorFromEdge( s.region.extent, edgeIndex=edgeI )

        # Indicate on the source
        areas = s.region.indicateFeatures(source, where=where, **kwargs)
        
        # exclude the indicated area from the total availability
        s._availability = np.min([s._availability, 1-areas],0)

    def excludePrior(s, prior, value=None, valueMin=None, valueMax=None, **kwargs):

        if not (valueMin is None and valueMax is None): value = (valueMin,valueMax)

        # make sure we have a Prior object
        if isinstance(prior, str): prior = Priors[prior]

        if not isinstance( prior, PriorSource): raise GlaesError("'prior' input must be a Prior object or an associated string")

        # try to get the default value if one isn't given
        if value is None:
            try:
                value = s.typicalExclusions[prior.displayName]
            except KeyError:
                raise GlaesError("Could not find a default exclusion set for %s"%prior.displayName)

        # Check the value input
        if isinstance(value, tuple):

            # Check the boundaries
            if not value[0] is None: prior.containsValue(value[0], True)
            if not value[1] is None: prior.containsValue(value[1], True)

            # Check edges
            if not value[0] is None: prior.valueOnEdge(value[0], True)
            if not value[1] is None: prior.valueOnEdge(value[1], True)
        else:
            if not value==0:
                print("WARNING: It is advisable to exclude by a value range instead of a singular value")    

        # Make the raster
        source = prior.generateRaster( s.region.extent )

        # Call the excluder
        s.excludeRasterType( source, value=value, **kwargs)

    def distributeItems(s, separation, pixelDivision=5, preprocessor=lambda x: x>=0.5, maxTurbines=10000000):
        # Preprocess availability
        workingAvailability = preprocessor(s.availability)
        if not workingAvailability.dtype == 'bool':
            raise s.GlaesError("Working availability must be boolean type")

        # Turn separation into pixel distances
        separation = separation / s.region.pixelSize
        sep2 = separation**2
        sepFloor = max(separation-1,0)
        sepFloor2 = sepFloor**2
        sepCeil = separation+1

        # Make geom list
        x = np.zeros((maxTurbines)) # initialize 1 000 000 possible x locations (can be expanded later)
        y = np.zeros((maxTurbines)) # initialize 1 000 000 possible y locations (can be expanded later)

        bot = 0
        cnt = 0

        # start searching
        yN, xN = workingAvailability.shape
        substeps = np.linspace(-0.5, 0.5, pixelDivision)
        substeps[0]+=0.0001 # add a tiny bit to the left/top edge (so that the point is definitely in the right pixel)
        substeps[-1]-=0.0001 # subtract a tiny bit to the right/bottom edge for the same reason
        
        for yi in range(yN):
            # update the "bottom" value
            tooFarBehind = yi-y[bot:cnt] > sepCeil # find only those values which have a y-component greater than the separation distance
            if tooFarBehind.size>0: 
                bot += np.argmin(tooFarBehind) # since tooFarBehind is boolean, argmin should get the first index where it is false

            #print("yi:", yi, "   BOT:", bot, "   COUNT:",cnt)

            for xi in np.argwhere(workingAvailability[yi,:]):
                # Clip the total placement arrays
                xClip = x[bot:cnt]
                yClip = y[bot:cnt]

                # calculate distances
                xDist = np.abs(xClip-xi)
                yDist = np.abs(yClip-yi)

                # Get the indicies in the possible range
                possiblyInRange = np.argwhere( xDist <= sepCeil ) # all y values should already be within the sepCeil 

                # only continue if there are no points in the immidiate range of the whole pixel
                immidiateRange = (xDist[possiblyInRange]*xDist[possiblyInRange]) + (yDist[possiblyInRange]*yDist[possiblyInRange]) <= sepFloor2
                if immidiateRange.any(): continue

                # Start searching in the 'sub pixel'
                found = False
                for xsp in substeps+xi:
                    xSubDist = np.abs(xClip[possiblyInRange]-xsp)
                    for ysp in substeps+yi:
                        ySubDist = np.abs(yClip[possiblyInRange]-ysp)

                        # Test if any points in the range are overlapping
                        overlapping = (xSubDist*xSubDist + ySubDist*ySubDist) <= sep2
                        if not overlapping.any():
                            found = True
                            break

                    if found: break

                # Add if found
                if found:
                    x[cnt] = xsp
                    y[cnt] = ysp
                    cnt += 1
                 
        # Convert identified points back into the region's coordinates
        coords = np.zeros((cnt,2))
        coords[:,0] = s.region.extent.xMin + (x[:cnt]+0.5)*s.region.pixelWidth # shifted by 0.5 so that index corresponds to the center of the pixel
        coords[:,1] = s.region.extent.yMax - (y[:cnt]+0.5)*s.region.pixelHeight # shifted by 0.5 so that index corresponds to the center of the pixel

        # Done!
        s.itemCoords = coords
        return coords
    
    def distributeAreas(s, targetArea=2000000, preprocessor=lambda x: x>=0.5, **kwargs):
        # Get the working availability
        workingAvailability = preprocessor(s.availability)
        if not workingAvailability.dtype == 'bool':
            raise s.GlaesError("Working availability must be boolean type")

        # polygonize availability
        geoms, values = gk.raster.polygonize(workingAvailability, bounds=s.region.extent, noDataValue=0, shrink=True)
        
        # partition each of the new geometries
        newGeoms = []
        for gi in range(0, len(geoms)):
            g = geoms[gi]
            
            gArea = g.Area()
            if gArea <= targetArea*1.6:
                newGeoms.append(g.Clone())
                
            else:
                #gTmp, vTmp = gk.geom.partitionArea(g, targetArea=fudgedTargetArea, resolution=s.region.pixelSize, **kwargs)
                try:
                    gTmp = gk.geom.partition(g, targetArea=targetArea, growStep=s.region.pixelSize*3)
                except Exception as e:
                    print(gi)
                    raise e 

                newGeoms.extend(gTmp)
                
        # finalize geom list
        def inRange(x):
            area = x.Area()
            if area >= (0.5*targetArea) and area <= (2.1 * targetArea):
                return True
            else:
                return False
        newGeoms = np.array(list(filter(inRange, newGeoms)))

        # get centroids and return
        return Areas( np.array([g.Centroid().GetPoints() for g in newGeoms]), newGeoms)


class WeightedCriterionCalculator(object):
    typicalValueScores = {
        # THESE NEED TO BE CHECKED!!!!
        "access_distance": ((0,1), (5000,0.5), (20000,0), ),
        "agriculture_proximity": ((0,0), (100,0.5), (1000,1), ),
        "agriculture_arable_proximity": ((0,0), (100,0.5), (1000,1), ),
        "agriculture_pasture_proximity": ((0,0), (100,0.5), (1000,1), ),
        "agriculture_permanent_crop_proximity": ((0,0), (100,0.5), (1000,1), ),
        "agriculture_heterogeneous_proximity": ((0,0), (100,0.5), (1000,1), ),
        "airfield_proximity": ((0,0), (3000,0.5), (8000,1), ),
        "airport_proximity": ((0,0), (4000,0.5), (10000,1), ),
        "connection_distance": ((0,1), (10000,0.5), (20000,0), ),
        "dni_threshold": ((0,0), (4,0.5), (8,1), ),
        "elevation_threshold": ((1500,1), (1750,0.5), (2000,0), ),
        "ghi_threshold": ((0,0), (4,0.5), (8,1), ),
        "industrial_proximity": ((0,0), (300,0.5), (1000,1), ),
        "lake_proximity": ((0,0), (300,0.5), (1000,1), ),
        "mining_proximity": ((0,0), (200,0.5), (1000,1), ),
        "ocean_proximity": ((0,0), (300,0.5), (1000,1), ),
        "power_line_proximity": ((0,0), (150,0.5), (500,1), ),
        "protected_biosphere_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_bird_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_habitat_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_landscape_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_natural_monument_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_park_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_reserve_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "protected_wilderness_proximity": ((0,0), (1000,0.5), (2000,1), ),
        "railway_proximity": ((0,0), (200,0.5), (1000,1), ),
        "river_proximity": ((0,0), (400,0.5), (1000,1), ),
        "roads_proximity": ((0,0), (200,0.5), (1000,1), ),
        "roads_main_proximity": ((0,0), (200,0.5), (1000,1), ),
        "roads_secondary_proximity": ((0,0), (100,0.5), (1000,1), ),
        "settlement_proximity": ((0,0), (700,0.5), (2000,1), ),
        "settlement_urban_proximity": ((0,0), (1500,0.5), (3000,1), ),
        "slope_threshold": ((0,1), (11,0.5), (20,0), ),
        "slope_north_facing_threshold": ((0,1), (3,0.5), (5,0), ),
        "waterbody_proximity": ((0,0), (400,0.5), (1000,1), ),
        "wetland_proximity": ((0,0), (200,0.5), (1000,1), ),
        "windspeed_100m_threshold": ((0,0), (5,0.5), (8,1), ),
        "windspeed_50m_threshold": ((0,0), (5,0.5), (8,1), ),
        "woodland_coniferous_proximity": ((0,0), (300,0.5), (1000,1), ),
        "woodland_deciduous_proximity": ((0,0), (300,0.5), (1000,1), ),
        "woodland_mixed_proximity": ((0,0), (300,0.5), (1000,1), )}

    def __init__(s, region, exclusions=None, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._unnormalizedWeights = OrderedDict()
        s._totalWeight = 0
        s._result = None
        s.noData = -1

        # Keep an exclusion matrix
        if not exclusions is None:
            if not exclusions.dtype == np.bool:
                raise GlaesError("Exclusion matrix must be a boolean type")
            if not exclusions.shape == s.region.mask.shape:
                raise GlaesError("Exclusion matrix shape must match the region's mask shape")
            s.exclusions = exclusions
        else:
            s.exclusions = None
    
    def save(s, output, **kwargs):
        s.region.createRaster(output=output, data=s.result, **kwargs)

    def draw(s, ax=None, dataScaling=None, geomSimplify=None, output=None, view='local'):
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

        if view == 'local': result = s.resultLocal
        elif view == 'global': result = s.resultGlobal
        elif view == 'raw': result = s.resultRaw
        else: raise GlaesError("view not understood")

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
        return s.region.applyMask( out, noData=s.noData)

    @property
    def resultGlobal(s):
        """The pixelated areas left over after all weighted overlays"""
        out = s.result/s.totalWeight
        return s.region.applyMask( out, noData=s.noData)

    @property
    def resultRaw(s):
        """The pixelated areas left over after all weighted overlays"""
        return s.region.applyMask( s.result, noData=s.noData)

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
    def addCriterion(s, source, vs=None, name=None, weight=1, resampleAlg='cubic', **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        if isinstance(source, str): source = Priors[source]
        if isinstance(source, PriorSource):
            name = source.displayName if name is None else name
            
            if vs is None:
                vs = s.typicalValueScores[source.displayName]
                
            source = source.generateRaster( s.region.extent)

            skipClip=True # we dont need to clip again

        else: 
            skipClip=False # we will likely still need to clip
            if name is None: 
                if isinstance(source, str):
                    name=basename(source)
                else:
                    raise GlaesError("A 'name' input must be provided when source is not a prior or a path")

        # make sure known inputs are okay
        knownValues = [x[0] for x in vs]
        knownScores = [x[1] for x in vs]

        if knownValues[0] > knownValues[-1]: # if known values are given in DESCENDING order, flip both
            knownValues = knownValues[::-1]
            knownScores = knownScores[::-1]

        # make a mutator
        def mutator(data):
            return np.interp(data, knownValues, knownScores)

        # mclip,mutate,andwarp the datasource
        clippedDS = source if skipClip else s.region.extent.clipRaster(source)
        mutDS = gk.raster.mutateValues( clippedDS, processor=mutator)
        result = s.region.warp(mutDS, resampleAlg=resampleAlg)

        newWeights = weight*result
        s._totalWeight += weight

        # append
        s._unnormalizedWeights[name] = newWeights

        # make sure to clear any old result
        s._result = None

    def combine(s, combiner='sum'):
        # Set combiner
        if combiner == 'sum':
            result = np.zeros(s.region.mask.shape)
            for k,v in s._unnormalizedWeights.items(): result+=v
        
        elif combiner == 'mult':
            result = np.ones(s.region.mask.shape)
            for k,v in s._unnormalizedWeights.items(): result*=v
        
        else:
            result = combiner(s._unnormalizedWeights)

        # apply mask if one exists
        if not s.exclusions is None:
            result *= s.exclusions

        # do combination
        s._result = result

    def extractValues(s, locations, view='local', srs=None, mode='linear-spline', **kwargs):
        # get result 
        if view == 'local': result = s.resultLocal
        elif view == 'global': result = s.resultGlobal
        elif view == 'raw': result = s.resultRaw
        else: raise GlaesError("view not understood")

        # Fill no data
        if not mode=='near': # there's no point if we're using 'near'
            result[ result==s.noData ] = 0

        # make result into a dataset
        ds = s.region.createRaster(data=result)

        # make sure we have an srs
        if srs is None: srs = s.region.srs
        else: srs = gk.srs.loadSRS(srs)

        # extract values
        vals = gk.raster.interpolateValues(ds, locations, pointSRS=srs, mode=mode, **kwargs)

        # done!
        return vals
