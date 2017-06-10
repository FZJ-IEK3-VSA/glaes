import geokit as gk
from os.path import join, dirname, basename, isfile
from glob import glob
import re
import numpy as np
from collections import namedtuple, OrderedDict
from difflib import SequenceMatcher as SM

from .priors import Priors, PriorSource


###############################
# Make an Exclusion Calculator
class ExclusionCalculator(object):
    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._availability = np.array(s.region.mask)
    
    def save(s, output, **kwargs):
        s.region.createRaster(output=output, data=s.availability, **kwargs)

    def draw(s, ax=None, dataScaling=None, geomSimplify=None, output=None):
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

    ## General excluding functions
    def excludeRasterType(s, source, value=None, verbose=True, **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        if isinstance(source, PriorSource):
            untouchedValue = kwargs.pop("untouchedValue",'noData')
            noDataValue = kwargs.pop("noDataValue", 99999999)

            if value is None:
                if source.excludeDirection == 'lessThan': 
                    value = (None, source.typicalExclusion)
                else: 
                    value = (source.typicalExclusion, None)

            source = source.generateRaster( s.region.extent, untouchedValue=untouchedValue, noDataValue=noDataValue )

        # Indicate on the source
        areas = s.region.indicateValues(source, value, **kwargs)
        
        # exclude the indicated area from the total availability
        s._availability = np.min([s._availability, 1-areas],0)

    def excludeVectorType(s, source, where=None, verbose=True, **kwargs):
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


class WeightedCriterionCalculator(object):
    def __init__(s, region, **kwargs):

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.maskPixels = s.region.mask.sum()

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
    def addCriterion(s, source, vs=None, name=None, weight=1, resampleAlg='cubic', **kwargs):
        """Exclude areas as calcuclated by one of the indicator functions in glaes.indicators

        * if not 'value' input is given, the default buffer/threshold value is chosen (see the individual function's 
          docstring for more information)
        """
        if isinstance(source, PriorSource):
            untouchedValue = kwargs.pop("untouchedValue",'noData')
            noDataValue = kwargs.pop("noDataValue", 99999999)
            name = source.displayName if name is None else name
            
            if vs is None: 
                vs = [(source.edges[0],0), (source.typicalExclusion, 0.5), (source.edges[-1], 1.0)]
                
            source = source.generateRaster( s.region.extent, untouchedValue=untouchedValue, noDataValue=noDataValue )

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