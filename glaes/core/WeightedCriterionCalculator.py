from .ExclusionCalculator import *

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
