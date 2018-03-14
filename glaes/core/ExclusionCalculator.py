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
    """The ExclusionCalculator object makes land eligibility (LE) analyses easy and quick. Once initialized to a particular region, the ExclusionCalculator object can be used to incorporate any geospatial dataset (so long as it is interpretable by GDAL) into the LE analysis. 


    NOTE: By default, ExclusionCalculator is always initialized at 100x100 meter resolution in the EPSG3035 projection system. This is well suited to LE analyses in Europe, however if another region is being investigated or else if another resolution or projection system is desired for any other reason, this can be incorporated as well during the initialization stage.


    Initialization:
        * ExclusionCalculator can be initialized by passing a specific shapefile describing the investigation region:

            >>> ec = ExclusionCalculator(<path>)
        
        * Or a specific srs and resolution can be used:

            >>> ec = ExclusionCalculator(<path>, pixelSize=0.001, srs='latlon')

        * In fact, the ExclusionCalculator initialization is simply a call to geokit.RegionMask.load, so see that for more information. This also means that any geokit.RegoinMask object can be used to initialize the ExclusionCalculator

            >>> rm = geokit.RegionMask.load(<path>, pad=..., srs=..., pixelSize=..., ...)
            >>> ec = ExclusionCalculator(rm)

    Usage:
        * The ExclusionCalculator object contains a member name "availability", which contains the most up to date result of the LE analysis
            - Just after initialization, the the availability matrix is filled with 1's, meaning that all locations are available
            - After excluding locations based off various geospatial datasets, cells in the availability matrix are changed to a value between 0 and 1, where 0 means completely unavailable, 1 means fully available, and intermediate values indicate a pixel which is only partly excluded.
        * Exclusions can be applied by using one of the 'excludeVectorType', 'excludeRasterType', or 'excludePrior' methods
            - The correct method to use depends on the format of the datasource used for exclusions
        * After all exclusions have been applied...
            - The 'draw' method can be used to visualize the result
            - The 'save' method will save the result to a raster file on disc
            - The 'availability' member can be used to extract the availability matrix as a NumPy matrix for further usage
    """
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
        """Initialize the ExclusionCalculator

        Inputs:
            region : The region definition for which land exclusions will be calculated
                - str : A path to a vector file containing the region definition
                - geokit.RegionMask : A preinitialized RegionMask object

            kwargs: 
                * All keyword arguments are passed on to a call to geokit.RegionMask.load
                * Most notably (most only operate when region is a path):
                    - 'srs' can be used to define the reference system to use
                    - 'pixelSize' can be used to define the resolution (in units of the srs)
                    - 'select' can be used to filter the vector source and extract a particular feature
        """

        # load the region
        s.region = gk.RegionMask.load(region, **kwargs)
        s.srs = s.region.srs
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._availability = np.array(s.region.mask, dtype=np.uint8)*100
        #s._availability[~s.region.mask] = 255

        # Make a list of item coords
        s.itemCoords=None
        s._itemCoords=None
    
    def save(s, output, threshold=None, **kwargs):
        """Save the current availability matrix to a raster file
        
        Output will be a byte-valued raster with the following convention:
            0 -> unavailable 
            50 -> Semi-available
            100 -> fully eligibile
            255 -> "no data" (out of region)

        Inputs:
            output - str : The path of the output raster file
            
            threshold - float : The acceptable threshold indicating an available pixel
                * Use this to process the availability matrix before saving it (will save a little bit of space)
                
            kwargs: 
                * All keyword arguments are passed on to a call to geokit.RegionMask.createRaster
                * Most notably:
                    - 'noDataValue' is used to define the no data value
                    - 'dtype' is used to define the data type of the resulting raster
                    - 'overwrite' is used to force overwrite an existing file

        """

        meta={
            "description":"The availability of each pixel for the contextual purpose",
            "units":"percent-available"
        }

        data = s.availability
        if not threshold is None:
            data = (data>=threshold).astype(np.uint8)*100

        data[~s.region.mask] = 255
        s.region.createRaster(output=output, data=data, noData=255, meta=meta, **kwargs)


    def draw(s, ax=None, dataScaling=None, geomSimplify=None, output=None, noBorder=True, goodColor="#9bbb59", excludedColor="#a6161a", figsize=(8,8), legend=True, legendargs={"loc":"lower left"}, pad={"left":0.25}):
        """Draw the current availability matrix on a matplotlib figure

        Inputs:
            ax - matplotlib axis object : The axis to draw the figure onto
                * If given as 'None', then a fresh axis will be produced and displayed or saved immediately
                * When not 'None', then this function returns a handle to the drawn image which can be used however you see fit

            dataScaling - int : A down scaling factor to apply to the visualized matrix
                * Use this when visualizing a large area consumes too much memory

            geomSimplify - int : A down scaling factor to apply when drawing the geometry borders of the ExclusionCalculator's region
                * Use this when the region's geometry is extremely detailed compared to the scale over which it is drawn

            output - str : A path to save the output figure to
                * Only applies when 'ax' is None
                * If this is None and 'ax' is None, the figure is displayed immediately

            noBorder - T/F : A flag determining whether or not to show the borders of the plot's axis
                * Only useful when 'ax' is None

            goodColor : The color to apply to 'good' locations (having a value of 1)
                - str : An HTML color code, or any other string interpretable by matplotlib
                - (r,g,b) : Red, green, blue values given as a tuple
                    * Each must be between 0..1

            excludedColor : The color to apply to 'excluded' locations (having a value of 0)
                * See above for options
    
        """

        # import some things
        from matplotlib.colors import LinearSegmentedColormap
        
        # Do we need to make an axis?
        if ax is None:
            doShow = True
            # import some things
            import matplotlib.pyplot as plt

            # make a figure and axis
            plt.figure(figsize=figsize)
            ax = plt.subplot(111)
        else: doShow=False

        # fix bad inputs
        if dataScaling: dataScaling = -1*abs(dataScaling)
        if geomSimplify: geomSimplify = abs(geomSimplify)

        # plot the region background
        s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc=excludedColor, ec='None', zorder=0)

        # plot the availability
        a2b = LinearSegmentedColormap.from_list('alpha_to_blue',[(1,1,1,0),goodColor])
        gk.raster.drawImage(s.availability, bounds=s.region.extent, ax=ax, scaling=dataScaling, cmap=a2b, vmax=100)

        # Draw the region boundaries
        edge = s.region.drawGeometry(ax=ax, simplification=geomSimplify, fc='None', ec='k', linewidth=3)

        # Draw Items?
        if not s._itemCoords is None:
            items = ax.plot(s._itemCoords[:,0], s._itemCoords[:,1], 'ok')

        # Done!
        if doShow:
            from matplotlib.patches import Patch
            p = s.percentAvailable
            a = s.region.mask.sum(dtype=np.int64)*s.region.pixelWidth*s.region.pixelHeight
            areaLabel = s.region.srs.GetAttrValue("Unit").lower()
            if areaLabel=="metre" or areaLabel=="meter":
                a = a/1000000
                areaLabel = "km"
            elif areaLabel=="feet" or areaLabel=="foot":
                areaLabel = "ft"
            elif areaLabel=="degree":
                areaLabel = "deg"

            if a<0.001:
                regionLabel = "{0:.3e} ${1}^2$".format(a, areaLabel)
            elif a<0: 
                regionLabel = "{0:.4f} ${1}^2$".format(a, areaLabel)
            elif a<1000: 
                regionLabel = "{0:.2f} ${1}^2$".format(a, areaLabel)
            else: 
                regionLabel = "{0:,.0f} ${1}^2$".format(a, areaLabel)

            patches = [
                Patch( ec="k", fc="None", linewidth=3, label=regionLabel),
                Patch( color=excludedColor, label="Excluded: %.2f%%"%(100-p) ),
                Patch( color=goodColor, label="Eligible: %.2f%%"%(p) ),
            ]
            if not s._itemCoords is None:
                h = plt.plot([],[],'ok', label="Items: {:,d}".format(s._itemCoords.shape[0]) )
                patches.append( h[0] )

            if legend:
                _legendargs = dict(loc="lower right", fontsize=14)
                _legendargs.update(legendargs)
                plt.legend(handles=patches, **_legendargs)

            ax.set_aspect('equal')
            ax.autoscale(enable=True)
            if not pad is None:
                yMin,yMax = ax.get_ylim()
                xMin,xMax = ax.get_xlim()

                _yMin,_yMax = ax.get_ylim()
                _xMin,_xMax = ax.get_xlim()

                if "left" in pad:   
                    _xMin = xMin - pad["left"]*(xMax-xMin)
                if "right" in pad:   
                    _xMax = xMax + pad["right"]*(xMax-yMin)
                if "down" in pad:   
                    _yMin = yMin - pad["down"]*(yMax-yMin)
                if "up" in pad:   
                    _yMax = yMax + pad["up"]*(yMax-yMin)

                ax.set_ylim(_yMin, _yMax)
                ax.set_xlim(_xMin, _xMax)

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
        """A matrix containing the availability of each location after all applied exclusions.
            * A value of 100 is interpreted as fully available
            * A value of 0 is interpreted as completely excluded
            * In between values are...in between"""
        return s._availability 

    @property
    def percentAvailable(s): 
        """The percent of the region which remains available"""
        return 100*s.availability.sum(dtype=np.int64)/100/s.region.mask.sum()

    @property
    def areaAvailable(s): 
        """The area of the region which remains available
            * Units are defined by the srs used to initialize the ExclusionCalculator"""
        return s.availability.sum(dtype=np.int64)*s.region.pixelWidth*s.region.pixelHeight

    ## General excluding functions
    def excludeRasterType(s, source, value=None, valueMin=None, valueMax=None, prewarp=False, invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the values in a raster datasource

        Inputs:
            source : The raster datasource defining the values for each location
                - str : A path to a raster data source
                - gdal Dataset : An open gdal dataset object held in memory
            
            value : The exact value, or value range to exclude
                - Numeric : The exact value to exclude
                    * Generally this should only be done when the raster datasource contains integer values, 
                      otherwise a range of values should be used to avoid float comparison errors
                - ( Numeric, Numeric ) : The low and high boundary describing the range of values to exclude
                    * If either boundary is given as None, then it is interpreted as unlimited

            valueMin - Numeric : A convenience input when the desired exclusion range is all values above 
                a minimal value
                * This is equivalent to value=(valueMin, None)

            valueMax - Numeric : A convenience input when the desired exclusion range is all values below 
                a maximal value
                * This is equivalent to value=(None, valueMax)
            
            prewarp: When not False, the source will be warped to the calculator's mask context before processing
                T/F : If True, warping will be performed using the bilieanr resample algorithm
                str : Warp using the indicated resampleAlgorithm (options: near, bilinear, cubic, average)
                dict : A dictionary of arguments corresponding to geokit.RegionMask.warp

            kwargs
                * All other keyword arguments are passed on to a call to geokit.RegionMask.indicateValues
                * Most importantly...
                    - 'resampeAlg' is used to define how the indication matrix is warped to fit the region mask
                    - 'resolutionDiv' is used to increase the resolution of the working matrix during processing
                    - 'buffer' is used to add a buffer region (given in units of the ExclusionCalculator's srs) around the raw indicated areas
        """
        if value is None and valueMin is None and valueMax is None:
            raise GlaesError("One of value, valueMin, or valueMax must be given")

        # Do prewarp, if needed
        if prewarp:
            prewarpArgs = dict(resampleAlg="bilinear")
            if isinstance(prewarp, str): prewarpArgs["resampleAlg"] = prewarp
            elif isinstance(prewarp, dict): prewarpArgs.update(prewarp)
            
            source = s.region.warp(source, returnAsSource=True, **prewarpArgs)

        # Indicate on the source
        if not (valueMin is None and valueMax is None): value = (valueMin,valueMax)
        areas = (s.region.indicateValues(source, value, **kwargs)*100).astype(np.uint8)
        
        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min([s._availability, areas if invert else 100-areas],axis=0)
        elif mode == "include":
            s._availability = np.max([s._availability, 100-areas if invert else areas],axis=0)
            s._availability[~s.region.mask] = 0
        else:
            raise GlaesError("mode must be 'exclude' or 'include'")

    def excludeVectorType(s, source, where=None, invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the features in a vector datasource

        Inputs:
            source : The raster datasource defining the features to indicate from
                - str : A path to a vector data source
                - gdal Dataset : An open gdal dataset object held in memory
            
            where - str : A filtering statement to apply to the datasource before the initial indication
                * This is an SQL like statement which can operate on features in the datasource
                * For tips, see "http://www.gdal.org/ogr_sql.html"
                * For example...
                    - If the datasource had features which each have an attribute called 'type' and only features with the type "protected" are wanted, the correct statement would be: 
                        where="type='protected'"
            
            invert - T/F : Flag causing the exclusion of all unindicated areas, instead of all indicated areas 
            
            kwargs
                * All other keyword arguments are passed on to a call to geokit.RegionMask.indicateFeatures
                * Most importantly...
                    - 'resolutionDiv' is used to increase the resolution of the working matrix during processing
                    - 'buffer' is used to add a buffer region (given in units of the ExclusionCalculator's srs) around the raw indicated features
        """
        if isinstance(source, PriorSource):
            edgeI = kwargs.pop("edgeIndex", np.argwhere(source.edges==source.typicalExclusion))
            source = source.generateVectorFromEdge( s.region.extent, edgeIndex=edgeI )

        # Indicate on the source
        areas = (s.region.indicateFeatures(source, where=where, **kwargs)*100).astype(np.uint8)
        
        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min([s._availability, areas if invert else 100-areas],axis=0)
        elif mode == "include":
            s._availability = np.max([s._availability, 100-areas if invert else areas],axis=0)
            s._availability[~s.region.mask] = 0
        else:
            raise GlaesError("mode must be 'exclude' or 'include'")

    def excludePrior(s, prior, value=None, valueMin=None, valueMax=None, invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the values in one of the Prior datasources

            * The Prior datasources are currently only defined over Europe
            * All Prior datasources are defined in the EPSG3035 projection system with 100x100 meter resolution
            * For each call to excludePrior, a temporary raster datasource is generated around the ExclusionCalculator's region, after which a call to ExclusionCalculator.excludeRasterType is made, therefore all the same inputs apply here as well

        Inputs:
            source - str : The name of the Prior datasource defining the values for each location
                * If the name does not exactly match one of the Prior datasources, the best fitting name will be used (and you will be informed about which one is chosen)
                * See the ExclusionCalculator.typicalExclusions dictionary for the Prior dataset names and what a typical exclusion threshold would be
                * A list of Prior datasets names can also be found in Priors.sources
            
            value : The exact value, or value range to exclude
                - Numeric : The exact value to exclude
                    * Generally this should only be done when the raster datasource contains integer values, 
                      otherwise a range of values should be used to avoid float comparison errors
                - ( Numeric, Numeric ) : The low and high boundary describing the range of values to exclude
                    * If either boundary is given as None, then it is interpreted as unlimited
                * If value, valueMin, and valueMax are all None, the typical exclusion threshold given from ExclusionCalculator.typicalExclusions is used

            valueMin - Numeric : A convenience input when the desired exclusion range is all values above 
                a minimal value
                * This is equivalent to value=(valueMin, None)

            valueMax - Numeric : A convenience input when the desired exclusion range is all values below 
                a maximal value
                * This is equivalent to value=(None, valueMax)

            kwargs
                * All other keyword arguments are passed on to a call to geokit.RegionMask.indicateValues
                * Most importantly...
                    - 'buffer' is used to add a buffer region (given in units of the ExclusionCalculator's srs) around the raw indicated areas
        """
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
        s.excludeRasterType( source, value=value, invert=invert, mode=mode, **kwargs)

    def shrinkAvailability(s, dist, threshold=50, **kwargs):
    	"""Shrinks the current availability by a given distance in the given SRS"""
    	geom = gk.geom.convertMask(s.availability>=threshold, bounds=s.region.extent, srs=s.region.srs, flat=True)
    	geom = geom.Buffer(dist)
    	newAvail = (s.region.indicateGeoms(geom, **kwargs)*100).astype(np.uint8)
    	s._availability = newAvail

    def pruneIsolatedAreas(s, minSize, threshold=50):
        """Removes areas which are smaller than 'minSize'

        * minSize is given in units of the calculator's srs
        """ 
        # Create a vector file of geometries larger than 'minSize'
        geoms = gk.geom.convertMask( s.availability>=threshold, bounds=s.region.extent, srs=s.region.srs)
        geoms = list(filter( lambda x: x.Area()>=minSize, geoms ))
        vec = gk.vector.createVector(geoms)

        # Replace current availability matrix
        s._availability = s.region.rasterize( vec).astype(np.uint8 )*100

    def distributeItems(s, separation, pixelDivision=5, threshold=50, maxItems=10000000, outputSRS=4326, output=None, asArea=False, minArea=100000):
        """Distribute the maximal number of minimally separated items within the available areas
        
        Returns a list of x/y coordinates (in the ExclusionCalculator's srs) of each placed item

        Inputs:
            separation - float : The minimal distance between two items

            pixelDivision - int : The inter-pixel fidelity to use when deciding where items can be placed

            threshold : The minimal availability value to allow placing an item on

            maxItems - int : The maximal number of items to place in the area
                * Used to initialize a placement list and prevent using too much memory when the number of placements gets absurd

            outputSRS : The output SRS system to use
                * The default (4326) corresponds to regular lat/lon

            output : A path to an output shapefile
        """
        # Preprocess availability
        workingAvailability = s.availability >= threshold
        if not workingAvailability.dtype == 'bool':
            raise s.GlaesError("Working availability must be boolean type")
        workingAvailability[~s.region.mask] = False
        # Turn separation into pixel distances
        separation = separation / s.region.pixelSize
        sep2 = separation**2
        sepFloor = max(separation-1,0)
        sepFloor2 = sepFloor**2
        sepCeil = separation+1

        # Make geom list
        x = np.zeros((maxItems))
        y = np.zeros((maxItems))

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

                # only continue if there are no points in the immediate range of the whole pixel
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

        s._itemCoords = coords

        if not outputSRS is None:
            newCoords = gk.srs.xyTransform(coords, fromSRS=s.region.srs, toSRS=outputSRS)
            newCoords = np.column_stack( [ [v[0] for v in newCoords], [v[1] for v in newCoords]] )
            coords = newCoords
        s.itemCoords = coords

        # Make shapefile
        if not output is None:
            srs = gk.srs.loadSRS(outputSRS) if not outputSRS is None else s.region.srs
            # Should the locations be converted to areas?
            if asArea:
                # Do Voronoi
                from scipy.spatial import Voronoi
                pts = np.concatenate([s._itemCoords,
                    [ (s.region.extent.xMin*0.95,s.region.extent.yMin*0.95,), (s.region.extent.xMin*0.95,s.region.extent.yMax*1.05,),
                      (s.region.extent.xMax*1.05,s.region.extent.yMin*0.95,), (s.region.extent.xMax*1.05,s.region.extent.yMax*1.05,), ]])

                v = Voronoi(pts)
                
                # Create regions
                geoms = []
                for reg in v.regions:
                    path = []
                    if -1 in reg or len(reg)==0: continue
                    for pid in reg:
                        path.append(v.vertices[pid])
                    path.append(v.vertices[reg[0]])
                        
                    geoms.append( gk.geom.polygon(path, srs=s.region.srs ))

                if not len(geoms) == len(s._itemCoords):
                    raise GlaesError("Mismatching geometry count")

                # Create a list of geometry from each region WITH availability
                vec = gk.vector.createVector(geoms, fieldVals={"pid":range(1,len(geoms)+1)})
                areaMap = s.region.rasterize(vec, attribute="pid", dtype=int) * (s._availability>threshold)

                geoms = gk.geom.vectorize(areaMap, bounds=s.region.extent, srs=s.region.srs, flat=True)
                geoms = list(filter(lambda x:x.Area()>=minArea, geoms))

                if not srs.IsSame(s.region.srs):
                    geoms = gk.geom.transform(geoms, fromSRS=s.region.srs, toSRS=srs)

                # Add 'area' column
                geoms = pd.DataFrame({"geom":geoms})
                geoms["areas"] = [g.Area() for g in geoms]

            else: # Just write the points                
                geoms = [gk.geom.point(loc, srs=srs) for loc in coords]
            
            gk.vector.createVector(geoms, output=output)

        else:
            return coords

    def distributeAreas(s, targetArea=2000000, threshold=50):
        """Distribute the maximal number of roughly equal sized partitions within the available areas
        
        Returns a list of tuples, each containing x/y coordinates (in the ExclusionCalculator's srs) of a partition as well as the partition's geometry

        Inputs:
            targetArea - float : The desired area (in units of the ExclusionCalculator's srs) for each partition

            threshold : The minimal availability value to allow placing an item on
        """
        # Get the working availability
        workingAvailability = s.availability >= threshold
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
