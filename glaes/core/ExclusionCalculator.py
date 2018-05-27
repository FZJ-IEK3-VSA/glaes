from .util import *
from .priors import *

Areas = namedtuple('Areas', "coordinates geoms")

###############################
# Make an Exclusion Calculator
class ExclusionCalculator(object):
    """The ExclusionCalculator object makes land eligibility (LE) analyses easy 
    and quick. Once initialized to a particular region, the ExclusionCalculator 
    object can be used to incorporate any geospatial dataset (so long as it is 
    interpretable by GDAL) into the LE analysis. 


    Note:
    ----- 
    By default, ExclusionCalculator is always initialized at 100x100 meter 
    resolution in the EPSG3035 projection system. This is well-suited to LE 
    analyses in Europe, however if another region is being investigated or else 
    if another resolution or projection system is desired for any other reason, 
    this can be incorporated as well during the initialization stage.

    If you need to find a new projection system for your analyses, the following 
    website is helpful: http://spatialreference.org/ref/epsg/


    Initialization:
    ---------------
    * ExclusionCalculator can be initialized by passing a specific vector file
      describing the investigation region:

        >>> ec = ExclusionCalculator(<path>)
    
    * A particular srs and resolution can be used:

        >>> ec = ExclusionCalculator(<path>, pixelRes=0.001, srs='latlon')

    * In fact, the ExclusionCalculator initialization is simply a call to 
      geokit.RegionMask.load, so see that for more information. This also means 
      that any geokit.RegoinMask object can be used to initialize the 
      ExclusionCalculator

        >>> rm = geokit.RegionMask.load(<path>, pad=..., srs=..., pixelRes=..., ...)
        >>> ec = ExclusionCalculator(rm)

    Usage:
    ------
    * The ExclusionCalculator object contains a member name "availability", which
      contains the most up to date result of the LE analysis
        - Just after initialization, the the availability matrix is filled with 
          100's, meaning that all locations are available
        - After excluding locations based off various geospatial datasets, cells
          in the availability matrix are changed to a value between 0 and 100, 
          where 0 means completely unavailable, 100 means fully available, and 
          intermediate values indicate a pixel which is only partly excluded.

    * Exclusions can be applied by using one of the 'excludeVectorType', 
      'excludeRasterType', or 'excludePrior' methods
        - The correct method to use depends on the format of the datasource used
          for exclusions
    * After all exclusions have been applied...
        - The 'draw' method can be used to visualize the result
        - The 'save' method will save the result to a raster file on disc
        - The 'availability' member can be used to extract the availability matrix
          as a NumPy matrix for further usage
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
        "dni_threshold": (None, 3.0 ),
        "elevation_threshold": (1800, None ),
        "ghi_threshold": (None, 3.0 ),
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

    def __init__(s, region, srs=3035, pixelRes=100, where=None, padExtent=0, **kwargs):
        """Initialize the ExclusionCalculator

        Parameters:
        -----------
        region : str, geokit.RegionMask 
            The regional definition for the land eligibility analysis
            * If given as a string, must be a path to a vector file
            * If given as a RegionMask, it is taken directly despite other 
              arguments

        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs context of the generated RegionMask object
            * The default srs EPSG3035 is only valid for a European context
            * If an integer is given, it is treated as an EPSG identifier
              - Look here for options: http://spatialreference.org/ref/epsg/
              * Only effective if 'region' is a path to a vector

        pixelRes : float or tuple
            The generated RegionMask's native pixel size(s)
            * If float : A pixel size to apply to both the X and Y dimension
            * If (float float) : An X-dimension and Y-dimension pixel size
            * Only effective if 'region' is a path to a vector

        where : str, int; optional
            If string -> An SQL-like where statement to apply to the source
            If int -> The feature's ID within the vector dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            * Only effective if 'region' is a path to a vector
            Example: If the source vector has a string attribute called "ISO" and 
                     a integer attribute called "POP", you could use....

                where = "ISO='DEU' AND POP>1000"


        padExtent : float; optional
            An amount by which to pad the extent before generating the RegionMask
            * Only effective if 'region' is a path to a vector

        kwargs: 
            * Keyword arguments are passed on to a call to geokit.RegionMask.load
            * Only take effect when the 'region' argument is a string
    
        """

        # load the region
        s.region = gk.RegionMask.load(region, srs=srs, pixelRes=pixelRes, where=where, padExtent=padExtent, **kwargs)
        s.srs = s.region.srs
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._availability = np.array(s.region.mask, dtype=np.uint8)*100
        #s._availability[~s.region.mask] = 255

        # Make a list of item coords
        s.itemCoords=None
        s._itemCoords=None
        s._areas=None
    
    def save(s, output, threshold=None, **kwargs):
        """Save the current availability matrix to a raster file
        
        Output will be a byte-valued raster with the following convention:
            0     -> unavailable 
            1..99 -> Semi-available
            100   -> fully eligibile
            255   -> "no data" (out of region)

        Parameters:
        -----------
        output : str
            The path of the output raster file
            * Must end in ".tif"
        
        threshold : float; optional
            The acceptable threshold indicating an available pixel
            * Use this to process the availability matrix before saving it (will 
              save a little bit of space)
            
        kwargs: 
            * All keyword arguments are passed on to a call to 
              geokit.RegionMask.createRaster
            * Most notably:
                - 'dtype' is used to define the data type of the resulting raster
                - 'overwrite' is used to force overwrite an existing file

        """

        meta={
            "description":"The availability of each pixel",
            "units":"percent-available"
        }

        data = s.availability
        if not threshold is None:
            data = (data>=threshold).astype(np.uint8)*100

        data[~s.region.mask] = 255
        s.region.createRaster(output=output, data=data, noData=255, meta=meta, **kwargs)


    def draw(s, ax=None, goodColor="#9bbb59", excludedColor="#a6161a", legend=True, legendargs={"loc":"lower left"}, dataScalingFactor=1, geomSimplificationFactor=5000, **kwargs):
        """Draw the current availability matrix on a matplotlib figure

        Note:
        -----
        To save the result somewhere, call 'plt.savefig(...)' immediately 
        calling this function. To directly view the result, call 'plt.show()'

        Parameters:
        -----------
        ax: matplotlib axis object; optional
            The axis to draw the figure onto
            * If given as 'None', then a fresh axis will be produced and displayed
              or saved immediately

        goodColor: A matplotlib color
            The color to apply to 'good' locations (having a value of 100)
            
        excludedColor: A matplotlib color
            The color to apply to 'excluded' locations (having a value of 0)
            
        legend: bool; optional
            If True, a legend will be drawn

        legendargs: dict; optional
            Arguments to pass to the drawn legend (via axes.legend(...))

        dataScalingFactor: int; optional
            A down scaling factor to apply to the visualized availability matrix
            * Use this when visualizing a large areas
            * seting this to 1 will apply no scaling

        geomSimplificationFactor: int
            A down scaling factor to apply when drawing the geometry borders of 
            the ExclusionCalculator's region
            * Use this when the region's geometry is extremely detailed compared 
              to the scale over which it is drawn
            * Setting this to None will apply no simplification
        
        **kwargs:
            All keyword arguments are passed on to a call to geokit.drawImage

        Returns:
        --------
        matplotlib axes object
        

        """
        # import some things
        from matplotlib.colors import LinearSegmentedColormap
        
        # First draw the availability matrix
        b2g = LinearSegmentedColormap.from_list('bad_to_good',[excludedColor,goodColor])

        if not "figsize" in kwargs:
            ratio = s.region.mask.shape[1]/s.region.mask.shape[0]
            kwargs["figsize"] = (8*ratio*1.2, 8)

        kwargs["topMargin"] = kwargs.get("topMargin", 0.01)
        kwargs["bottomMargin"] = kwargs.get("bottomMargin", 0.02)
        kwargs["rightMargin"] = kwargs.get("rightMargin", 0.01)
        kwargs["leftMargin"] = kwargs.get("leftMargin", 0.02)
        kwargs["hideAxis"] = kwargs.get("hideAxis", True)
        kwargs["cmap"] = kwargs.get("cmap", b2g)
        kwargs["cbar"] = kwargs.get("cbar", False)
        kwargs["vmin"] = kwargs.get("vmin", 0)
        kwargs["vmax"] = kwargs.get("vmax", 100)
        kwargs["cbarTitle"] = kwargs.get("cbarTitle", "Pixel Availability")

        axh1 = s.region.drawImage( s.availability, ax=ax, drawSelf=False, scaling=dataScalingFactor, **kwargs)

        # # Draw the mask to blank out the out of region areas
        # w2a = LinearSegmentedColormap.from_list('white_to_alpha',[(1,1,1,1),(1,1,1,0)])
        # axh2 = s.region.drawImage( s.region.mask, ax=axh1, drawSelf=False, cmap=w2a, cbar=False)

        # Draw the Regional geometry
        axh3 = s.region.drawSelf( fc='None', ax=axh1, linewidth=2, simplificationFactor=geomSimplificationFactor )        
        
        # Draw Points, maybe?
        if not s._itemCoords is None:
            axh1.ax.plot(s._itemCoords[:,0],s._itemCoords[:,1],'ok')
            
        # Draw Areas, maybe?
        if not s._areas is None:
            gk.drawGeoms( s._areas, srs=s.region.srs, ax=axh1, fc='None', ec="k", linewidth=1, simplificationFactor=None )

        # Make legend?
        if legend:
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
                h = axh1.ax.plot([],[],'ok', label="Items: {:,d}".format(s._itemCoords.shape[0]) )
                patches.append( h[0] )

            _legendargs = dict(loc="lower right", fontsize=14)
            _legendargs.update(legendargs)
            axh1.ax.legend(handles=patches, **_legendargs)        

        # Done!!
        return axh1.ax

    @property
    def availability(s): 
        """A matrix containing the availability of each location after all applied exclusions.
            * A value of 100 is interpreted as fully available
            * A value of 0 is interpreted as completely excluded
            * In between values are...in between"""
        tmp = s._availability.astype(np.float32)
        tmp[~s.region.mask] = np.nan
        return tmp

    @property
    def percentAvailable(s): 
        """The percent of the region which remains available"""
        return s._availability.sum(dtype=np.int64)/s.region.mask.sum()

    @property
    def areaAvailable(s): 
        """The area of the region which remains available
            * Units are defined by the srs used to initialize the ExclusionCalculator"""
        return s._availability.sum(dtype=np.int64)*s.region.pixelWidth*s.region.pixelHeight

    ## General excluding functions
    def excludeRasterType(s, source, value=None, buffer=None, resolutionDiv=1, prewarp=False, invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the values in a raster datasource

        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource defining the criteria values for each location
            
        value : tuple, or numeric 
            The exact value, or value range to exclude
            * If Numeric, should be The exact value to exclude
                * Generally this should only be done when the raster datasource 
                  contains integer values, otherwise a range of values should be 
                  used to avoid float comparison errors
            * If ( Numeric, Numeric ), the low and high boundary describing the 
              range of values to exclude
                * If either boundary is given as None, then it is interpreted as 
                  unlimited

        buffer : float; optional
            A buffer region to add around the indicated pixels
            * Units are in the RegionMask's srs
            * The buffering occurs AFTER the indication and warping step and
              so it may not represent the original dataset exactly
              - Buffering can be made more accurate by increasing the 
                'resolutionDiv' input
        
        resolutionDiv : int; optional
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details
        
        prewarp : bool or str or dict; optional
            When given, the source will be warped to the calculator's mask context 
            before processing
            * If True, warping will be performed using the bilinear scheme
            * If str, warp using the indicated resampleAlgorithm 
              - options: 'near', 'bilinear', 'cubic', 'average'
            * If dict, a dictionary of arguments is expected
              - These are passed along to geokit.RegionMask.warp

        invert: bool; optional
            If True, flip indications

        mode: string; optional
            * If 'exclude', then the indicated pixels are subtracted from the 
              current availability matrix
            * If 'include', then the indicated pixel are added back into the
              availability matrix

        kwargs
            * All other keyword arguments are passed on to a call to 
              geokit.RegionMask.indicateValues

        """
        # Do prewarp, if needed
        if prewarp:
            prewarpArgs = dict(resampleAlg="bilinear")
            if isinstance(prewarp, str): prewarpArgs["resampleAlg"] = prewarp
            elif isinstance(prewarp, dict): prewarpArgs.update(prewarp)
            
            source = s.region.warp(source, returnAsSource=True, **prewarpArgs)

        # Indicate on the source
        areas = (s.region.indicateValues(source, value, buffer=buffer, resolutionDiv=resolutionDiv, applyMask=False, **kwargs)*100).astype(np.uint8)
        
        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min([s._availability, areas if invert else 100-areas],axis=0)
        elif mode == "include":
            s._availability = np.max([s._availability, 100-areas if invert else areas],axis=0)
            s._availability[~s.region.mask] = 0
        else:
            raise GlaesError("mode must be 'exclude' or 'include'")

    def excludeVectorType(s, source, where=None, buffer=None, bufferMethod='geom', invert=False, mode="exclude", resolutionDiv=1, **kwargs):
        """Exclude areas based off the features in a vector datasource
        
        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource defining the criteria values for each location
            
        where : str
            A filtering statement to apply to the datasource before the indication
            * This is an SQL like statement which can operate on features in the 
              datasource
            * For tips, see "http://www.gdal.org/ogr_sql.html"
            * For example...
              - If the datasource had features which each have an attribute 
                called 'type' and only features with the type "protected" are 
                wanted, the correct statement would be: 
                    where="type='protected'"
        
        buffer : float; optional
            A buffer region to add around the indicated pixels
            * Units are in the RegionMask's srs

        bufferMethod : str; optional
            An indicator determining the method to use when buffereing
            * Options are: 'geom' and 'area'
            * If 'geom', the function will attempt to grow each of the geometries
              directly using the ogr library
              - This can fail sometimes when the geometries are particularly 
                complex or if some of the geometries are not valid (as in, they 
                have self-intersections)
            * If 'area', the function will first rasterize the raw geometries and
              will then apply the buffer to the indicated pixels
              - This is the safer option although is not as accurate as the 'geom'
                option since it does not capture the exact edges of the geometries
              - This method can be made more accurate by increasing the 
                'resolutionDiv' input
        
        resolutionDiv : int; optional
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details

        invert: bool; optional
            If True, flip indications

        mode: string; optional
            * If 'exclude', then the indicated pixels are subtracted from the 
              current availability matrix
            * If 'include', then the indicated pixel are added back into the
              availability matrix

        kwargs
            * All other keyword arguments are passed on to a call to 
              geokit.RegionMask.indicateFeatures
        
        """
        if isinstance(source, PriorSource):
            edgeI = kwargs.pop("edgeIndex", np.argwhere(source.edges==source.typicalExclusion))
            source = source.generateVectorFromEdge( s.region.extent, edgeIndex=edgeI )

        # Indicate on the source
        areas = (s.region.indicateFeatures(source, where=where, buffer=buffer, resolutionDiv=resolutionDiv, 
                                           bufferMethod=bufferMethod, applyMask=False, **kwargs)*100).astype(np.uint8)
        
        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min([s._availability, areas if invert else 100-areas],axis=0)
        elif mode == "include":
            s._availability = np.max([s._availability, 100-areas if invert else areas],axis=0)
            s._availability[~s.region.mask] = 0
        else:
            raise GlaesError("mode must be 'exclude' or 'include'")

    def excludePrior(s, prior, value=None, buffer=None, invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the values in one of the Prior data sources

        * The Prior datasources are currently only defined over Europe
        * All Prior datasources are defined in the EPSG3035 projection system 
          with 100x100 meter resolution
        * For each call to excludePrior, a temporary raster datasource is generated
          around the ExclusionCalculator's region, after which a call to 
          ExclusionCalculator.excludeRasterType is made, therefore all the same 
          inputs apply here as well

        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource defining the criteria values for each location
            
        value : tuple or numeric 
            The exact value, or value range to exclude
            * If Numeric, should be The exact value to exclude
                * Generally this should only be done when the raster datasource 
                  contains integer values, otherwise a range of values should be 
                  used to avoid float comparison errors
            * If ( Numeric, Numeric ), the low and high boundary describing the 
              range of values to exclude
                * If either boundary is given as None, then it is interpreted as 
                  unlimited

        buffer : float; optional
            A buffer region to add around the indicated pixels
            * Units are in the RegionMask's srs
            * The buffering occurs AFTER the indication and warping step and
              so it may not represent the original dataset exactly
              - Buffering can be made more accurate by increasing the 
                'resolutionDiv' input

        invert: bool; optional
            If True, flip indications

        mode: string; optional
            * If 'exclude', then the indicated pixels are subtracted from the 
              current availability matrix
            * If 'include', then the indicated pixel are added back into the
              availability matrix

        kwargs
            * All other keyword arguments are passed on to a call to 
              geokit.RegionMask.indicateValues
        """

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
                warn("It is advisable to exclude by a value range instead of a singular value when using the Prior datasets", UserWarning)

        # Project to 'index space'
        try:
            v1, v2 = value
            if not v1 is None:
                v1 = np.interp(v1, prior._values_wide, np.arange(prior._values_wide.size)) 
            if not v2 is None:
                v2 = np.interp(v2, prior._values_wide, np.arange(prior._values_wide.size)) 

            value = (v1,v2)
        except TypeError:
            if not value == 0:
                value = np.interp(value, prior._values_wide, np.arange(prior._values_wide.size)) 
        #source = prior.generateRaster( s.region.extent,)

        # Call the excluder
        s.excludeRasterType( prior.path, value=value, invert=invert, mode=mode, **kwargs)

    def excludeRegionEdge(s, buffer):
        """Exclude some distance from the region's edge
        
        Parameters:
        -----------
        buffer : float
                A buffer region to add around the indicated pixels
                * Units are in the RegionMask's srs
        """
        s.excludeVectorType(s.region.vector, buffer=-buffer, invert=True)

    def shrinkAvailability(s, dist, threshold=50):
    	"""Shrinks the current availability by a given distance in the given SRS"""
    	geom = gk.geom.polygonizeMask(s._availability>=threshold, bounds=s.region.extent.xyXY, srs=s.region.srs, flat=False)
    	geom = [g.Buffer(-dist) for g in geom]
    	newAvail = (s.region.indicateGeoms(geom)*100).astype(np.uint8)
    	s._availability = newAvail

    def pruneIsolatedAreas(s, minSize, threshold=50):
        """Removes contiguous areas which are smaller than 'minSize'

        * minSize is given in units of the calculator's srs
        """ 
        # Create a vector file of geometries larger than 'minSize'
        geoms = gk.geom.polygonizeMask( s._availability>=threshold, bounds=s.region.extent.xyXY, srs=s.region.srs, flat=False)
        geoms = list(filter( lambda x: x.Area()>=minSize, geoms ))
        vec = gk.core.util.quickVector(geoms)

        # Replace current availability matrix
        s._availability = s.region.indicateFeatures(vec, applyMask=False).astype(np.uint8 )*100

    