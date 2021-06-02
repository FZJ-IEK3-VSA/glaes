import geokit as gk
import re
import numpy as np
from os.path import isfile
from collections import namedtuple
from warnings import warn
import pandas as pd
import hashlib
from osgeo import gdal


from .util import GlaesError, glaes_logger
from .priors import Priors, PriorSource

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
        "access_distance": (5000, None),
        "agriculture_proximity": (None, 50),
        "agriculture_arable_proximity": (None, 50),
        "agriculture_pasture_proximity": (None, 50),
        "agriculture_permanent_crop_proximity": (None, 50),
        "agriculture_heterogeneous_proximity": (None, 50),
        "airfield_proximity": (None, 3000),
        "airport_proximity": (None, 5000),
        "connection_distance": (10000, None),
        "dni_threshold": (None, 3.0),
        "elevation_threshold": (1800, None),
        "ghi_threshold": (None, 3.0),
        "industrial_proximity": (None, 300),
        "lake_proximity": (None, 400),
        "mining_proximity": (None, 100),
        "ocean_proximity": (None, 1000),
        "power_line_proximity": (None, 200),
        "protected_biosphere_proximity": (None, 300),
        "protected_bird_proximity": (None, 1500),
        "protected_habitat_proximity": (None, 1500),
        "protected_landscape_proximity": (None, 500),
        "protected_natural_monument_proximity": (None, 1000),
        "protected_park_proximity": (None, 1000),
        "protected_reserve_proximity": (None, 500),
        "protected_wilderness_proximity": (None, 1000),
        "camping_proximity": (None, 1000),
        "touristic_proximity": (None, 800),
        "leisure_proximity": (None, 1000),
        "railway_proximity": (None, 150),
        "river_proximity": (None, 200),
        "roads_proximity": (None, 150),
        "roads_main_proximity": (None, 200),
        "roads_secondary_proximity": (None, 100),
        "sand_proximity": (None, 1000),
        "settlement_proximity": (None, 500),
        "settlement_urban_proximity": (None, 1000),
        "slope_threshold": (10, None),
        "slope_north_facing_threshold": (3, None),
        "wetland_proximity": (None, 1000),
        "waterbody_proximity": (None, 300),
        "windspeed_100m_threshold": (None, 4.5),
        "windspeed_50m_threshold": (None, 4.5),
        "woodland_proximity": (None, 300),
        "woodland_coniferous_proximity": (None, 300),
        "woodland_deciduous_proximity": (None, 300),
        "woodland_mixed_proximity": (None, 300)}

    def __init__(s, region, start_raster=None, srs=3035, pixelRes=100, where=None, padExtent=0, initialValue=True,
                 verbose=True, **kwargs):
        """Initialize the ExclusionCalculator

        Parameters:
        -----------
        region : str, ogr.Geometry, geokit.RegionMask
            The regional definition for the land eligibility analysis
            * If given as a string, must be a path to a vector file.
              - NOTE: Either the vector file should contain exactly 1 feature,
                a "where" statement should be used to select a specific feature,
                or "limitOne=False" should be specified (to join all features)
            * If given as a RegionMask, it is taken directly despite other
              arguments

        srs : str, Anything acceptable to geokit.srs.loadSRS()
            The srs context of the generated RegionMask object
            * The default srs EPSG3035 is only valid for a European context
            * If an integer is given, it is treated as an EPSG identifier
              - Look here for options: http://spatialreference.org/ref/epsg/
              * Only effective if 'region' is a path to a vector
            * If a string is specified, then a new srs can be automatically
              generated using the Lambert Azimuthal Equal Area projection type
              - Must follow the form "LAEA" or "LAEA:<lat>,<lon>" where <lat> 
                and <lon> are the latitute and of the center point of the new 
                projection
              - Specifying "LAEA" instructs the constructor to determine X and Y
                automatically from the given 'region' input
                - NOTE: Only works when the 'region' input is an ogr.Geometry or
                  a path to a vector file

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


        initialValue : bool or str; optional
            Used to control the initial state of the ExclusionCalculator
            * If "True", the region is assumed to begin as fully available
            * If "False", the region is assumed to begin as completely unavailable
            * If a path to a ".tif" file is given, then the ExclusionCalculator is initialized
                by warping (using the 'near' algorithm) from the given raster, and excluding 
                pixels with a value of 0 

        kwargs:
            * Keyword arguments are passed on to a call to geokit.RegionMask.load
            * Only take effect when the 'region' argument is a string

        """
        # Set simple flags
        s.verbose = verbose

        # Create spatial reference system (but only if a RegionMask isnt already given)
        if not isinstance(region, gk.RegionMask) and isinstance(srs, str) and srs[0:4] == "LAEA":
            import osr
            import ogr
            if len(srs) > 4:  # A center point was given
                m = re.compile("LAEA:([0-9.-]+),([0-9.-]+)").match(srs)
                if m is None:
                    raise RuntimeError(
                        "SRS string is not understandable. Must be parsable with: 'LAEA:([0-9.-]+),([0-9.-]+)'")
                center_y, center_x = map(float, m.groups())

            else:  # A center point should be determined
                if isinstance(region, ogr.Geometry):
                    if not region.GetSpatialReference().IsSame(gk.srs.EPSG4326):
                        region = gk.geom.transform(
                            region, toSRS=gk.srs.EPSG4326)
                    centroid = region.Centroid()
                    center_x = centroid.GetX()
                    center_y = centroid.GetY()
                elif isinstance(region, str):
                    _ext = gk.Extent.fromVector(region, where=where)

                    center_x = (_ext.xMin + _ext.xMax) / 2
                    center_y = (_ext.yMin + _ext.yMax) / 2
                    if not _ext.srs.IsSame(gk.srs.EPSG4326):
                        center_x, center_y, _ = gk.srs.xyTransform(
                            (center_x, center_y), fromSRS=_ext.srs, toSRS=gk.srs.EPSG4326)
                else:
                    raise RuntimeError(
                        "Automatic center determination is only possible when the 'region' input is an ogr.Geometry Object or a path to a vector file")

            srs = osr.SpatialReference()
            srs.ImportFromProj4(
                '+proj=laea +lat_0={} +lon_0={} +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'.format(
                    center_y, center_x))

        # load the region
        s.region = gk.RegionMask.load(
            region, start_raster=start_raster, srs=srs, pixelRes=pixelRes, where=where, padExtent=padExtent, **kwargs)
        s.srs = s.region.srs
        s.maskPixels = s.region.mask.sum()

        # Make the total availability matrix
        s._availability = np.array(s.region.mask, dtype=np.uint8) * 100
        s._availability_per_criterion = np.array(s.region.mask, dtype=np.uint8) * 100

        if initialValue == True:
            pass
        elif initialValue == False:
            s._availability *= 0
        elif isinstance(initialValue, str):
            assert isfile(initialValue)
            s._availability = np.array(s.region.mask, dtype=np.uint8) * 100
            s.excludeRasterType(initialValue, value=0)
        else:
            raise ValueError('initialValue "{}" is not known'.format(initialValue))

        # Make a list of item coords
        s.itemCoords = None
        s._itemCoords = None
        s._areas = None

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

        meta = {
            "description": "The availability of each pixel",
            "units": "percent-available"
        }

        data = s.availability
        if not threshold is None:
            data = (data >= threshold).astype(np.uint8) * 100

        data[~s.region.mask] = 255
        return s.region.createRaster(output=output, data=data,
                                     noData=255, meta=meta, **kwargs)

    def draw(s, ax=None, goodColor=(255/255, 255/255, 255/255), excludedColor=(2/255, 61/255, 107/255), itemsColor=(51/255, 153/255, 255/255), legend=True, legendargs={"loc": "lower left"}, srs=None, dataScalingFactor=1, geomSimplificationFactor=None, german=False, **kwargs):
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

        itemsColor: A matplotlib color
            The color to apply to predicted items. Default is black.

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

        german: bool
            If true legend will be in German

        **kwargs:
            All keyword arguments are passed on to a call to geokit.drawImage

        Returns:
        --------
        matplotlib axes object


        """
        # import some things
        from matplotlib.colors import LinearSegmentedColormap

        # First draw the availability matrix
        b2g = LinearSegmentedColormap.from_list(
            'bad_to_good', [excludedColor, goodColor])

        if ax is None and not "figsize" in kwargs:
            ratio = s.region.mask.shape[1] / s.region.mask.shape[0]
            kwargs["figsize"] = (8 * ratio * 1.2, 8)

        kwargs["cmap"] = kwargs.get("cmap", b2g)
        kwargs["cbar"] = kwargs.get("cbar", False)
        kwargs["vmin"] = kwargs.get("vmin", 0)
        kwargs["vmax"] = kwargs.get("vmax", 100)
        kwargs["cbarTitle"] = kwargs.get("cbarTitle", "Pixel Availability")

        if srs is None:
            kwargs["topMargin"] = kwargs.get("topMargin", 0.01)
            kwargs["bottomMargin"] = kwargs.get("bottomMargin", 0.02)
            kwargs["rightMargin"] = kwargs.get("rightMargin", 0.01)
            kwargs["leftMargin"] = kwargs.get("leftMargin", 0.02)
            kwargs["hideAxis"] = kwargs.get("hideAxis", True)

            axh1 = s.region.drawImage(
                s.availability,
                ax=ax,
                drawSelf=False,
                scaling=dataScalingFactor,
                **kwargs)

            srs = s.region.srs
        else:
            mat = s._availability.copy()
            no_data = 255
            mat[~s.region.mask] = no_data
            availability_raster = s.region.createRaster(data=mat, noData=no_data)
            axh1 = gk.drawRaster(
                availability_raster,
                ax=ax,
                srs=srs,
                cutlineFillValue=no_data,
                **kwargs)

        # # Draw the mask to blank out the out of region areas
        # w2a = LinearSegmentedColormap.from_list('white_to_alpha',[(1,1,1,1),(1,1,1,0)])
        # axh2 = s.region.drawImage( s.region.mask, ax=axh1, drawSelf=False, cmap=w2a, cbar=False)

        # Draw the Regional geometry
        axh3 = gk.drawGeoms(
            s.region.geometry,
            fc='None',
            srs=srs,
            ax=axh1,
            linewidth=2,
            simplificationFactor=geomSimplificationFactor)

        # Draw Points, maybe?
        if not s._itemCoords is None:
            points = s._itemCoords
            if not srs.IsSame(s.region.srs):
                points = gk.srs.xyTransform(
                    points,
                    fromSRS=s.region.srs,
                    toSRS=srs,
                    outputFormat="xy"
                )

                points = np.column_stack([points.x, points.y])
            axh1.ax.plot(points[:, 0], points[:, 1], color=itemsColor, marker='o', linestyle='None')

        # Draw Areas, maybe?
        if not s._areas is None:
            gk.drawGeoms(
                s._areas,
                srs=srs,
                ax=axh1,
                fc='None',
                ec="k",
                linewidth=1,
                simplificationFactor=None)

        # Make legend?
        if legend:
            from matplotlib.patches import Patch
            p = s.percentAvailable
            a = s.region.mask.sum(dtype=np.int64) * \
                s.region.pixelWidth * s.region.pixelHeight
            areaLabel = s.region.srs.GetAttrValue("Unit").lower()
            if areaLabel == "metre" or areaLabel == "meter":
                a = a / 1000000
                areaLabel = "km"
            elif areaLabel == "feet" or areaLabel == "foot":
                areaLabel = "ft"
            elif areaLabel == "degree":
                areaLabel = "deg"

            if a < 0.001:
                regionLabel = "{0:.3e} ${1}^2$".format(a, areaLabel)
            elif a < 0:
                regionLabel = "{0:.4f} ${1}^2$".format(a, areaLabel)
            elif a < 1000:
                regionLabel = "{0:.2f} ${1}^2$".format(a, areaLabel)
            else:
                regionLabel = "{0:,.0f} ${1}^2$".format(a, areaLabel)

            patches = [
                Patch(ec="k", fc="None", linewidth=3, label=regionLabel),
                Patch(color=excludedColor, label=f"{'Ausgeschlossen' if german else 'Excluded'}: %.2f%%" % (100 - p)),
                Patch(color=goodColor, label=f"{'Verfügbar' if german else 'Eligible'}: %.2f%%" % (p)),
            ]
            if not s._itemCoords is None:
                h = axh1.ax.plot([], [], color=itemsColor, marker='o', linestyle='None', label="{}: {:,d}".format(
                    'Elemente' if german else 'Items', s._itemCoords.shape[0]))
                patches.append(h[0])

            _legendargs = dict(loc="lower right", fontsize=14)
            _legendargs.update(legendargs)
            axh1.ax.legend(handles=patches, **_legendargs)

        # Done!!
        return axh1.ax

    def drawWithSmopyBasemap(s, zoom=4, excludedColor=(2/255, 61/255, 107/255, 128/255), ax=None, figsize=None, smopy_kwargs=dict(attribution="© OpenStreetMap contributors", attribution_size=12), **kwargs):
        """
        This wrapper around the original ExclusionCalculator.draw function adds a basemap bethind the drawn eligibility map

        NOTE:
        * The basemap is drawn using the Smopy python package. See here: https://github.com/rossant/smopy
        * Be careful to adhere to the usage guidelines of the chosen tile source
            - By default, this source is OSM. See here: https://wiki.openstreetmap.org/wiki/Tile_servers

        !IMPORTANT! If you will publish any images drawn with this method, it's likely that the tile source
        will require an attribution to be written on the image. For example, if using OSM tile (the default),
        you have to write "© OpenStreetMap contributors" clearly on the map. But this is different for each
        tile source!

        Tip:
        * Start with a low zoom value (e.g. 4) and zoom in until you find something reasonable

        Parameters:
        -----------
            zoom : int
                The desired zoom level of the basemap
                * Should be between 1 - 20
                * The higher the number, the more you're zooming in
                * Note that, for each increase in the zoom level, the numer of tiles 
                    fetched increases by a factor of 4

            excludeColor : (r, g, b, a)
                The color to give to excluded points

            ax : matplotlib axes 
                The axes to draw on
                * If not given, one will be generated

            figsize : (width, height)
                The size of the figure to draw
                * Is only effective when ax=None

            smopy_kwargs : dict
                * Keyword arguments to pass on to gk.raster.drawSmopyMap

            kwargs
                * All other keyword arguments are passed on to ExclusionCalcularot.draw

        Returns:
        --------

        matplotlib axes
        """

        if ax is None:
            import matplotlib.pyplot as plt

            if figsize is None:
                ratio = s.region.mask.shape[1] / s.region.mask.shape[0]
                plt.figure(figsize=(8 * ratio * 1.2, 8))
            else:
                plt.figure(figsize=figsize)

            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax, srs, bounds = s.region.extent.drawSmopyMap(zoom, ax=ax, **smopy_kwargs)
        s.draw(ax=ax, srs=srs, goodColor=[0, 0, 0, 0], excludedColor=excludedColor, **kwargs)

        return ax

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
        return s._availability.sum(dtype=np.int64) / s.region.mask.sum()

    # TODO: Push Git
    @property
    def percentAvailablePerCriterion(s):
        """The percent of the region which remains available"""
        return s._availability_per_criterion.sum(dtype=np.int64) / s.region.mask.sum()

    @property
    def clearPercentAvailablePerCriterion(s):
        """The percent of the region which remains available"""
        s._availability_per_criterion = np.array(s.region.mask, dtype=np.uint8) * 100
        return

    @property
    def areaAvailable(s):
        """The area of the region which remains available
            * Units are defined by the srs used to initialize the ExclusionCalculator"""
        return s._availability[s.region.mask].sum(dtype=np.int64) * s.region.pixelWidth * s.region.pixelHeight / 100

    def _hasEqualContext(self, source):
        """
        Internal function which checks if a given raster source has the same context as 
        the ExclusionCalculator. This checks SRS, extent, and pixel resolution
        """
        if not isfile(source) or not gk.util.isRaster(source):
            # print("Is not a raster!")
            return False

        ri_extent = gk.Extent.fromRaster(source)
        if not ri_extent == self.region.extent:
            # print("Extent mismatch!")
            return False

        if not ri_extent.srs.IsSame(self.srs):
            # print("SRS mismatch!")
            return False

        ri = gk.raster.rasterInfo(source)
        if not np.isclose(ri.pixelWidth, self.region.pixelWidth):
            # print("pixelWidth mismatch!")
            return False

        if not np.isclose(ri.pixelHeight, self.region.pixelHeight):
            # print("pixelHeight mismatch!")
            return False

        return True

    # General excluding functions
    def excludeRasterType(s, source, value=None, buffer=None, resolutionDiv=1, intermediate=None, prewarp=False,
                          invert=False, mode="exclude", **kwargs):
        """Exclude areas based off the values in a raster datasource

        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource defining the criteria values for each location

        value : numeric, tuple, iterable, or str
            The exact value, or value range to exclude
            * If Numeric, should be The exact value to exclude
                * Generally this should only be done when the raster datasource
                  contains integer values, otherwise a range of values should be
                  used to avoid float comparison errors
            * If ( Numeric, Numeric ), the low and high boundary describing the
              range of values to exclude
                * If either boundary is given as None, then it is interpreted as
                  unlimited
            * If any other iterable : The list of exact values to accept 
            * If str : The formatted set of elements to accept 
              - Each element in the set is seperated by a "," 
              - Each element must be either a singular numeric value, or a range
              - A range element begins with either "[" or "(", and ends with either "]" or ")"
                and should have an '-' in between
                - "[" and "]" imply inclusivity
                - "(" and ")" imply exclusivity
                - Numbers on either side can be omitted, implying no limit on that side
                - Examples:
                  - "[1-5]" -> Indicate values from 1 up to 5, inclusively
                  - "[1-5)" -> Indicate values from 1 up to 5, but not including 5
                  - "(1-]"  -> Indicate values above 1 (but not including 1) up to infinity
                  - "[-5]"  -> Indicate values from negative infinity up to and including 5
                  - "[-]"   -> Indicate values from negative infinity to positive infinity (dont do this..)
              - All whitespaces will be ignored (so feel free to use them as you wish)
              - Example:
                - "[-2),[5-7),12,(22-26],29,33,[40-]" will indicate all of the following:
                  - Everything below 2, but not including 2
                  - Values between 5 up to 7, but not including 7
                  - 12
                  - Values above 22 up to and including 26
                  - 29
                  - 33
                  - Everything above 40, including 40

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


        intermediate : path, optional
            Path to an intermediate result raster file for this set of function arguments.
            When not None, the ExclusionCalculator will check if data from the intermediate
            input file can be used to cache the exclusion calculation result of this criterion.            
            * If path to intermediate file exists, metadata (buffer, resolution,
              prewarp, invert, mode, kwargs will be compared to current arguments) 
            * If metadata matches, intermediate file will be excluded instead of new
              calculation
            * If metadata does not match, exclusion will be calculated anew from source file
              and new intermediate file with resulting exclusion area is generated at this path.
            When None, the exclusion will be calculated anew for the given values in any case.

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
        # Perform check for intermediate file
        if intermediate is not None:
            if isinstance(source, gdal.Dataset):
                h = hashlib.sha256(source.ReadAsArray().tobytes())
                source_id = "Memory:" + h.hexdigest()
            else:
                source_id = str(source)

            # create dictionary of function arguments to compare against metadata
            metadata = {
                'AREA_OR_POINT': 'Area',
                'exclusion_type': "Raster",
                'source': source_id,
                'value': str(value),
                'buffer': str(buffer),
                'resolutionDiv': str(resolutionDiv),
                'prewarp': str(prewarp),
                'invert': str(invert),
                'mode': str(mode)
            }

            for k, v in kwargs.items():
                metadata[k] = v

        # check if we can apply the intermediate file
        if intermediate is not None and \
                isfile(intermediate) and \
                gk.raster.rasterInfo(intermediate).meta == metadata and \
                s._hasEqualContext(intermediate):

            if s.verbose and intermediate is not None:
                 glaes_logger.info("Applying intermediate exclusion file: " + intermediate)

            indications = gk.raster.extractMatrix(intermediate)

        else:  # We need to compute the exclusion
            if s.verbose and intermediate is not None:
                glaes_logger.info("Computing intermediate exclusion file: " + intermediate)
                if isfile(intermediate):
                    glaes_logger.warning("Overwriting previous intermediate exclusion file: " + intermediate)

            # Do prewarp, if needed
            if prewarp:
                prewarpArgs = dict(resampleAlg="bilinear")
                if isinstance(prewarp, str):
                    prewarpArgs["resampleAlg"] = prewarp
                elif isinstance(prewarp, dict):
                    prewarpArgs.update(prewarp)

                source = s.region.warp(source, returnMatrix=False, **prewarpArgs)

            # Indicate on the source
            indications = (
                    s.region.indicateValues(
                        source,
                        value,
                        buffer=buffer,
                        resolutionDiv=resolutionDiv,
                        forceMaskShape=True,
                        applyMask=False,
                        **kwargs) * 100
            ).astype(np.uint8)

            # check if intermediate file usage is selected and create intermediate raster file with exlcusion arguments as metadata
            if intermediate is not None:
                s.region.createRaster(output=intermediate, data=indications, meta=metadata)

        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min(
                [s._availability, indications if invert else 100 - indications], axis=0)

            s._availability_per_criterion = np.min(
                [s._availability_per_criterion, indications if invert else 100 - indications], axis=0)
        elif mode == "include":
            s._availability = np.max(
                [s._availability, 100 - indications if invert else indications], axis=0)
            s._availability[~s.region.mask] = 0

            s._availability_per_criterion = np.max(
                [s._availability_per_criterion, 100 - indications if invert else indications], axis=0)
            s._availability_per_criterion[~s.region.mask] = 0
        else:
            raise GlaesError("mode must be 'exclude' or 'include'")

    def excludeVectorType(s, source, where=None, buffer=None, bufferMethod='geom', invert=False, mode="exclude",
                          resolutionDiv=1, intermediate=None, **kwargs):
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


        intermediate : path, optional
            Path to the intermediate results tif file for this set of function arguments.
            When not None, the exclusioncalculator will check if data from intermediate
            input files can be used to save calculation of this particular exclusion criterion.            
            * If path to intermediate file exists, metadata (buffer, resolution,
              prewarp, invert, mode, kwargs will be compared to current arguments) 
            * If metadata matches, intermediate file will be excluded instead of new
              calculation
            * If metadata does not match, exclusion will be calculated anew from source file
              and new intermediate file with resulting exclusion area is generated at this path.
            When None, the exclusion will be calculated anew for the given values in any case.

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
        # Perform check for intermediate file
        if intermediate is not None:
            # TODO: Find a way to get a hash signiture of an in-memory vector file
            # if isinstance(source, gdal.Dataset):
            #     h = hashlib.sha256(source.ReadAsArray().tobytes())
            #     source_id = "Memory:" + h.hexdigest()
            # else:
            #     source_id = str(source)
            source_id = str(source)

            # create dictionary of function arguments to compare against metadata
            metadata = {
                'AREA_OR_POINT': 'Area',
                'exclusion_type': "Vector",
                'source': source_id,
                'where': str(where),
                'buffer': str(buffer),
                'bufferMethod': str(bufferMethod),
                'invert': str(invert),
                'resolutionDiv': str(resolutionDiv),
                'mode': str(mode)
            }

            for k, v in kwargs.items():
                metadata[k] = v

        # check if we can apply the intermediate file
        if intermediate is not None and \
                isfile(intermediate) and \
                gk.raster.rasterInfo(intermediate).meta == metadata and \
                s._hasEqualContext(intermediate):

            if s.verbose and intermediate is not None:
                glaes_logger.info("Applying intermediate exclusion file: " + intermediate)

            indications = gk.raster.extractMatrix(intermediate)

        else:  # We need to compute the exclusion
            if s.verbose and intermediate is not None:
                glaes_logger.info("Computing intermediate exclusion file: " + intermediate)
                if isfile(intermediate):
                    glaes_logger.warning("Overwriting previous intermediate exclusion file: " + intermediate, UserWarning)

            if isinstance(source, PriorSource):
                edgeI = kwargs.pop("edgeIndex", np.argwhere(
                    source.edges == source.typicalExclusion))
                source = source.generateVectorFromEdge(
                    s.region.extent, edgeIndex=edgeI)

            # Indicate on the source
            indications = (
                    s.region.indicateFeatures(
                        source,
                        where=where,
                        buffer=buffer,
                        resolutionDiv=resolutionDiv,
                        bufferMethod=bufferMethod,
                        applyMask=False,
                        forceMaskShape=True,
                        **kwargs) * 100
            ).astype(np.uint8)

            # check if intermediate file usage is selected and create intermediate raster file with exlcusion arguments as metadata
            if intermediate is not None:
                s.region.createRaster(output=intermediate, data=indications, meta=metadata)

        # exclude the indicated area from the total availability
        if mode == "exclude":
            s._availability = np.min(
                [s._availability, indications if invert else 100 - indications], axis=0)

            s._availability_per_criterion = np.min(
                [s._availability_per_criterion, indications if invert else 100 - indications], axis=0)
        elif mode == "include":
            s._availability = np.max(
                [s._availability, 100 - indications if invert else indications], axis=0)
            s._availability[~s.region.mask] = 0

            s._availability_per_criterion = np.max(
                [s._availability_per_criterion, 100 - indications if invert else indications], axis=0)
            s._availability_per_criterion[~s.region.mask] = 0
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
        if isinstance(prior, str):
            prior = Priors[prior]

        if not isinstance(prior, PriorSource):
            raise GlaesError(
                "'prior' input must be a Prior object or an associated string")

        # try to get the default value if one isn't given
        if value is None:
            try:
                value = s.typicalExclusions[prior.displayName]
            except KeyError:
                raise GlaesError(
                    "Could not find a default exclusion set for %s" % prior.displayName)

        # Check the value input
        if isinstance(value, tuple):

            # Check the boundaries
            if not value[0] is None:
                prior.containsValue(value[0], True)
            if not value[1] is None:
                prior.containsValue(value[1], True)

            # Check edges
            if not value[0] is None:
                prior.valueOnEdge(value[0], True)
            if not value[1] is None:
                prior.valueOnEdge(value[1], True)

        else:
            if not value == 0:
                warn(
                    "It is advisable to exclude by a value range instead of a singular value when using the Prior datasets",
                    UserWarning)

        # Project to 'index space'
        try:
            v1, v2 = value
            if not v1 is None:
                v1 = np.interp(v1, prior._values_wide,
                               np.arange(prior._values_wide.size))
            if not v2 is None:
                v2 = np.interp(v2, prior._values_wide,
                               np.arange(prior._values_wide.size))

            value = (v1, v2)
        except TypeError:
            if not value == 0:
                value = np.interp(value, prior._values_wide,
                                  np.arange(prior._values_wide.size))
        # source = prior.generateRaster( s.region.extent,)

        # Call the excluder
        s.excludeRasterType(prior.path, value=value,
                            invert=invert, mode=mode, **kwargs)

    def excludeRegionEdge(s, buffer, **kwargs):
        """Exclude some distance from the region's edge

        Parameters:
        -----------
        buffer : float
                A buffer region to add around the indicated pixels
                * Units are in the RegionMask's srs
        """
        s.excludeVectorType(s.region.vector, buffer=-buffer, invert=True, **kwargs)

    def excludeSet(s, exclusion_set, filterSourceLists=True, filterMissingError=True, **paths):
        """
        Iteratively exclude a set of exclusion constraints

        Parameters:
        -----------
            exclusion_set : pandas.DataFrame
                The rows of this dataframe dictate the exclusions which are performed
                in the given order

                * The following columns names are used:
                    - 'name'  -> The name of the contraint
                    - 'type'  -> The type of the contraint ['prior', 'raster', or 'vector']
                    - 'value' -> The vale/where-statement to use
                    - 'buffer'-> The buffer value (if not given, 0 is assumed)
                    - 'mode'  -> The mode (if not given, 'exclude' is assumed)
                    - 'invert'-> The inversion state (if not given, False is assumed)

                * For raster or prior types, 'value' can be given in several ways:
                    - "XXX"      -> translates to value=XXX. i.e. "exclude exactly XXX"
                    - "XXX-YYY"  -> translates to value=(XXX,YYY). i.e. "exclude between XXX and YYY"
                    - "None-XXX" -> translates to value=(None,XXX). i.e. "everything below XXX"
                    - "-XXX"     -> also translates to value=(None,XXX)
                    - "XXX-None" -> translates to value=(XXX, None). i.e. "everything above XXX"
                    - "XXX-"     -> also translates to value=(XXX, None)

                * For raster types, see the note in ExclusionCalculator.excludeRasterType regarding 
                    passing string-type value inputs
                    - For example, "[-2),[5-7),12,(22-26],29,33,[40-]" will indicate pixels with values:
                        - Below 2, but not including 2
                        - Between 5 up to 7, but not including 7
                        - Equal to 12
                        - Above 22 up to and including 26
                        - Equal to 29
                        - Equal to 33
                        - Above 40, including 40

                * For vector types, the 'value' is just the SQL-like where statement

            filterSourceLists : bool
                If True, then paths to lists of vector files or raster files will be filtered
                using self.region.Extent.filterSources(...)

            filterMissingError : bool
                If True, then if a path is given which does not exist, a RuntimError is raised. Otherwise
                    a user warning is given.
                Only effective when `filterSourceLists` is True

            verbose : bool
                If True, progress statements are given

            **paths
                All extra arguments should correspond to the paths on disk for each of the
                'name's specified in the exclusion_set input
        """
        verbose = s.verbose
        exclusion_set = exclusion_set.copy()

        # Make sure inputs are okay
        assert isinstance(exclusion_set, pd.DataFrame)
        assert "name" in exclusion_set.columns
        assert "type" in exclusion_set.columns
        assert "value" in exclusion_set.columns

        if not "buffer" in exclusion_set.columns:
            exclusion_set['buffer'] = 0
        if not "exclusion_mode" in exclusion_set.columns:
            exclusion_set['exclusion_mode'] = 'exclude'
        if not "invert" in exclusion_set.columns:
            exclusion_set['invert'] = False
        if not "resolutionDiv" in exclusion_set.columns:
            exclusion_set['resolutionDiv'] = 1

        for p in paths:
            assert isinstance(p, str)

        # Exclude rows one by one
        for i, row in exclusion_set.iterrows():
            if np.isnan(row.buffer) or row.buffer == 0:
                buffer = None
            else:
                buffer = float(row.buffer)

            if row.type == "prior":
                if verbose:
                    glaes_logger.info("Excluding Prior {} with value {}, buffer {}, mode {}, and invert {} ".format(
                        row['name'],
                        row.value,
                        buffer,
                        row.exclusion_mode,
                        row.invert,
                    ))

                if isinstance(row.value, str):
                    try:
                        value_low, value_high = row.value.split("-")
                        value_low = None if value_low == "None" else float(value_low)
                        value_high = None if value_high == "None" else float(value_high)

                        value = value_low, value_high
                    except:
                        value = float(value)

                s.excludePrior(
                    prior=row['name'],
                    value=value,
                    buffer=buffer,
                    invert=row.invert,
                    mode=row.exclusion_mode)

            elif row.type == "raster":
                value = str(row.value)
                if verbose:
                    glaes_logger.info("Excluding Raster {} with value {}, buffer {}, mode {}, and invert {} ".format(
                        row['name'],
                        value,
                        buffer,
                        row.exclusion_mode,
                        row.invert
                    ))

                sources = paths[row['name']]
                if gk.util.isRaster(sources):
                    sources = [sources, ]

                if filterSourceLists:
                    sources = list(s.region.extent.filterSources(sources, error_on_missing=filterMissingError))
                    if verbose and len(sources) == 0:
                        glaes_logger.info("  No suitable sources in extent! ")

                for source in sources:
                    s.excludeRasterType(
                        source=source,
                        value=value,
                        buffer=buffer,
                        resolutionDiv=row.resolutionDiv,
                        prewarp=False,
                        invert=row.invert,
                        mode=row.exclusion_mode, )

            elif row.type == "vector":
                if verbose:
                    glaes_logger.info("Excluding Vector {} with where-statement \"{}\", buffer {}, mode {}, and invert {} ".format(
                            row['name'],
                            row.value,
                            buffer,
                            row.exclusion_mode,
                            row.invert
                        ))

                if row.value == "" or row.value == "None":
                    value = None
                else:
                    value = row.value

                sources = paths[row['name']]
                if gk.util.isVector(sources):
                    sources = [sources, ]

                if filterSourceLists:
                    sources = list(s.region.extent.filterSources(sources, error_on_missing=filterMissingError))
                    if verbose and len(sources) == 0:
                        glaes_logger.info("  No suitable sources in extent! ")

                # print(sources)
                for source in sources:
                    # print("SOURCE: ", source)
                    s.excludeVectorType(
                        source=source,
                        where=value,
                        buffer=buffer,
                        resolutionDiv=row.resolutionDiv,
                        invert=row.invert,
                        mode=row.exclusion_mode)

        if verbose:
            glaes_logger.info("Done!")

    def shrinkAvailability(s, dist, threshold=50):
        """Shrinks the current availability by a given distance in the given SRS"""
        geom = gk.geom.polygonizeMask(
            s._availability >= threshold, bounds=s.region.extent.xyXY, srs=s.region.srs, flat=False)
        geom = [g.Buffer(-dist) for g in geom]
        newAvail = (s.region.indicateGeoms(geom) * 100).astype(np.uint8)
        s._availability = newAvail

    def pruneIsolatedAreas(s, minSize, threshold=50):
        """Removes contiguous areas which are smaller than 'minSize'

        * minSize is given in units of the calculator's srs
        """
        # Create a vector file of geometries larger than 'minSize'
        geoms = gk.geom.polygonizeMask(
            s._availability >= threshold, bounds=s.region.extent.xyXY, srs=s.region.srs, flat=False)
        geoms = list(filter(lambda x: x.Area() >= minSize, geoms))
        vec = gk.core.util.quickVector(geoms)

        # Replace current availability matrix
        s._availability = s.region.indicateFeatures(
            vec, applyMask=False).astype(np.uint8) * 100

    def distributeItems(s, separation, pixelDivision=5, threshold=50, maxItems=10000000, outputSRS=None, output=None, asArea=False, minArea=100000, maxAcceptableDistance=None, axialDirection=None, sepScaling=None, _voronoiBoundaryPoints=10, _voronoiBoundaryPadding=5, _stamping=True):
        """Distribute the maximal number of minimally separated items within the available areas

        Returns a list of x/y coordinates (in the ExclusionCalculator's srs) of each placed item

        Inputs:
            separation : The minimal distance between two items
                - float : The separation distance when axialDirection is None
                - (float, float) : The separation distance in the axial and transverse direction

            pixelDivision - int : The inter-pixel fidelity to use when deciding where items can be placed

            threshold : The minimal availability value to allow placing an item on

            maxItems - int : The maximal number of items to place in the area
                * Used to initialize a placement list and prevent using too much memory when the number of placements gets absurd

            outputSRS : The output SRS system to use
                * 4326 corresponds to regular lat/lon

            output : A path to an output shapefile

            axialDirection : The axial direction in degrees
                - float : The direction to apply to all points
                - np.ndarray : The directions at each pixel (must match availability matrix shape)
                - str : A path to a raster file containing axial directions

            maxAcceptableDistance : A maximum distance to allow between items
                - Computes a post-placement distance matrix for the located placements
                - If the placement's nearest neighbor is greater than `maxAcceptableDistance`, then it is removed
                - Input can be given as:
                    - Y[float] - Meaning that the nearest neighbor must be within the given distance, Y
                    - (Y1[int], Y2[float], ...) - Meaning that the first neighbor must be within a distance of Y1,
                      the second nearest neighbor should be within a distance of Y2, and so forth.
                - Ex.
                    - "maxAcceptableDistance=(1000, 2000, 3000)" means that if the nearest 3 neighbors are not within a 
                      distance of 1000, 2000, and 3000 meters, respectively, then the placement in question will be deleted

            sepScaling : An additional scaling factor which can be applied to each pixel
                - float : The scaling to apply to all points
                - np.ndarray : The scalings at each pixel (must match availability matrix shape)
                - str : A path to a raster file containing scaling factors
        """

        # TODO: CLEAN UP THIS FUNCTION BY REMOVING AREA DISTRIBUTION AND FILE SAVING, AND ASSOCIATED PARAMETERS

        # Preprocess availability
        workingAvailability = s._availability >= threshold
        if not workingAvailability.dtype == 'bool':
            raise s.GlaesError("Working availability must be boolean type")

        workingAvailability[~s.region.mask] = False

        # Handle a gradient file, if one is given
        if not axialDirection is None:
            if isinstance(axialDirection, str):  # Assume a path to a raster file is given
                axialDirection = s.region.warp(
                    axialDirection, resampleAlg='near')
            # Assume a path to a raster file is given
            elif isinstance(axialDirection, np.ndarray):
                if not axialDirection.shape == s.region.mask.shape:
                    raise GlaesError(
                        "axialDirection matrix does not match context")
            else:  # axialDirection should be a single value
                axialDirection = np.radians(float(axialDirection))

            useGradient = True
        else:
            useGradient = False

        # Read separation scaling file, if given
        if not sepScaling is None:
            if isinstance(sepScaling, str) or isinstance(sepScaling, gdal.Dataset):  # Assume a path to a raster file is given
                sepScaling = s.region.warp(sepScaling, resampleAlg='near', applyMask=False,)
                matrixScaling = True
            # Assume a numpy array is given
            elif isinstance(sepScaling, np.ndarray):
                if not sepScaling.shape == s.region.mask.shape:
                    raise GlaesError(
                        "sepScaling matrix does not match context")
                matrixScaling = True
            else:  # sepScaling should be a single value
                matrixScaling = False

        else:
            sepScaling = 1
            matrixScaling = False

        # Turn separation into pixel distances
        if not s.region.pixelWidth == s.region.pixelHeight:
            warn(
                "Pixel width does not equal pixel height. Therefore, the average will be used to estimate distances")
            pixelRes = (s.region.pixelWidth + s.region.pixelHeight) / 2
        else:
            pixelRes = s.region.pixelWidth

        if useGradient:
            try:
                sepA, sepT = separation
            except:
                raise GlaesError(
                    "When giving axial direction data, a separation tuple is expected")

            sepA, sepT = float(sepA), float(sepT)  # Cast as float to avoid integer overflow errors
            sepA = sepA * sepScaling / pixelRes
            sepT = sepT * sepScaling / pixelRes

            sepA2 = sepA ** 2
            sepT2 = sepT ** 2

            sepFloorA = np.maximum(sepA - np.sqrt(2), 0)
            sepFloorT = np.maximum(sepT - np.sqrt(2), 0)
            if not matrixScaling and (sepFloorA < 1 or sepFloorT < 1):
                raise GlaesError(
                    "Seperations are too small compared to pixel size")

            sepFloorA2 = np.power(sepFloorA, 2)
            sepFloorT2 = np.power(sepFloorT, 2)

            sepCeil = np.maximum(sepA, sepT) + 1

            stampFloor = min(sepFloorA2.min(), sepFloorT2.min()) if matrixScaling else min(sepFloorA2, sepFloorT2)
            stampWidth = int(np.ceil(np.sqrt(stampFloor)) + 1)
        else:
            separation = float(separation)  # Cast as float to avoid integer overflow errors
            separation = separation * sepScaling / pixelRes
            sep2 = np.power(separation, 2)
            sepFloor = np.maximum(separation - np.sqrt(2), 0)
            sepFloor2 = sepFloor**2
            sepCeil = separation + 1

            stampFloor = sepFloor2.min() if matrixScaling else sepFloor2
            stampWidth = int(np.ceil(np.sqrt(stampFloor)) + 1)

        if _stamping:
            _xy = np.linspace(-stampWidth, stampWidth, stampWidth * 2 + 1)
            _xs, _ys = np.meshgrid(_xy, _xy)

            # print("STAMP FLOOR:", stampFloor, stampWidth)
            stamp = (np.power(_xs, 2) + np.power(_ys, 2)
                     ) >= (stampFloor)  # (stampFloor - np.sqrt(stampFloor) * 2)

        if isinstance(sepCeil, np.ndarray) and sepCeil.size > 1:
            sepCeil = sepCeil.max()

        # Make geom list
        x = np.zeros((maxItems))
        y = np.zeros((maxItems))

        bot = 0
        cnt = 0

        # start searching
        yN, xN = workingAvailability.shape
        substeps = np.linspace(-0.5, 0.5, pixelDivision)
        # add a tiny bit to the left/top edge (so that the point is definitely in the right pixel)
        substeps[0] += 0.0001
        # subtract a tiny bit to the right/bottom edge for the same reason
        substeps[-1] -= 0.0001

        for yi in range(yN):
            # update the "bottom" value
            # find only those values which have a y-component greater than the separation distance
            tooFarBehind = yi - y[bot:cnt] > sepCeil
            if tooFarBehind.size > 0:
                # since tooFarBehind is boolean, argmin should get the first index where it is false
                bot += np.argmin(tooFarBehind)

            # print("yi:", yi, "   BOT:", bot, "   COUNT:",cnt)

            for xi in np.argwhere(workingAvailability[yi, :]):
                # point could have been excluded from a previous stamp
                if not workingAvailability[yi, xi]:
                    continue

                # Clip the total placement arrays
                xClip = x[bot:cnt]
                yClip = y[bot:cnt]
                if matrixScaling:
                    if useGradient:
                        _sepFloorA2 = sepFloorA2[yi, xi]
                        _sepFloorT2 = sepFloorT2[yi, xi]

                        if _sepFloorA2 < 1 or _sepFloorT2 < 1:
                            raise GlaesError(
                                "Seperations are too small compared to pixel size")

                        _sepA2 = sepA2[yi, xi]
                        _sepT2 = sepT2[yi, xi]
                    else:
                        _sepFloor2 = sepFloor2[yi, xi]
                        if _sepFloor2 < 1:
                            raise GlaesError(
                                "Seperations are too small compared to pixel size")
                        _sep2 = sep2[yi, xi]
                else:
                    if useGradient:
                        _sepFloorA2 = sepFloorA2
                        _sepFloorT2 = sepFloorT2
                        _sepA2 = sepA2
                        _sepT2 = sepT2
                    else:
                        _sepFloor2 = sepFloor2
                        _sep2 = sep2

                # calculate distances
                xDist = xClip - xi
                yDist = yClip - yi

                # Get the indicies in the possible range
                # pir => Possibly In Range,
                pir = np.argwhere(np.abs(xDist) <= sepCeil)
                # all y values should already be within the sepCeil

                # only continue if there are no points in the immediate range of the whole pixel
                if useGradient:
                    if isinstance(axialDirection, np.ndarray):
                        grad = np.radians(axialDirection[yi, xi])
                    else:
                        grad = axialDirection

                    cG = np.cos(grad)
                    sG = np.sin(grad)

                    dist = np.power((xDist[pir] * cG - yDist[pir] * sG), 2) / _sepFloorA2 +\
                           np.power(
                               (xDist[pir] * sG + yDist[pir] * cG), 2) / _sepFloorT2

                    immidiatelyInRange = dist <= 1

                else:
                    immidiatelyInRange = np.power(
                        xDist[pir], 2) + np.power(yDist[pir], 2) <= _sepFloor2

                if immidiatelyInRange.any():
                    continue

                # Determine if a placement has been found
                if pixelDivision == 1:
                    found = True
                    xsp = xi
                    ysp = yi
                else:
                    # Start searching in the 'sub pixel'
                    found = False
                    for xsp in substeps + xi:
                        xSubDist = xClip[pir] - xsp
                        for ysp in substeps + yi:
                            ySubDist = yClip[pir] - ysp

                            # Test if any points in the range are overlapping
                            if useGradient:  # Test if in rotated ellipse
                                dist = (np.power((xSubDist * cG - ySubDist * sG), 2) / _sepA2) +\
                                       (np.power((xSubDist * sG + ySubDist * cG), 2) / _sepT2)
                                overlapping = dist <= 1

                            else:  # test if in circle
                                overlapping = (np.power(xSubDist, 2) +
                                               np.power(ySubDist, 2)) <= _sep2

                            if not overlapping.any():
                                found = True
                                break

                        if found:
                            break

                # Add if found
                if found:
                    x[cnt] = xsp
                    y[cnt] = ysp
                    cnt += 1

                    if _stamping:
                        xspi = int(np.round(xsp))
                        yspi = int(np.round(ysp))

                        stamp_center = stampWidth
                        if xspi - stampWidth < 0:
                            _x_low = 0
                            _x_low_stamp = stamp_center - xspi
                        else:
                            _x_low = xspi - stampWidth
                            _x_low_stamp = 0

                        if yspi - stampWidth < 0:
                            _y_low = 0
                            _y_low_stamp = stamp_center - yspi
                        else:
                            _y_low = yspi - stampWidth
                            _y_low_stamp = 0

                        if xspi + stampWidth > (xN - 1):
                            _x_high = xN - 1
                            _x_high_stamp = stamp_center + (xN - xspi - 1)
                        else:
                            _x_high = xspi + stampWidth
                            _x_high_stamp = stamp_center + stampWidth

                        if yspi + stampWidth > (yN - 1):
                            _y_high = yN - 1
                            _y_high_stamp = stamp_center + (yN - yspi - 1)
                        else:
                            _y_high = yspi + stampWidth
                            _y_high_stamp = stamp_center + stampWidth

                        _stamp = stamp[_y_low_stamp:_y_high_stamp + 1,
                                 _x_low_stamp:_x_high_stamp + 1]

                        workingAvailability[_y_low:_y_high + 1,
                        _x_low:_x_high + 1] *= _stamp

        # Convert identified points back into the region's coordinates
        coords = np.zeros((cnt, 2))
        # shifted by 0.5 so that index corresponds to the center of the pixel
        coords[:, 0] = s.region.extent.xMin + (x[:cnt] + 0.5) * s.region.pixelWidth
        # shifted by 0.5 so that index corresponds to the center of the pixel
        coords[:, 1] = s.region.extent.yMax - \
                       (y[:cnt] + 0.5) * s.region.pixelHeight

        s._itemCoords = coords

        if not outputSRS is None:
            newCoords = gk.srs.xyTransform(
                coords, fromSRS=s.region.srs, toSRS=outputSRS)
            newCoords = np.column_stack(
                [[v[0] for v in newCoords], [v[1] for v in newCoords]])
            coords = newCoords
        s.itemCoords = coords

# Filter by max acceptable distance, maybe
        if maxAcceptableDistance is not None:
            try:
                maxAcceptableDistance = [float(x) for x in maxAcceptableDistance]
            except:
                maxAcceptableDistance = [float(maxAcceptableDistance)]

            maxAcceptableDistance2 = np.power(maxAcceptableDistance, 2)

            sel = []
            for i in range(s._itemCoords.shape[0]):
                x = s._itemCoords[i, 0]
                y = s._itemCoords[i, 1]

                X = np.concatenate((s._itemCoords[:i, 0], s._itemCoords[(i + 1):, 0]))
                Y = np.concatenate((s._itemCoords[:i, 1], s._itemCoords[(i + 1):, 1]))
                subsel = np.abs(X - x) <= max(maxAcceptableDistance)
                subsel *= np.abs(Y - y) <= max(maxAcceptableDistance)

                subX = X[subsel]
                subY = Y[subsel]
                dist2 = np.power(subX - x, 2) + np.power(subY - y, 2)

                if dist2.shape[0] < len(maxAcceptableDistance2):
                    sel.append(False)
                else:
                    isokay = True
                    dist2 = np.sort(dist2)
                    for j, md2 in enumerate(maxAcceptableDistance2):
                        isokay = isokay and dist2[j] <= md2
                    sel.append(isokay)

            s._itemCoords = s._itemCoords[sel, :]
            s.itemCoords = s.itemCoords[sel, :]

        # Make areas
        if asArea:
            warn("Area distribution will soon be removed from 'distributeItems'. Use 'distributeArea' instead",
                 DeprecationWarning)

            ext = s.region.extent.pad(_voronoiBoundaryPadding, percent=True)

            # Do Voronoi
            from scipy.spatial import Voronoi

            # Add boundary points around the 'good' points so that we get bounded regions for each 'good' point
            pts = np.concatenate([s._itemCoords,
                                  [(x, ext.yMin) for x in np.linspace(
                                      ext.xMin, ext.xMax, _voronoiBoundaryPoints)],
                                  [(x, ext.yMax) for x in np.linspace(
                                      ext.xMin, ext.xMax, _voronoiBoundaryPoints)],
                                  [(ext.xMin, y) for y in np.linspace(
                                      ext.yMin, ext.yMax, _voronoiBoundaryPoints)][1:-1],
                                  [(ext.xMax, y) for y in np.linspace(ext.yMin, ext.yMax, _voronoiBoundaryPoints)][
                                  1:-1], ])

            v = Voronoi(pts)

            # Create regions
            geoms = []
            for reg in v.regions:
                path = []
                if -1 in reg or len(reg) == 0:
                    continue
                for pid in reg:
                    path.append(v.vertices[pid])
                path.append(v.vertices[reg[0]])

                geoms.append(gk.geom.polygon(path, srs=s.region.srs))

            if not len(geoms) == len(s._itemCoords):
                raise GlaesError("Mismatching geometry count")

            # Create a list of geometry from each region WITH availability
            vec = gk.vector.createVector(
                geoms, fieldVals={"pid": range(1, len(geoms) + 1)})
            areaMap = s.region.rasterize(
                vec, value="pid", dtype=int) * (s._availability > threshold)

            geoms = gk.geom.polygonizeMatrix(
                areaMap, bounds=s.region.extent, srs=s.region.srs, flat=True)
            geoms = list(filter(lambda x: x.Area() >= minArea, geoms.geom))

            # Save in the s._areas container
            s._areas = geoms

        # Make shapefile
        if not output is None:
            warn("Shapefile output will soon be removed from 'distributeItems'. Use 'saveItems' or 'saveAreas' instead",
                 DeprecationWarning)
            srs = gk.srs.loadSRS(
                outputSRS) if not outputSRS is None else s.region.srs
            # Should the locations be converted to areas?
            if asArea:
                if not srs.IsSame(s.region.srs):
                    geoms = gk.geom.transform(
                        geoms, fromSRS=s.region.srs, toSRS=srs)

                # Add 'area' column
                areas = [g.Area() for g in geoms]
                geoms = pd.DataFrame({"geom": geoms, "area": areas})

            else:  # Just write the points
                geoms = gk.LocationSet(
                    s._itemCoords,
                    srs=s.srs
                ).asGeom(srs=srs if outputSRS is None else outputSRS)

            gk.vector.createVector(geoms, output=output)
        else:
            if asArea:
                return geoms
            else:
                return coords

    def distributeAreas(s, points=None, minArea=100000, threshold=50, _voronoiBoundaryPoints=10,
                        _voronoiBoundaryPadding=5):
        if points is None:
            try:
                points = s._itemCoords
            except:
                raise GlaesError(
                    "Point data could not be found. Have you ran 'distributeItems'?")
        else:
            points = np.array(points)
            s = points[:, 0] >= s.region.extent.xMin
            s = s & (points[:, 0] <= s.region.extent.xMax)
            s = s & (points[:, 1] >= s.region.extent.yMin)
            s = s & (points[:, 1] <= s.region.extent.yMax)

            if not s.any():
                raise GlaesError("None of the given points are in the extent")

        ext = s.region.extent.pad(_voronoiBoundaryPadding, percent=True)

        # Do Voronoi
        from scipy.spatial import Voronoi

        # Add boundary points around the 'good' points so that we get bounded regions for each 'good' point
        pts = np.concatenate([points,
                              [(x, ext.yMin) for x in np.linspace(
                                  ext.xMin, ext.xMax, _voronoiBoundaryPoints)],
                              [(x, ext.yMax) for x in np.linspace(
                                  ext.xMin, ext.xMax, _voronoiBoundaryPoints)],
                              [(ext.xMin, y) for y in np.linspace(
                                  ext.yMin, ext.yMax, _voronoiBoundaryPoints)][1:-1],
                              [(ext.xMax, y) for y in np.linspace(ext.yMin, ext.yMax, _voronoiBoundaryPoints)][1:-1], ])

        v = Voronoi(pts)

        # Create regions
        geoms = []
        for reg in v.regions:
            path = []
            if -1 in reg or len(reg) == 0:
                continue
            for pid in reg:
                path.append(v.vertices[pid])
            path.append(v.vertices[reg[0]])

            geoms.append(gk.geom.polygon(path, srs=s.region.srs))

        if not len(geoms) == len(s._itemCoords):
            raise RuntimeError("Mismatching geometry count")

        # Create a list of geometry from each region WITH availability
        vec = gk.vector.createVector(
            geoms, fieldVals={"pid": range(1, len(geoms) + 1)})
        areaMap = s.region.rasterize(
            vec, value="pid", dtype=int) * (s._availability > threshold)

        geoms = gk.geom.polygonizeMatrix(
            areaMap, bounds=s.region.extent, srs=s.region.srs, flat=True)
        geoms = list(filter(lambda x: x.Area() >= minArea, geoms.geom))

        # Save in the s._areas container
        s._areas = geoms
        return geoms

    def saveItems(s, output, srs=None, data=None):
        # Get srs
        srs = gk.srs.loadSRS(srs) if not srs is None else s.region.srs

        # transform?
        if not srs.IsSame(s.region.srs):
            points = gk.srs.xyTransform(
                s._itemCoords, fromSRS=s.region.srs, toSRS=srs, outputFormat="raw")
        else:
            points = s._itemCoords
        points = [gk.geom.point(pt[0], pt[1], srs=srs) for pt in points]

        # make shapefile
        if data is None:
            data = pd.DataFrame(dict(geom=points))
        else:
            data = pd.DataFrame(data)
            data['geom'] = points

        return gk.vector.createVector(data, output=output)

    def saveAreas(s, output, srs=None, data=None):
        # Get srs
        srs = gk.srs.loadSRS(srs) if not srs is None else s.region.srs

        # transform?
        if not srs.IsSame(s.region.srs):
            geoms = gk.geom.transform(
                s._areas, fromSRS=s.region.srs, toSRS=srs)
        else:
            geoms = s._areas

        # make shapefile
        areas = [g.Area() for g in geoms]
        if data is None:
            data = pd.DataFrame({"geom": geoms, "area": areas})
            # data = pd.DataFrame(dict(geom=geoms))
        else:
            data = pd.DataFrame(data)
            data['geom'] = geoms
            data['area'] = areas

        return gk.vector.createVector(data, output=output)
