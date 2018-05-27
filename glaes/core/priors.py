from .util import *

# Sort out the data paths
defaultPriorDir = join(dirname(dirname(__file__)), "data", "priors")

# Prior datasource class
class PriorSource(object):
    """The PriorSource object loads one of the Prior datasets and makes it 
    accessible for use in general purpose geospatial analyses"""
    class _LoadFail(Exception):pass

    def __init__(s, path):
        """Initialize a PriorSource object by passing it a path on disk"""
        s.path = path
        ds = gk.raster.loadRaster(path)
        ri = gk.raster.rasterInfo(ds)

        # Check if we're dealing with a GLAES prior
        if ri.meta.get("GLAES_PRIOR", "NO") != "YES": raise PriorSource._LoadFail()

        # Load basic values
        s.displayName = ri.meta.get("DISPLAY_NAME", splitext(basename(path))[0])
        
        s.unit = ri.meta.get("UNIT", "Unknown")
        s.description = ri.meta.get("DESCRIPTION", "Unknown")
        s.alternateName = ri.meta.get("ALTERNATE_NAME", None)

        s.xMin = ri.xMin
        s.xMax = ri.xMax
        s.yMin = ri.yMin
        s.yMax = ri.yMax
        s.bounds = ri.bounds
        s.srs = ri.srs
        s.dx = ri.dx
        s.dy = ri.dy

        # create edges and estimation-values
        try:
            valMap = json.loads(ri.meta["VALUE_MAP"])
        except Exception as e:
            print(path)
            raise e

        s.edgeStr = []
        s.edges = []
        s.values = []

        numRE = re.compile("^(?P<qualifier>[<>]?=?)(?P<value>-?[0-9.]+)$")

        # Arange values and qualifiers
        rawValues = []
        qualifiers = []
        for i in range(253):
            try:
                valString = valMap["%d"%i]
            except KeyError: # should fail when we've reached the end of the precalculated edges
                break

            s.edgeStr.append(valString)

            try:
                qualifier, value = numRE.search(valString).groups()
            except Exception as e:
                print(valString)
                raise e

            rawValues.append(float(value))
            qualifiers.append(qualifier)

        # set values
        for i in range(len(rawValues)):
            # estimate a value
            if qualifiers[i]=="<": s.values.append( rawValues[i]-0.001 ) # subtract a little bit
            elif qualifiers[i]==">": s.values.append( rawValues[i]+0.001 ) # add a little bit
            else: 
                if qualifiers[i]=="<=" and i!=0: 
                    val = (rawValues[i]+rawValues[i-1])/2
                elif qualifiers[i]==">=" and i!=(len(rawValues)-1): 
                    val = (rawValues[i]+rawValues[i+1])/2 
                else:
                    val = rawValues[i]

                s.values.append(val)
        
        # make into numpy arrays
        s.edges = np.array(rawValues)
        s.values = np.array(s.values)

        if not s.edges.size == s.values.size: raise RuntimeError(basename(path)+": edges length does not match values length")

        # make nodata and untouched value
        qualifier, value = numRE.search( valMap["%d"%(s.values.size-1)] ).groups() # Get the last calculated edge
        value = float(value)

        # estimate a value
        if qualifier=="<=": # set the untouched value to everything above value
            s.untouchedTight = value+0.001 
            s.untouchedWide = value+100000000000000 
        elif qualifier==">=": # do the opposite in this case
            s.untouchedTight = value-0.001 
            s.untouchedWide = value-100000000000000
        else: 
            s.untouchedValue = value
        s.noData = -999999

        tmp = s.values.tolist()
        tmp.append(s.untouchedWide)
        s._values_wide = np.array(tmp)

        tmp = s.values.tolist()
        tmp.append(s.untouchedTight)
        s._values_tight = np.array(tmp)

        # Make the doc string
        doc = ""
        doc += "%s\n"%s.description
        doc += "UNITS: %s\n"%s.unit
        doc += "VALUE MAP:\n"
        doc += "  Raw Value : Precalculated Edge : Estimated Value\n"
        for i in range(len(s.edges)): 
            doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(i, s.edgeStr[i], s.values[i])
        doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(254, "untouched", s.untouchedTight)
        doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(255, "no-data", s.noData)

        s.__doc__ = doc

    def containsValue(s, val, verbose=False):
        """Checks if a given value is withing the known values in the Prior source

        * If 'verbose' is true, a warning is issued when the given value is outside
        of the Prior's known edge values
        """
        if val <= s.edges.max() and val >= s.edges.min(): 
            return True
        else: 
            if verbose:
                warn("%s: %f is outside the predefined boundaries (%f - %f)"%(s.displayName,val,s.edges.min(),s.edges.max()), UserWarning)
            return False

    def valueOnEdge(s, val, verbose=False):
        """Checks is a given value is exactly on one of the precomputed edge values

        * If 'verbose' is true, a warning is issued when the given value is more
        than 5% deviant from the closest precomputed edge
        """
        bestI = np.argmin(np.abs(s.edges-val))
        bestEdge = s.edges[bestI]

        if (abs(bestEdge-val) < 0.0001):
            return True
        elif abs((bestEdge-val)/(bestEdge if bestEdge!=0 else val)) <= 0.05:
            return False
        else:
            if verbose:
                warn("PRIORS-%s: %f is significantly different from the closest precalculated edge (%f)"%(s.displayName,val,bestEdge), UserWarning)
            return False

    #### Make a datasource generator
    def generateRaster(s, extent, untouched='tight', **kwargs):
        """Generates a raster datasource around the indicated extent

        Parameters:
        -----------
        extent: geokit.Extent or tuple
            Describes the geographic boundaries around which to create the new 
            raster dataset
            * Using an Extent object is the most robust method
            * If a tuple is given, (lonMin, latMin, lonMax, latMax) is expected
                - In this case, an Extent object is created immediately and cast
                  to the Prior's srs (EPSG3035)
            * In truth, anything acceptable to geokit.Extent.load() could be 
              given as an input here

        untouched: str; optional
            Determines how to treat values outside of the Prior's edge list
            * If 'tight', pixels which are untouched are given a value slightly
              beyond than the final edge
            * If 'wide', they are given a value far away from the final edge
        
        **kwargs: 
            All keyword arguments are passed along to geokit.raster.mutateRaster

        Returns:
        --------
        gdal.Dataset

        """
        # Be sure we have an extent object which fits to the srs and resolution of the Priors
        if not isinstance(extent, gk.Extent):
            extent = gk.Extent.load(extent).castTo(gk.srs.EPSG3035).fit(100)
        
        # make better values
        values = s.values
        if untouched.lower()=='tight':
            untouchedValue = s.untouchedTight
        elif untouched.lower()=='wide':
            untouchedValue = s.untouchedWide
        else:
            raise RuntimeError("'untouched' must be 'Tight' or 'Wide")

        # make a mutator function to make indexes to estimated values
        def mutator(data):
            noData = data == 255 
            untouched = data == 254
            result = np.interp(data, range(len(values)), values)
            result[untouched] = untouchedValue
            result[noData] = s.noData
            return result

        # mutate main source
        mutDS = gk.raster.mutateRaster(s.path, bounds=extent.xyXY, boundsSRS=extent.srs, 
                                       processor=mutator, noData=s.noData, **kwargs)

        # return
        return mutDS

    #### Make a datasource generator
    def generateVector(s, extent, value, output=None):
        """Generates a vector datasource around the indicated extent and at an
        approximation at the indicated value
        
        * If a value is given that corresponds to one of the pre-calculated edges,
          the Prior source is 'polygonized' exactly at that edge
        * If a value is given which falls between two edges, the closest edge is
          polygonized and the resulting geometry is shrunk or grown to make up
          the difference
          - Be careful, this is a costly procedure! 

        Note:
        -----
        This procedure really only makes sense for the Priors which represent 
        the distance from something, such as 'roads proximity'. It isn't very
        meaningful to use this for a quantity-based prior, such as "terrain slope"

        Parameters:
        -----------
        extent: geokit.Extent or tuple
            Describes the geographic boundaries around which to create the new 
            raster dataset
            * Using an Extent object is the most robust method
            * If a tuple is given, (lonMin, latMin, lonMax, latMax) is expected
                - In this case, an Extent object is created immediately and cast
                  to the Prior's srs (EPSG3035)
            * In truth, anything acceptable to geokit.Extent.load() could be 
              given as an input here

        value: numeric
            The edge to attempt to reconstruct
            
        output: str; optional
            A place to put the output if its not needed in memory

        Returns:
        --------
        gdal.Dataset

        """
        # Be sure we have an extent object which fits to the srs and resolution of the Priors
        if not isinstance(extent, gk.Extent):
            extent = gk.Extent.load(extent)
        extent= extent.castTo(gk.srs.EPSG3035).fit(100)

        # get closest edge
        edgeDiffs = np.abs(value-s.edges)
        edgeI = np.argmin( edgeDiffs )

        # Extract the matrix around the extent and test against edge index
        mat = extent.extractMatrix( s.path, strict=True ) <= edgeI
        
        # Polygonize
        geoms = gk.geom.polygonizeMask(mat, bounds=extent.xyXY, srs=extent.srs, flat=False, shrink=False)

        # Do extra grow
        if edgeDiffs[edgeI]/s.edges[edgeI] > 0.01: 
            extraDist = value-s.edges[edgeI]
            geoms = [g.Buffer(extraDist) for g in geoms]

        # merge to one geometry
        geom = gk.geom.flatten(geoms)

        # create vector
        vecDS = gk.vector.createVector(geom, srs=extent.srs, output=output)

        # return
        return vecDS

    def extractValues(s, points, **kwargs):
        values = s.values.tolist()
        values.append(s.untouchedTight)

        indicies = gk.raster.extractValues(s.path, points=points, **kwargs)
        
        if isinstance(indicies, list): 
            return np.array([values[i.data] for i in indicies ])
        else: 
            return values[indicies.data]
        
# Load priors
class PriorSet(object):
    """The PriorSet object loads and manages Prior datasets

    * Individual Prior datasets can be extracted either by using the Priors[<name>]
      or Priors.<name> conventions
    * If one needs to change the the Prior directory, this can be done by calling
      the Priors.loadDirectory( <directory> ) function
    """
    def __init__(s,path):
        """Initialize a PriorSet object by passing a path, normally a user shouldn't
        need to interact with this initializer"""
        s._sources = OrderedDict()
        s.loadDirectory(path)

    def loadDirectory(s, path):
        """Looks into a directory and attempts to load all raster (.tif) files as
        if they were a Prior dataset

        * Each call to this function adds to any other previously identified Priors
        """
        for f in glob(join(path,"*.tif")):
            if basename(f) == 'goodAreas.tif':continue
            
            try:

                p = PriorSource(f)
                
                if hasattr(s,p.displayName): warn("Overwriting '%s'"%p.displayName, UserWarning)

                s.sources[p.displayName] = p
                setattr(s, p.displayName, p)
                
                if p.alternateName != "NONE":
                    # make a new prior and update the displayName
                    p2 = PriorSource(f)
                    p2.displayName = p.alternateName
                    s.sources[p.alternateName] = p2
                    setattr(s, p.alternateName, p2)

            except PriorSource._LoadFail:
                warn("Could not parse file: %s"%(basename(f)), UserWarning)

    def regionIsOkay(s, region):
        """Checks if a given region is valid within the Prior Datasets

        * Not really intended for external use and will probably fail
        """
        # Check if region is okay
        goodPixels = region.indicateValues(join(s.path,"goodArea.tif"), value=1).sum()

        goodRatio = goodPixels/region.mask.sum()
        if goodRatio > 0.9999:
            # Evertyhing is okay
            return True
        elif goodRatio > 0.95:
            print("PRIORS-WARNING: A portion of the defined region is not included of the precalculated exclusion areas")
            return True
        else:
            return False

    @property
    def sources(s): 
        """An easily indexable/searchable list of the PriorSet's sources"""
        return s._sources

    @property
    def listKeys(s):
        """Returns the keys of ll loaded Priors"""
        k = list(s._sources.keys())
        k.sort()
        for _k in k: print(_k)

    def __getitem__(s,prior):
        if len(s.sources)==0:
            raise GlaesError("No priors have been installed. Use gl.setPriorDirectory( <path> ) or else place the files directly in the default prior data directory (%s)"%defaultPriorDir)
        try:
            output = s.sources[prior]
        except KeyError:
            priorNames = list(s.sources.keys())
            priorLow = prior.lower()
            scores = [SM(None, priorLow, priorName).ratio() for priorName in priorNames]

            bestMatch = priorNames[np.argmax(scores)]

            warn("Mapping '%s' to '%s'"%(prior, bestMatch), UserWarning)

            output = s.sources[bestMatch]

        return output

    def combinePriors(s, reg, priorNames, combiner='min'):
        """Combines two or more priors into a single prior

        * Still experimental, and will probably fail
        """

        # make output matrix
        outputMatrix = np.ones(reg.mask.shape)*999999999

        # append each matrix
        for name in priorNames:
            tmp = reg.warp(s[name].generateRaster(reg.extent, "Wide"), applyMask=False)
            if combiner == 'min':
                outputMatrix = np.min([outputMatrix,tmp], 0)
            elif combiner == 'max':
                outputMatrix = np.max([outputMatrix,tmp], 0)

        # make an output
        outputRaster = reg.createRaster(data=outputMatrix)
        return outputRaster


# MAKE THE PRIORS!
Priors = PriorSet(defaultPriorDir)
