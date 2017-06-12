import geokit as gk
import re 
import numpy as np
from glob import glob 
from os.path import dirname, basename, join, isdir, splitext
from collections import namedtuple, OrderedDict
import json
from warnings import warn
from difflib import SequenceMatcher as SM


# Sort out the data paths
priordir = join(dirname(__file__), "..", "..", "data", "europe")

# Typical criteria
Criterion = namedtuple("Criteria","doc typicalExclusion unit excludeDirection evaluationName untouchedValue noDataValue")

# Prior datasource class
class PriorSource(object):
    class _LoadFail(Exception):pass

    def __init__(s, path):
        s.path = path
        ds = gk.raster.loadRaster(path)

        # Check if we're dealign with a GLAES prior
        priorCheck = ds.GetMetadataItem("GLAES_PRIOR") 
        if priorCheck is None or priorCheck != "YES": raise s._LoadFail()

        # Load basic values
        s.displayName = splitext(basename(path))[0]
        s.unit = ds.GetMetadataItem("UNIT")
        s.description = ds.GetMetadataItem("DESCRIPTION")

        # create edges and estimation-values
        valMap = json.loads(ds.GetMetadataItem("VALUE_MAP"))

        s.edgeStr = []
        s.edges = []
        s.values = []
        numRE = re.compile("^(?P<qualifier>[<>]?=?)(?P<value>-?[0-9.]+)$")
        for i in range(len(valMap.keys())-1):
            valString = valMap["%d"%i]
            s.edgeStr.append(valString)

            try:
                qualifier, value = numRE.search(valString).groups()
            except Exception as e:
                print(valString)
                raise e
            value = float(value)
            s.edges.append( value )

            # estimate a value
            if qualifier=="<": s.values.append( value-0.001 )
            elif qualifier==">": s.values.append( value+0.001 )
            else: s.values.append( value )
        
        s.edges = np.array(s.edges)
        s.values = np.array(s.values)

        # make nodata and untouched value
        qualifier, value = numRE.search(valMap["254"]).groups()
        value = float(value)

        # estimate a value
        if qualifier=="<": s.untouchedValue = value-0.001
        elif qualifier==">": s.untouchedValue = value+0.001
        else: s.untouchedValue = value
        s.noDataValue = -999999

        # Make the doc string
        doc = ""
        doc += "%s\n"%s.description
        doc += "UNITS: %s\n"%s.unit
        doc += "VALUE MAP:\n"
        doc += "  Raw Value : Precalculated Edge : Estimated Value\n"
        for i in range(len(s.edges)): 
            doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(i, s.edgeStr[i], s.values[i])
        doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(254, "untouched", s.untouchedValue)
        doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(255, "no-data", s.noDataValue)

        s.__doc__ = doc

    def containsValue(s, val, verbose=False):
        if val <= s.edges.max() and val >= s.edges.min(): 
            return True
        else: 
            if verbose:
                warn("%s: %f is outside the predefined boundaries (%f - %f)"%(s.displayName,val,s.edges.min(),s.edges.max()), Warning)
            return False

    def valueOnEdge(s, val, verbose=False):
        bestI = np.argmin(np.abs(s.edges-val))
        bestEdge = s.edges[bestI]

        if (abs(bestEdge-val) < 0.0001):
            return True
        elif abs((bestEdge-val)/(bestEdge if bestEdge!=0 else val)) <= 0.05:
            return True
        else:
            if verbose:
                warn("%s: %f is significantly different from the closest precalculated edge (%f)"%(s.displayName,val,bestEdge), Warning)
            return False

    #### Make a datasource generator
    def generateRaster(s, extent ):
        
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
            result[untouched] = s.untouchedValue
            result[noData] = s.noDataValue
            return result

        # mutate main source
        clipDS = extent.clipRaster(s.path)
        mutDS = gk.raster.mutateValues(clipDS, processor=mutator, noData=s.noDataValue)

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
            if basename(f) == 'goodAreas.tif':continue
            
            try:
                p = PriorSource(f)
                setattr(s, p.displayName, p)
                s.sources[p.displayName] = p
            except PriorSource._LoadFail:
                print("WARNING: Could not parse file: %s"%(basename(f)))
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

    def __getitem__(s,prior):
        try:
            output = s.sources[prior]
        except KeyError:
            priorNames = list(s.sources.keys())
            scores = [SM(None, prior, priorName).ratio() for priorName in priorNames]

            bestMatch = priorNames[np.argmax(scores)]
            print("Mapping '%s' to '%s'"%(prior, bestMatch))

            output = s.sources[bestMatch]

        return output

# MAKE THE PRIORS!
Priors = PriorSet(priordir)
