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
priordir = join(dirname(__file__), "..", "..", "data", "priors")

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
        s.displayName = ds.GetMetadataItem("DISPLAY_NAME")
        if s.displayName is None: 
            s.displayName = splitext(basename(path))[0]
        
        s.unit = ds.GetMetadataItem("UNIT")
        s.description = ds.GetMetadataItem("DESCRIPTION")
        s.alternateName = ds.GetMetadataItem("ALTERNATE_NAME")

        # create edges and estimation-values
        try:
            valMap = json.loads(ds.GetMetadataItem("VALUE_MAP"))
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
        s.noDataValue = -999999

        # Make the doc string
        doc = ""
        doc += "%s\n"%s.description
        doc += "UNITS: %s\n"%s.unit
        doc += "VALUE MAP:\n"
        doc += "  Raw Value : Precalculated Edge : Estimated Value\n"
        for i in range(len(s.edges)): 
            doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(i, s.edgeStr[i], s.values[i])
        doc += "  {:^9} - {:^18s} - {:^15.3f}\n".format(254, "untouched", s.untouchedTight)
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
            return False
        else:
            if verbose:
                print("PRIORS-%s: %f is significantly different from the closest precalculated edge (%f)"%(s.displayName,val,bestEdge))
            return False

    #### Make a datasource generator
    def generateRaster(s, extent, untouched='Tight', **kwargs):
        
        # make better values
        values = s.values
        if untouched.lower()=='tight':
            untouchedValue = s.untouchedTight
        elif untouched.lower()=='wide':
            untouchedValue = s.untouchedWide
        else:
            raise RuntimeError("'untouched' must be 'Tight' or 'Wide")

        # make a mutator function to make indexes to estimated values
        #indexToValue = np.vectorize(lambda i: s.values[i]) # TODO: test 'interp' vs 'vectorize'
        #def mutator(data):
        #    return indexToValue(data)
        def mutator(data):
            noData = data == 255 
            untouched = data == 254
            result = np.interp(data, range(len(values)), values)
            result[untouched] = untouchedValue
            result[noData] = s.noDataValue
            return result

        # mutate main source
        clipDS = extent.clipRaster(s.path)
        mutDS = gk.raster.mutateValues(clipDS, processor=mutator, noData=s.noDataValue, **kwargs)

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

    def extractValues(s, points, **kwargs):
        values = s.values.tolist()
        values.append(s.untouchedTight)

        indicies = gk.raster.extractValues(s.path, points=points, **kwargs)
        
        if isinstance(indicies, list): 
            return np.array([values[i.data] for i in indicies ])
        else: 
            return values[indicies.data]
        
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
                s.sources[p.displayName] = p
                if p.alternateName != "NONE":
                    # make a new prior and update the displayName
                    p2 = PriorSource(f)
                    p2.displayName = p.alternateName
                    s.sources[p.alternateName] = p2

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
            print("PRIORS-WARNING: A portion of the defined region is not included of the precalculated exclusion areas")
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
            priorLow = prior.lower()
            scores = [SM(None, priorLow, priorName).ratio() for priorName in priorNames]

            bestMatch = priorNames[np.argmax(scores)]

            print("PRIORS: Mapping '%s' to '%s'"%(prior, bestMatch))

            output = s.sources[bestMatch]

        return output

    def combinePriors(s, reg, priorNames, combiner='min'):

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
Priors = PriorSet(priordir)
