from helpers import *

def test_ExclusionCalculator():
    # Test by giving a shapefile
    ec = gl.ExclusionCalculator(aachenShape)
    
    if ec.region.mask.shape != (523, 340) or \
       ec.region.mask.sum() !=  70944 or \
       not isclose(ec.region.mask.std(), 0.48968559141, 1e-6):
        raise RuntimeError("Fail")

    # Test by giving a region mask
    rm = gk.RegionMask.load(aachenShape)
    ec = gl.ExclusionCalculator(rm)

    if ec.region.mask.shape != (523, 340) or \
       ec.region.mask.sum() !=  70944 or \
       not isclose(ec.region.mask.std(), 0.48968559141, 1e-6):
        raise RuntimeError("Fail")

    # Test by giving a region mask with different resolution
    rm = gk.RegionMask.load(aachenShape, srs=gk.srs.EPSG4326, pixelSize=0.001)
    ec = gl.ExclusionCalculator(rm)

    if ec.region.mask.shape != (457, 446) or \
       ec.region.mask.sum() !=  90296 or \
       not isclose(ec.region.mask.std(), 0.496741981394, 1e-6):
        raise RuntimeError("Fail")

def test_excludeRasterType():
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, 12)

    if not (isclose(ec.availability.mean(), 32.497469, 1e-6) and isclose(ec.availability.std(), 46.836540, 1e-6)):
        raise RuntimeError("excludeRasterType: Single value exclusion - FAIL")

    # exclude value range
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, (5,12))

    if not (isclose(ec.availability.mean(), 31.626365, 1e-6) and isclose(ec.availability.std(), 46.501717, 1e-6)):
        raise RuntimeError("excludeRasterType: Value range exclusion - FAIL")
    
    # exclude value maximum
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, (None,12))

    if not (isclose(ec.availability.mean(), 23.848273, 1e-6) and isclose(ec.availability.std(), 42.615572, 1e-6)):
        raise RuntimeError("excludeRasterType: Value maximum exclusion - FAIL")
    
    ecMax12 = gl.ExclusionCalculator(aachenShape)
    ecMax12.excludeRasterType(clcRaster, valueMax=12)

    if not np.abs(ecMax12.availability-ec.availability).sum()/ec.availability.size < 0.000001:
        raise RuntimeError("excludeRasterType: valueMax argument - FAIL")

    # exclude value minimum
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, (13,None))

    if not (isclose(ec.availability.mean(), 16.048251, 1e-6) and isclose(ec.availability.std(), 36.705297, 1e-6)):
        raise RuntimeError("excludeRasterType: Value minimum exclusion - FAIL")
    
    ecMin13 = gl.ExclusionCalculator(aachenShape)
    ecMin13.excludeRasterType(clcRaster, valueMin=13)

    if not np.abs(ecMin13.availability-ec.availability).sum()/ec.availability.size < 0.000001:
        raise RuntimeError("excludeRasterType: valueMin argument - FAIL")

    # Make sure min and max align
    if ((ecMax12.availability>0.5) == (ecMin13.availability>0.5))[ecMax12.region.mask].any():
        raise RuntimeError("excludeRasterType: minimum and maximum overlap - FAIL")

    if not (np.logical_or( ecMax12.availability>0.5, ecMin13.availability>0.5 ) == ecMin13.region.mask).all():
        raise RuntimeError("excludeRasterType: minimum and maximum summation - FAIL")

    ### Test with a different projection system
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelSize=0.005)
    ec.excludeRasterType(clcRaster, 12)

    
def test_excludeVectorType():
    # exclude all features directly
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector)

    if not (isclose(ec.availability.mean(), 30.511191, 1e-6) and isclose(ec.availability.std(), 46.045485, 1e-6)):
        raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

    # exclude all features directly, new srs
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelSize=0.005)
    ec.excludeVectorType(cddaVector)

    if not (isclose(ec.availability.mean(), 32.915172, 1e-6) and isclose(ec.availability.std(), 46.990514, 1e-6)):
        raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

    # exclude a selection of features
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000")

    if not (isclose(ec.availability.mean(), 34.6693277, 1e-6) and isclose(ec.availability.std(), 47.5917101, 1e-6)):
        raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

    # exclude a selection of features with buffer
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000", buffer=400)

    if not (isclose(ec.availability.mean(), 31.0994267, 1e-6) and isclose(ec.availability.std(), 46.2900430, 1e-6)):
        raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

def test_excludePrior():
    # make a prior source 
    pr = gl.core.priors.PriorSource(priorSample)

    # test same srs
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, valueMin=400)
    if not (isclose(ec.availability.mean(), 9.92528961872, 1e-6) and isclose(ec.availability.std(), 29.0665453089, 1e-6)):
        raise RuntimeError("excludePrior: Same srs - FAIL")

    # test different srs and resolution
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelSize=0.001)
    ec.excludePrior(pr, valueMin=400)

    if not (isclose(ec.availability.mean(), 11.0289909823, 1e-6) and isclose(ec.availability.std(), 30.4952463815, 1e-6)):
        raise RuntimeError("excludePrior: Different srs - FAIL")

def test_multiple_exclusions():
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)

    # apply exclusions
    ec.excludePrior(pr, valueMax=400)   
    ec.excludeVectorType(cddaVector, where="YEAR>2000")
    ec.excludeRasterType(clcRaster, valueMax=12)

    if not (isclose(ec.availability.mean(), 15.2027443482, 1e-6) and isclose(ec.availability.std(), 35.6248783489, 1e-6)):
        raise RuntimeError("Multiple Exclusions - FAIL")

if __name__ == "__main__":
    test_ExclusionCalculator()
    test_excludeRasterType()
    test_excludeVectorType()
    test_excludePrior()
    test_multiple_exclusions()
