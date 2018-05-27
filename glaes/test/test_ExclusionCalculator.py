from helpers import *


def test_multiple_exclusions():
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)

    # apply exclusions
    ec.excludePrior(pr, valueMax=400)   
    ec.excludeVectorType(cddaVector, where="YEAR>2000")
    ec.excludeRasterType(clcRaster, valueMax=12)
    
    if not (np.isclose(ec.availability.mean(), 14.79785738387133, 1e-6) and np.isclose(np.nanstd(ec.availability), 35.249694265461024, 1e-6)):
        raise RuntimeError("Multiple Exclusions - FAIL")

def test_ExclusionCalculator___init__():
    # Test by giving a shapefile
    ec = gl.ExclusionCalculator(aachenShape)
    
    compare(ec.region.mask.shape ,(509, 304) )
    compareF(ec.region.mask.sum() , 70944 )
    compareF(ec.region.mask.std(), 0.498273451386)

    # Test by giving a region mask
    rm = gk.RegionMask.load(aachenShape, padExtent=5000)
    ec = gl.ExclusionCalculator(rm)

    compare(ec.region.mask.shape ,(609, 404) )
    compareF(ec.region.mask.sum() , 70944 )
    compareF(ec.region.mask.std(), 0.45299387483)

    # Test by giving a region mask with different resolution and srs
    rm = gk.RegionMask.load(aachenShape, srs=gk.srs.EPSG4326, pixelRes=0.001)
    ec = gl.ExclusionCalculator(rm)

    compare(ec.region.mask.shape ,(457, 446) )
    compareF(ec.region.mask.sum() , 90296 )
    compareF(ec.region.mask.std(), 0.496741981394)
    print( "ExclusionCalculator___init__ passed")

def test_ExclusionCalculator_save():

    ec = gl.ExclusionCalculator(aachenShape)

    ec.save("results/save1.tif")
    mat = gk.raster.extractMatrix("results/save1.tif")
    compare( np.nansum(mat-ec.availability), 0 )
    compareF(np.nansum(mat), 28461360)
    compareF(np.nanstd(mat), 77.2323849648)

    print( "ExclusionCalculator_save passed")

def test_ExclusionCalculator_draw():
    ec = gl.ExclusionCalculator(aachenShape)
    
    ec._availability[:, 140:160] = 0
    ec._availability[140:160, :] = 0
    
    ec.draw()
    plt.savefig("results/DrawnImage.png", dpi=200)
    plt.close()

    print( "ExclusionCalculator_draw passed, but outputs need to be checked")


def test_ExclusionCalculator_excludeRasterType():
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, 12)

    compareF(np.nanmean(ec.availability), 82.8033, "excludeRasterType: Single value exclusion")
    compareF(np.nanstd(ec.availability), 37.73514175, "excludeRasterType: Single value exclusion")

    # exclude value range
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, (5,12))

    compareF(np.nanmean(ec.availability), 81.16260529, "excludeRasterType: Value range exclusion")
    compareF(np.nanstd(ec.availability), 39.10104752, "excludeRasterType: Value range exclusion")
    
    # exclude value maximum
    ecMax12 = gl.ExclusionCalculator(aachenShape)
    ecMax12.excludeRasterType(clcRaster, (None,12))

    compareF(np.nanmean(ecMax12.availability), 58.52362442, "excludeRasterType: Value maximum exclusion")
    compareF(np.nanstd(ecMax12.availability), 49.26812363, "excludeRasterType: Value maximum exclusion")
    
    # exclude value minimum
    ecMin13 = gl.ExclusionCalculator(aachenShape)
    ecMin13.excludeRasterType(clcRaster, (13,None))

    compareF(np.nanmean(ecMin13.availability), 41.47637558, "excludeRasterType: Value minimum exclusion")
    compareF(np.nanstd(ecMin13.availability), 49.26812363, "excludeRasterType: Value minimum exclusion")
    
    # Make sure min and max align
    if ((ecMax12.availability[ecMax12.region.mask]>0.5) == (ecMin13.availability[ecMax12.region.mask]>0.5)).any():
        raise RuntimeError("excludeRasterType: minimum and maximum overlap - FAIL")

    if not (np.logical_or( ecMax12.availability[ecMax12.region.mask]>0.5, ecMin13.availability[ecMax12.region.mask]>0.5 ) == ecMin13.region.mask[ecMax12.region.mask]).all():
        raise RuntimeError("excludeRasterType: minimum and maximum summation - FAIL")

    ### Test with a different projection system
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.005)
    ec.excludeRasterType(clcRaster, 12)

    compareF(np.nanmean(ec.availability), 82.95262909, "excludeRasterType: Other SRS")
    compareF(np.nanstd(ec.availability), 32.26681137, "excludeRasterType: Other SRS")

    print( "ExclusionCalculator_excludeRasterType passed")

def test_ExclusionCalculator_excludeVectorType():
    # exclude all features directly
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector)

    compareF(np.nanmean(ec.availability), 76.47581482, "excludeVectorType: Single value exclusion, new srs")
    compareF(np.nanstd(ec.availability), 42.41498947, "excludeVectorType: Single value exclusion, new srs")

    # exclude all features directly, new srs
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.005)
    ec.excludeVectorType(cddaVector)

    compareF(np.nanmean(ec.availability), 76.31578827, "excludeVectorType: Single value exclusion, new srs ")
    compareF(np.nanstd(ec.availability), 42.51445770, "excludeVectorType: Single value exclusion, new srs ")

    # exclude a selection of features
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000")

    compareF(np.nanmean(ec.availability), 86.89811707, "excludeVectorType: Single value exclusion, new srs")
    compareF(np.nanstd(ec.availability), 33.74209595, "excludeVectorType: Single value exclusion, new srs")

    # exclude a selection of features with buffer
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000", buffer=400)

    compareF(np.nanmean(ec.availability), 77.95021057, "excludeVectorType: Single value exclusion, new srs")
    compareF(np.nanstd(ec.availability), 41.45823669, "excludeVectorType: Single value exclusion, new srs")

    print( "ExclusionCalculator_excludeVectorType passed")

def test_ExclusionCalculator_excludePrior():

    # make a prior source 
    pr = gl.core.priors.PriorSource(priorSample)

    # test same srs
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400,None))

    compareF(np.nanmean(ec.availability), 24.77587891, "excludePrior: Same srs")
    compareF(np.nanstd(ec.availability), 43.17109680, "excludePrior: Same srs")

    # test different srs and resolution
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.001)
    ec.excludePrior(pr, value=(400,None))

    compareF(np.nanmean(ec.availability), 24.83173180, "excludePrior: Different srs")
    compareF(np.nanstd(ec.availability), 41.84893036, "excludePrior: Different srs")

    print( "ExclusionCalculator_excludePrior passed")

def test_ExclusionCalculator_excludeRegionEdge():
    # make a prior source 
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None,400))

    ec.excludeRegionEdge(500)

    compareF(np.nanmean(ec.availability), 63.68544388, "excludeRegionEdge")
    compareF(np.nanstd(ec.availability), 48.09062958, "excludeRegionEdge")

    print( "ExclusionCalculator_excludeRegionEdge passed")
    
def test_ExclusionCalculator_shrinkAvailability():
    # make a prior source 
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None,400))

    ec.shrinkAvailability(500)

    compareF(np.nanmean(ec.availability), 41.88655853, "shrinkAvailability")
    compareF(np.nanstd(ec.availability), 49.33732986, "shrinkAvailability")

    print( "ExclusionCalculator_shrinkAvailability passed")
    
def test_ExclusionCalculator_pruneIsolatedAreas():
    # make a prior source 
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None,400))

    ec.pruneIsolatedAreas(12000000)

    compareF(np.nanmean(ec.availability), 65.41215515, "pruneIsolatedAreas")
    compareF(np.nanstd(ec.availability), 47.56538391, "pruneIsolatedAreas")

    print( "ExclusionCalculator_pruneIsolatedAreas passed")


if __name__ == "__main__":
    test_ExclusionCalculator___init__()
    test_ExclusionCalculator_save()
    test_ExclusionCalculator_draw()
    test_ExclusionCalculator_excludeRasterType()
    test_ExclusionCalculator_excludeVectorType()
    test_ExclusionCalculator_excludePrior()
    test_ExclusionCalculator_excludeRegionEdge()
    test_ExclusionCalculator_shrinkAvailability()
    test_ExclusionCalculator_pruneIsolatedAreas()
