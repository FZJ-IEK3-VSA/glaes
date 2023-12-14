import warnings
import matplotlib.pyplot as plt
from os.path import join, dirname, isfile
from osgeo import gdal
import numpy as np
import geokit as gk
import glaes as gl
import pandas as pd
import statistics


TESTDIR = dirname(__file__)
RESULTDIR = join(TESTDIR, "results")
aachenShape = gl._test_data_["aachenShapefile.shp"]
clcRaster = gl._test_data_["clc-aachen_clipped.tif"]
priorSample = gl._test_data_["roads_prior_clip.tif"]
cddaVector = gl._test_data_["CDDA_aachenClipped.shp"]
pointData = gl._test_data_["aachen_points.shp"]

def test_multiple_exclusions():
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)

    # apply exclusions
    ec.excludePrior(pr, value=(None, 400))
    ec.excludeVectorType(cddaVector, where="YEAR>2000")
    ec.excludeRasterType(clcRaster, value=(None, 12))

    assert np.isclose(np.nanmean(ec.availability), 37.1109619141, 1e-6)
    assert np.isclose(np.nanstd(ec.availability), 48.3101692200, 1e-6)

def test_excludePoints():
    ec1 = gl.ExclusionCalculator(aachenShape)
    points = gk.vector.extractFeatures(pointData)

    ec1.excludePoints(source=points, geometryShape="ellipse", direction=45,
                      saveToEC="Test")
    assert np.isclose(ec1.percentAvailable, 95.61485115020298)
    ec1.excludePoints(source=pointData, geometryShape="rectangle", direction=25)
    assert np.isclose(ec1.percentAvailable, 94.36879792512404)
    assert len(ec1._additionalPoints["Test"]["points"]) == 13

def test_ExclusionCalculator___init__():
    # Test by giving a shapefile
    ec = gl.ExclusionCalculator(aachenShape)

    assert ec.region.mask.shape == (509, 304)
    assert np.isclose(ec.region.mask.sum(), 70944)
    assert np.isclose(ec.region.mask.std(), 0.498273451386)

    # Test by giving a region mask
    rm = gk.RegionMask.load(aachenShape, padExtent=5000)
    ec = gl.ExclusionCalculator(rm)

    assert ec.region.mask.shape == (609, 404)
    assert np.isclose(ec.region.mask.sum(), 70944)
    assert np.isclose(ec.region.mask.std(), 0.45299387483)

    # Test by giving a region mask with different resolution and srs
    rm = gk.RegionMask.load(aachenShape, srs=gk.srs.EPSG4326, pixelRes=0.001)
    ec = gl.ExclusionCalculator(rm)

    assert ec.region.mask.shape == (457, 446)
    assert np.isclose(ec.region.mask.sum(), 90296)
    assert np.isclose(ec.region.mask.std(), 0.496741981394)


def test_ExclusionCalculator_save():

    ec = gl.ExclusionCalculator(aachenShape)

    ec.save(join(RESULTDIR, "save1.tif"))
    mat = gk.raster.extractMatrix(join(RESULTDIR, "save1.tif"))
    assert np.nansum(mat - ec.availability) == 0
    assert np.isclose(np.nansum(mat), 28461360)
    assert np.isclose(np.nanstd(mat), 77.2323849648)


def test_ExclusionCalculator_draw():
    ec = gl.ExclusionCalculator(aachenShape)

    ec._availability[:, 140:160] = 0
    ec._availability[140:160, :] = 0

    ec.draw()
    plt.savefig(join(RESULTDIR, "DrawnImage.png"), dpi=200)
    plt.close()


def test_ExclusionCalculator_excludeRasterType():
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, 12)

    assert np.isclose(np.nanmean(ec.availability), 82.8033)
    assert np.isclose(np.nanstd(ec.availability), 37.73514175)

    # exclude value range
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeRasterType(clcRaster, (5, 12))

    assert np.isclose(np.nanmean(ec.availability), 81.16260529)
    assert np.isclose(np.nanstd(ec.availability), 39.10104752)

    # Exclude iterable (should have the same result as the test above)
    ec = gl.ExclusionCalculator(gl._test_data_["aachenShapefile.shp"], srs=gk.srs.EPSG3035, pixelRes=100)
    ec.excludeRasterType(
        gl._test_data_['clc-aachen_clipped.tif'],
        value=[5, 6, 7, 8, 9, 10, 11, 12])

    assert np.isclose(np.nanmean(ec.availability), 81.16260529)
    assert np.isclose(np.nanstd(ec.availability), 39.10104752)

    # exclude value maximum
    ecMax12 = gl.ExclusionCalculator(aachenShape)
    ecMax12.excludeRasterType(clcRaster, (None, 12))

    assert np.isclose(np.nanmean(ecMax12.availability), 58.52362442)
    assert np.isclose(np.nanstd(ecMax12.availability), 49.26812363)

    # exclude value minimum
    ecMin13 = gl.ExclusionCalculator(aachenShape)
    ecMin13.excludeRasterType(clcRaster, (13, None))

    assert np.isclose(np.nanmean(ecMin13.availability), 41.47637558)
    assert np.isclose(np.nanstd(ecMin13.availability), 49.26812363)

    # Make sure min and max align
    s1 = ecMax12.availability[ecMax12.region.mask] > 0
    s2 = ecMin13.availability[ecMax12.region.mask] > 0
    assert np.logical_xor(s1, s2).all()

    # Test with a different projection system
    # exclude single value
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.005)
    ec.excludeRasterType(clcRaster, 12)
    print("AVAIL MEAN:", np.nanmean(ec.availability))

    assert np.isclose(np.nanmean(ec.availability), 82.95262909)
    assert np.isclose(np.nanstd(ec.availability), 32.26681137)

    # Test with complex value input
    ec = gl.ExclusionCalculator(gl._test_data_["aachenShapefile.shp"], srs='latlon', pixelRes=0.005)
    ec.excludeRasterType(
        gl._test_data_['clc-aachen_clipped.tif'],
        value="[-2),[5-7),12,(22-26],29,33,[40-]")

    assert np.isclose(np.nanmean(ec.availability), 49.5872573853)
    assert np.isclose(np.nanstd(ec.availability), 41.2754364014)

    # Test with intermediate functionaliy (creation and re-use)
    for i in range(2):
        ec = gl.ExclusionCalculator(
            gl._test_data_["aachenShapefile.shp"],
            srs='latlon',
            pixelRes=0.005,)
        ec.excludeRasterType(
            gl._test_data_['clc-aachen_clipped.tif'],
            value="[-2),[5-7),12,(22-26],29,33,[40-]",
            intermediate=join(RESULTDIR, "exclude_raster_intermediate.tif"))

        assert isfile(join(RESULTDIR, "exclude_raster_intermediate.tif"))
        assert np.isclose(np.nanmean(ec.availability), 49.5872573853)
        assert np.isclose(np.nanstd(ec.availability), 41.2754364014)


def test_ExclusionCalculator_excludeVectorType():
    # exclude all features directly
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector)

    assert np.isclose(np.nanmean(ec.availability), 76.47581482)
    assert np.isclose(np.nanstd(ec.availability), 42.41498947)

    # exclude all features directly, new srs
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.005)
    ec.excludeVectorType(cddaVector)

    assert np.isclose(np.nanmean(ec.availability), 76.31578827)
    assert np.isclose(np.nanstd(ec.availability), 42.51445770)

    # exclude a selection of features
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000")

    assert np.isclose(np.nanmean(ec.availability), 86.89811707)
    assert np.isclose(np.nanstd(ec.availability), 33.74209595)

    # exclude a selection of features with buffer
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludeVectorType(cddaVector, where="YEAR>2000", buffer=400)

    assert np.isclose(np.nanmean(ec.availability), 77.95021057)
    assert np.isclose(np.nanstd(ec.availability), 41.45823669)

    # test with intermediate functionality
    for i in range(2):
        ec = gl.ExclusionCalculator(aachenShape)
        ec.excludeVectorType(
            cddaVector,
            where="YEAR>2000",
            buffer=400,
            intermediate=join(RESULTDIR, "exclude_vector_intermediate.tif"))

        assert isfile(join(RESULTDIR, "exclude_vector_intermediate.tif"))
        assert np.isclose(np.nanmean(ec.availability), 77.95021057)
        assert np.isclose(np.nanstd(ec.availability), 41.45823669)


def test_ExclusionCalculator_excludePrior():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)

    # test same srs
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))

    assert np.isclose(np.nanmean(ec.availability), 24.77587891)
    assert np.isclose(np.nanstd(ec.availability), 43.17109680)

    # test different srs and resolution
    ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelRes=0.001)
    ec.excludePrior(pr, value=(400, None))

    assert np.isclose(np.nanmean(ec.availability), 24.83173180)
    assert np.isclose(np.nanstd(ec.availability), 41.84893036)


def test_ExclusionCalculator_excludeSet():
    ec = gl.ExclusionCalculator(aachenShape)
    exclusion_set = pd.read_csv(gl._test_data_["sample_exclusion_set.csv"])
    ec.excludeSet(
        exclusion_set=exclusion_set,
        clc=gl._test_data_['clc-aachen_clipped.tif'],
        osm_roads=gl._test_data_["aachenRoads.shp"],
        verbose=False,
    )

    assert np.isclose(np.nanmean(ec.availability), 15.230323)
    assert np.isclose(np.nanstd(ec.availability), 35.931458)


def test_ExclusionCalculator_excludeRegionEdge():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None, 400))

    ec.excludeRegionEdge(500)

    assert np.isclose(np.nanmean(ec.availability), 63.68544388)
    assert np.isclose(np.nanstd(ec.availability), 48.09062958)


def test_ExclusionCalculator_shrinkAvailability():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None, 400))

    ec.shrinkAvailability(500)

    assert np.isclose(np.nanmean(ec.availability), 41.88655853)
    assert np.isclose(np.nanstd(ec.availability), 49.33732986)


def test_ExclusionCalculator_pruneIsolatedAreas():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(None, 400))

    ec.pruneIsolatedAreas(12000000)

    assert np.isclose(np.nanmean(ec.availability), 65.41215515)
    assert np.isclose(np.nanstd(ec.availability), 47.56538391)


def test_ExclusionCalculator_distributeItems():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))

    # Do a regular distribution
    ec.distributeItems(1000, output=join(RESULTDIR, "distributeItems1.shp"),
                       outputSRS=3035)
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems1.shp"))
    assert geoms.shape[0] == 287

    minDist = 1000000
    for gi in range(geoms.shape[0] - 1):
        for gj in range(gi + 1, geoms.shape[0]):
            d = geoms.geom[gi].Distance(geoms.geom[gj])
            if d < minDist:
                minDist = d
                I = (gi, gj)

    assert minDist >= 999

    # Do an axial distribution
    ec.distributeItems((1000, 300), axialDirection=180,
                       output=join(RESULTDIR, "distributeItems2.shp"),
                       outputSRS=3035)
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems2.shp"))
    assert geoms.shape[0] == 882

    x = np.array([g.GetX() for g in geoms.geom])
    y = np.array([g.GetY() for g in geoms.geom])

    for gi in range(geoms.shape[0] - 1):
        d = (x[gi] - x[gi + 1:])**2 / 1000**2 + (y[gi] - y[gi + 1:])**2 / 300**2
        assert (d >= 1).all()  # Axial objects too close

    # Do make areas
    ec.distributeItems(2000, asArea=True,
                       output=join(RESULTDIR, "distributeItems3.shp"),
                       outputSRS=4326)
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems3.shp"))

    assert np.isclose(geoms.shape[0], 97)
    # Tests below are failing for 3.0.0<=gdal<3.4.0 due to problems when
    # polygonizing
    assert np.isclose(geoms.area.mean(), 0.000230714164474)
    assert np.isclose(geoms.area.std(), 8.2766693979e-05)

    # Do a variable separation distance placement
    ec = gl.ExclusionCalculator(
        gl._test_data_['aachenShapefile.shp'],
        pixelRes=25,
        srs="LAEA")

    ec.excludeRasterType(
        gl._test_data_['clc-aachen_clipped.tif'],
        value=(1, 2),
        invert=True)

    mat = np.zeros_like(ec.region.mask, dtype=np.uint16)
    for i in range(mat.shape[0]):
        mat[i, :] = (300 - 50) * i / mat.shape[0] + 100

    ras = ec.region.createRaster(data=mat)

    points = ec.distributeItems(
        separation=5,
        sepScaling=ras,
        _stamping=False)

    assert points.shape[0] == 335

    points = ec.distributeItems(
        separation=(8, 3),
        sepScaling=ras,
        axialDirection=0)
    assert points.shape[0] == 389


def test_ExclusionCalculator_distributeAreas():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))

    # Do a regular distribution and subsequent area assignment
    ec.distributeItems(1000, outputSRS=3035)
    ec.distributeAreas(minArea=20000)

    areas = [g.Area() for g in ec._areas]

    assert np.isclose(sum(areas), 175768748.40184686)
    assert np.isclose(statistics.stdev(areas), 218353.60307113524)
    assert np.isclose(statistics.mean(areas), 612434.6634210692)


def test_ExclusionCalculator_saveAreas():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))

    # Do a regular distribution and subsequent area assignment
    ec.distributeItems(1000, outputSRS=3035)
    ec.distributeAreas(minArea=20000)

    # save df via saveAreas() and reload for comparison
    ec.saveAreas(join(RESULTDIR, "saveAreas.shp"), 
                      srs=4326, 
                      savePolygons=True,
                      data=np.arange(len(ec._areas)),
                      )                
    df_saveAreas=gk.vector.extractFeatures(join(RESULTDIR, "saveAreas.shp"))
    
    df_inRamAreas=ec.saveAreas(srs=4326, 
                      savePolygons=True,
                      data=np.arange(len(ec._areas)),
                      )    
    # assert that values retrieved from saved file match
    assert np.isclose(df_saveAreas.area_m2.sum(), 175768748.40184686)
    assert np.isclose(df_saveAreas.area_m2.mean(), 612434.6634210692)
    assert np.isclose(df_saveAreas.area_m2.std(), 218353.60307113524)
    assert (len(df_saveAreas) == 287)
    
    # assert that values from df stored in variable match
    assert np.isclose(df_inRamAreas.area_m2.sum(), 175768748.40184686)
    assert np.isclose(df_inRamAreas.area_m2.mean(), 612434.6634210692)
    assert np.isclose(df_inRamAreas.area_m2.std(), 218353.60307113524)
    assert (len(df_inRamAreas) == 287)
    
