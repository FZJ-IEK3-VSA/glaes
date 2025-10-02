import statistics
import warnings
from copy import copy
from os.path import dirname, isfile, join

import geokit as gk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal

import glaes as gl

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

    ec1.excludePoints(source=points, geometryShape="ellipse", direction=45, saveToEC="Test")
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
    ec.excludeRasterType(gl._test_data_["clc-aachen_clipped.tif"], value=[5, 6, 7, 8, 9, 10, 11, 12])

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
    ec = gl.ExclusionCalculator(aachenShape, srs="latlon", pixelRes=0.005)
    ec.excludeRasterType(clcRaster, 12)
    print("AVAIL MEAN:", np.nanmean(ec.availability))

    assert np.isclose(np.nanmean(ec.availability), 82.95262909)
    assert np.isclose(np.nanstd(ec.availability), 32.26681137)

    # Test with complex value input
    ec = gl.ExclusionCalculator(gl._test_data_["aachenShapefile.shp"], srs="latlon", pixelRes=0.005)
    ec.excludeRasterType(
        gl._test_data_["clc-aachen_clipped.tif"],
        value="[-2),[5-7),12,(22-26],29,33,[40-]",
    )

    assert np.isclose(np.nanmean(ec.availability), 49.5872573853)
    assert np.isclose(np.nanstd(ec.availability), 41.2754364014)

    # Test with intermediate functionaliy (creation and re-use)
    for i in range(2):
        ec = gl.ExclusionCalculator(
            gl._test_data_["aachenShapefile.shp"],
            srs="latlon",
            pixelRes=0.005,
        )
        ec.excludeRasterType(
            gl._test_data_["clc-aachen_clipped.tif"],
            value="[-2),[5-7),12,(22-26],29,33,[40-]",
            intermediate=join(RESULTDIR, "exclude_raster_intermediate.tif"),
        )

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
    ec = gl.ExclusionCalculator(aachenShape, srs="latlon", pixelRes=0.005)
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
            intermediate=join(RESULTDIR, "exclude_vector_intermediate.tif"),
        )

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
    ec = gl.ExclusionCalculator(aachenShape, srs="latlon", pixelRes=0.001)
    ec.excludePrior(pr, value=(400, None))

    assert np.isclose(np.nanmean(ec.availability), 24.83173180)
    assert np.isclose(np.nanstd(ec.availability), 41.84893036)


def test_ExclusionCalculator_excludeSet():
    ec = gl.ExclusionCalculator(aachenShape)
    exclusion_set = pd.read_csv(gl._test_data_["sample_exclusion_set.csv"])
    ec.excludeSet(
        exclusion_set=exclusion_set,
        clc=gl._test_data_["clc-aachen_clipped.tif"],
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
    # create a copy to repeat the process
    ec2 = copy(ec)

    # Do a regular distribution
    ec.distributeItems(1000, output=join(RESULTDIR, "distributeItems1.shp"), outputSRS=3035)
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

    ec2.distributeItems(
        1000,
        output=join(RESULTDIR, "distributeItems1b.shp"),
        outputSRS=3035,
        avoidRegionBorders=True,
    )
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems1b.shp"))
    assert geoms.shape[0] == 252
    # make sure that all placements fall within the region less the 500m border corridor
    assert (
        gk.vector.extractFeatures(
            join(RESULTDIR, "distributeItems1b.shp"),
            geoms=gk.drawGeoms(ec.region.geometry.Buffer(-500)),
        ).shape[0]
        == 252
    )

    # Do an axial distribution
    ec.distributeItems(
        (1000, 300),
        axialDirection=180,
        output=join(RESULTDIR, "distributeItems2.shp"),
        outputSRS=3035,
    )
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems2.shp"))
    assert geoms.shape[0] == 882

    x = np.array([g.GetX() for g in geoms.geom])
    y = np.array([g.GetY() for g in geoms.geom])

    for gi in range(geoms.shape[0] - 1):
        d = (x[gi] - x[gi + 1 :]) ** 2 / 1000**2 + (y[gi] - y[gi + 1 :]) ** 2 / 300**2
        assert (d >= 1).all()  # Axial objects too close

    # Do make areas
    ec.distributeItems(
        2000,
        asArea=True,
        output=join(RESULTDIR, "distributeItems3.shp"),
        outputSRS=4326,
    )
    geoms = gk.vector.extractFeatures(join(RESULTDIR, "distributeItems3.shp"))

    assert np.isclose(geoms.shape[0], 97)
    # Tests below are failing for 3.0.0<=gdal<3.4.0 due to problems when
    # polygonizing
    assert np.isclose(geoms.area.mean(), 0.000230714164474)
    assert np.isclose(geoms.area.std(), 8.2766693979e-05)

    # Do a variable separation distance placement
    ec = gl.ExclusionCalculator(gl._test_data_["aachenShapefile.shp"], pixelRes=25, srs="LAEA")

    ec.excludeRasterType(gl._test_data_["clc-aachen_clipped.tif"], value=(1, 2), invert=True)

    mat = np.zeros_like(ec.region.mask, dtype=np.uint16)
    for i in range(mat.shape[0]):
        mat[i, :] = (300 - 50) * i / mat.shape[0] + 100

    ras = ec.region.createRaster(data=mat)

    points = ec.distributeItems(separation=5, sepScaling=ras, _stamping=False)

    assert points.shape[0] == 335

    points = ec.distributeItems(separation=(8, 3), sepScaling=ras, axialDirection=0)
    assert points.shape[0] == 389


def test_ExclusionCalculator_distributeAreas():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))

    # Do a regular distribution and subsequent area assignment
    ec.distributeItems(1000, outputSRS=3035)
    items = ec.itemCoords
    item_cords_check = [
        [4053400.01, 3099899.99],
        [4052300.01, 3099799.99],
        [4054700.01, 3099399.99],
        [4051500.01, 3099175.0],
        [4053500.01, 3098900.01],
        [4055425.0, 3098700.01],
        [4054375.0, 3098400.01],
        [4051000.01, 3098300.01],
        [4050600.01, 3097375.0],
        [4050825.0, 3096400.01],
        [4050200.01, 3095599.99],
        [4050925.0, 3094900.01],
        [4048800.01, 3094799.99],
        [4049799.99, 3094675.0],
        [4048100.01, 3094075.0],
        [4051375.0, 3094000.01],
        [4050299.99, 3093800.01],
        [4052375.0, 3093899.99],
        [4047500.01, 3093250.0],
        [4053099.99, 3093200.01],
        [4048499.99, 3093150.0],
        [4051300.01, 3093000.01],
        [4053825.0, 3092500.01],
        [4046900.01, 3092425.0],
        [4047899.99, 3092325.0],
        [4052025.0, 3092300.01],
        [4054699.99, 3092000.01],
        [4051000.01, 3091999.99],
        [4046100.01, 3091800.01],
        [4055625.0, 3091600.01],
        [4050000.01, 3091299.99],
        [4051450.0, 3091100.01],
        [4045200.01, 3091099.99],
        [4046725.0, 3091000.01],
        [4054700.01, 3091000.01],
        [4049200.01, 3090675.0],
        [4055575.0, 3090500.01],
        [4043300.01, 3090399.99],
        [4053800.01, 3090299.99],
        [4044900.01, 3090125.0],
        [4049950.0, 3090000.01],
        [4054725.0, 3089900.01],
        [4042400.01, 3089799.99],
        [4049000.01, 3089675.0],
        [4056025.0, 3089600.01],
        [4053200.01, 3089475.0],
        [4041100.01, 3089399.99],
        [4044300.01, 3089199.99],
        [4049499.99, 3088800.01],
        [4056650.0, 3088800.01],
        [4052700.01, 3088600.01],
        [4040500.01, 3088599.99],
        [4045025.0, 3088500.01],
        [4048500.01, 3088599.99],
        [4056200.01, 3087900.01],
        [4049125.0, 3087800.01],
        [4044400.01, 3087700.01],
        [4053150.0, 3087700.01],
        [4040600.01, 3087600.01],
        [4048200.01, 3087400.01],
        [4057075.0, 3087400.01],
        [4052100.01, 3087399.99],
        [4062400.01, 3087299.99],
        [4045225.0, 3087125.0],
        [4059900.01, 3086999.99],
        [4061400.01, 3086999.99],
        [4056200.01, 3086899.99],
        [4041050.0, 3086700.01],
        [4063225.0, 3086725.0],
        [4044300.01, 3086699.99],
        [4051400.01, 3086675.0],
        [4057700.01, 3086600.01],
        [4058699.99, 3086699.99],
        [4048650.0, 3086500.01],
        [4052375.0, 3086425.0],
        [4055200.01, 3086399.99],
        [4060625.0, 3086300.01],
        [4062125.0, 3086300.01],
        [4047700.01, 3086175.0],
        [4056825.0, 3086100.01],
        [4059425.0, 3086000.01],
        [4041499.99, 3085800.01],
        [4043700.01, 3085875.0],
        [4051100.01, 3085700.01],
        [4054300.01, 3085799.99],
        [4058150.0, 3085700.01],
        [4062950.0, 3085725.0],
        [4061350.0, 3085600.01],
        [4044599.99, 3085425.0],
        [4055225.0, 3085400.01],
        [4047200.01, 3085300.01],
        [4060150.0, 3085300.01],
        [4042300.01, 3085200.01],
        [4048199.99, 3085275.0],
        [4056225.0, 3085299.99],
        [4053400.01, 3085199.99],
        [4058975.0, 3085100.01],
        [4043275.0, 3084950.0],
        [4051725.0, 3084900.01],
        [4054325.0, 3084800.01],
        [4045175.0, 3084600.01],
        [4046175.0, 3084699.99],
        [4056950.0, 3084600.01],
        [4060875.0, 3084600.01],
        [4044175.0, 3084500.01],
        [4050600.01, 3084599.99],
        [4052650.0, 3084500.01],
        [4047650.0, 3084400.01],
        [4048800.01, 3084475.0],
        [4037400.01, 3084399.99],
        [4055900.01, 3084299.99],
        [4049699.99, 3084025.0],
        [4051325.0, 3083900.01],
        [4046625.0, 3083800.01],
        [4044625.0, 3083600.01],
        [4048275.0, 3083600.01],
        [4042900.01, 3083599.99],
        [4056350.0, 3083400.01],
        [4045800.01, 3083225.0],
        [4055200.01, 3083299.99],
        [4047350.0, 3083100.01],
        [4049150.0, 3083100.01],
        [4050600.01, 3083199.99],
        [4043725.0, 3083025.0],
        [4038800.01, 3082899.99],
        [4042200.01, 3082799.99],
        [4051475.0, 3082700.01],
        [4048225.0, 3082600.01],
        [4054400.01, 3082599.99],
        [4044525.0, 3082400.01],
        [4055375.0, 3082300.01],
        [4043025.0, 3082225.0],
        [4045525.0, 3082250.0],
        [4039500.01, 3082175.0],
        [4046500.01, 3082025.0],
        [4047499.99, 3081900.01],
        [4051000.01, 3081800.01],
        [4048675.0, 3081700.01],
        [4042200.01, 3081650.0],
        [4043900.01, 3081600.01],
        [4054425.0, 3081600.01],
        [4045200.01, 3081300.01],
        [4051875.0, 3081300.01],
        [4040900.01, 3081199.99],
        [4049499.99, 3081125.0],
        [4042975.0, 3081000.01],
        [4046175.0, 3081075.0],
        [4047175.0, 3080950.0],
        [4041775.0, 3080700.01],
        [4044350.0, 3080700.01],
        [4048300.01, 3080750.0],
        [4054875.0, 3080700.01],
        [4051000.01, 3080699.99],
        [4045275.0, 3080300.01],
        [4043425.0, 3080100.01],
        [4049075.0, 3080100.01],
        [4047499.99, 3080000.01],
        [4055499.99, 3079900.01],
        [4050400.01, 3079875.0],
        [4051450.0, 3079800.01],
        [4044600.01, 3079550.0],
        [4046600.01, 3079550.0],
        [4048300.01, 3079400.01],
        [4053400.01, 3079499.99],
        [4042800.01, 3079300.01],
        [4049600.01, 3079225.0],
        [4054325.0, 3079100.01],
        [4043725.0, 3078900.01],
        [4045375.0, 3078900.01],
        [4055325.0, 3078900.01],
        [4052700.01, 3078775.0],
        [4051500.01, 3078699.99],
        [4046299.99, 3078500.01],
        [4048900.01, 3078500.01],
        [4050299.99, 3078500.01],
        [4042700.01, 3078300.01],
        [4044550.0, 3078325.0],
        [4053599.99, 3078325.0],
        [4054575.0, 3078100.01],
        [4052125.0, 3077900.01],
        [4043575.0, 3077800.01],
        [4051025.0, 3077800.01],
        [4045350.0, 3077700.01],
        [4046750.0, 3077600.01],
        [4049350.0, 3077600.01],
        [4041900.01, 3077499.99],
        [4048300.01, 3077399.99],
        [4042825.0, 3077100.01],
        [4044600.01, 3077025.0],
        [4046100.01, 3076825.0],
        [4041400.01, 3076625.0],
        [4047075.0, 3076600.01],
        [4048750.0, 3076500.01],
        [4045175.0, 3076200.01],
        [4047800.01, 3075900.01],
        [4046600.01, 3075700.01],
        [4041300.01, 3075625.0],
        [4044600.01, 3075375.0],
        [4047325.0, 3075000.01],
        [4048199.99, 3074500.01],
        [4048650.0, 3073600.01],
        [4049475.0, 3073025.0],
        [4049000.01, 3072125.0],
        [4049975.0, 3071900.01],
        [4050799.99, 3071325.0],
        [4049400.01, 3071075.0],
        [4051599.99, 3070700.01],
        [4050900.01, 3069975.0],
        [4051799.99, 3069525.0],
        [4051100.01, 3068800.01],
        [4058900.01, 3068799.99],
        [4051975.0, 3068300.01],
        [4058100.01, 3068099.99],
        [4059075.0, 3067800.01],
        [4051200.01, 3067650.0],
        [4057500.01, 3067275.0],
        [4063800.01, 3066999.99],
        [4064799.99, 3066975.0],
        [4050800.01, 3066725.0],
        [4051750.0, 3066400.01],
        [4057000.01, 3066400.01],
        [4063000.01, 3066299.99],
        [4065450.0, 3066200.01],
        [4064250.0, 3066100.01],
        [4052475.0, 3065700.01],
        [4057450.0, 3065500.01],
        [4062300.01, 3065575.0],
        [4053350.0, 3065200.01],
        [4054225.0, 3064700.01],
        [4056800.01, 3064725.0],
        [4058075.0, 3064700.01],
        [4062799.99, 3064700.01],
        [4061800.01, 3064499.99],
        [4055150.0, 3064300.01],
        [4058999.99, 3064300.01],
        [4059999.99, 3064299.99],
        [4057400.01, 3063900.01],
        [4060875.0, 3063800.01],
        [4062250.0, 3063600.01],
        [4060100.01, 3063150.0],
        [4061325.0, 3062900.01],
        [4059500.01, 3062325.0],
        [4061775.0, 3062000.01],
        [4055700.01, 3061699.99],
        [4058800.01, 3061600.01],
        [4060900.01, 3061500.01],
        [4062225.0, 3061100.01],
        [4055500.01, 3060700.01],
        [4058100.01, 3060799.99],
        [4059099.99, 3060625.0],
        [4062950.0, 3060400.01],
        [4056125.0, 3059900.01],
        [4057300.01, 3059999.99],
        [4063825.0, 3059900.01],
        [4058299.99, 3059800.01],
        [4064825.0, 3059799.99],
        [4056750.0, 3059100.01],
        [4055700.01, 3058975.0],
        [4063800.01, 3058900.01],
        [4064799.99, 3058800.01],
        [4053700.01, 3058199.99],
        [4056199.99, 3058100.01],
        [4065250.0, 3057900.01],
        [4054625.0, 3057800.01],
        [4051800.01, 3057699.99],
        [4052799.99, 3057599.99],
        [4055499.99, 3057300.01],
        [4050900.01, 3057250.0],
        [4053725.0, 3057200.01],
        [4056499.99, 3057125.0],
        [4054599.99, 3056700.01],
        [4053700.01, 3056200.01],
        [4054575.0, 3055700.01],
        [4053500.01, 3055200.01],
        [4055399.99, 3055125.0],
        [4054375.0, 3054700.01],
        [4055975.0, 3054300.01],
        [4055000.01, 3053900.01],
        [4056599.99, 3053500.01],
        [4057575.0, 3053275.0],
        [4055625.0, 3053100.01],
        [4058475.0, 3052825.0],
        [4059399.99, 3052425.0],
        [4060099.99, 3051700.01],
        [4060550.0, 3050800.01],
        [4061700.01, 3050499.99],
        [4060100.01, 3049900.01],
    ]
    assert np.allclose(items, item_cords_check), str(items)
    ec.distributeAreas(minArea=20000)

    areas = [g.Area() for g in ec._areas]
    areas_check = [
        299996.80000653863,
        459996.20000457764,
        399996.400005281,
        509995.80000677705,
        809995.600007236,
        539995.7999988496,
        289997.4000029564,
        539995.8000081778,
        429996.4000091255,
        399996.4000031948,
        549995.2000051141,
        489996.400005877,
        389996.6000059247,
        599996.0000049472,
        639995.2000076473,
        659995.6000063121,
        99998.2000041902,
        729995.600006491,
        619995.4000053704,
        349995.80000805855,
        419996.2000038326,
        799995.2000052929,
        489996.4000008702,
        539996.0000052452,
        669995.6000064015,
        889995.0000112057,
        539995.6000089049,
        149997.8000048399,
        489996.00000566244,
        409995.80000543594,
        659994.8000019789,
        169997.80000463128,
        699995.8000069857,
        259997.40000462532,
        389996.400005579,
        879994.8000019491,
        439996.0000114441,
        539995.6000076532,
        589996.0000041425,
        659995.4000091553,
        429996.40000513196,
        479996.20000576973,
        489995.80000552535,
        499996.40000480413,
        409996.6000083387,
        449995.80000922084,
        439996.6000055671,
        949993.4000039995,
        919995.2000057101,
        379996.6000034511,
        829995.000007242,
        209997.40000468493,
        849995.1999984682,
        559996.4000057578,
        589996.0000033379,
        739996.000002563,
        659995.8000086546,
        859994.6000086069,
        379996.6000043154,
        469996.20000234246,
        469996.2000056505,
        489996.00000452995,
        599995.8000041842,
        469996.20000594854,
        609995.0000051856,
        609995.6000101566,
        509996.20000863075,
        599995.8000079393,
        489996.2000063658,
        419996.6000072062,
        1019993.0000071526,
        729994.8000079691,
        869994.4000103474,
        599995.8000060916,
        509996.2000077367,
        879995.4000083804,
        499995.00001105666,
        529996.2000012994,
        889994.8000103831,
        469995.80000448227,
        729995.4000051022,
        449996.20000326633,
        559994.6000172496,
        879993.4000111818,
        799994.8000076711,
        679995.4000041485,
        579995.8000041544,
        669994.4000084996,
        1019994.4000059068,
        579996.4000054002,
        729996.0000048876,
        339997.00000330806,
        509995.6000081897,
        349997.00000327826,
        549996.0000081062,
        569995.6000029147,
        609995.400005877,
        519996.0000026226,
        769995.2000021636,
        559995.8000070453,
        669995.8000069261,
        449996.2000038326,
        849995.0000028312,
        469995.8000035882,
        629994.8000096977,
        669995.8000110686,
        989994.6000162065,
        349996.0000053644,
        309997.0000062883,
        309997.400005579,
        949993.4000076354,
        499996.000004977,
        399996.4000056684,
        869994.6000082195,
        909995.2000082135,
        419996.6000045538,
        309996.60000795126,
        679996.0000047982,
        849993.8000046015,
        649995.2000056207,
        159998.20000460744,
        449996.0000067055,
        639995.6000044644,
        79997.80001157522,
        179997.20000508428,
        249997.40000596642,
        269997.0000053346,
        849994.8000064194,
        419996.80000293255,
        909995.0000053048,
        189996.80001175404,
        209997.2000041008,
        1049994.800006032,
        409996.4000047147,
        539996.0000034869,
        609996.0000066757,
        639995.6000037193,
        1029994.8000085354,
        719995.200006932,
        1039995.0000062287,
        879994.8000094295,
        589996.2000067234,
        669994.8000112176,
        559995.0000061095,
        729994.8000028431,
        569996.000004977,
        599995.4000059366,
        679994.6000048518,
        499996.40000489354,
        839993.4000118971,
        899995.4000032544,
        599994.4000124633,
        569995.8000075817,
        809995.2000097036,
        669994.0000034869,
        709994.8000077009,
        219997.40000462532,
        679995.6000064313,
        589996.00000453,
        899995.200006187,
        609995.6000060141,
        379996.4000098407,
        609996.2000043988,
        669995.6000085771,
        549995.800010562,
        389996.8000046909,
        509995.8000063598,
        679995.6000090837,
        879994.6000161171,
        499994.80001044273,
        499995.4000046849,
        529996.8000030518,
        449996.20000836253,
        669996.0000000894,
        289997.20000600815,
        479996.20000192523,
        349997.00000584126,
        769995.4000039101,
        979995.6000038981,
        749994.6000014246,
        709995.4000100791,
        999994.2000086904,
        549996.0000026226,
        629995.2000088096,
        849994.0000010729,
        469996.0000050068,
        869995.2000025809,
        889995.0000179112,
        539996.2000050545,
        659995.8000013232,
        349997.4000042081,
        599995.800005436,
        449996.00000825524,
        749995.8000060022,
        659994.6000081003,
        799995.6000061035,
        29998.800011873245,
        39998.400015711784,
        399996.4000029564,
        729995.800004065,
        1069994.000008285,
        569995.4000090659,
        669995.2000105381,
        539994.6000123024,
        889994.8000070453,
        459996.20000356436,
        919994.4000043273,
        819994.8000080884,
        769995.6000036895,
        819993.0000061989,
        469996.20000863075,
        509996.00000333786,
        469996.20000493526,
        539994.8000078797,
        349996.80000656843,
        519995.8000075817,
        799993.200017035,
        1009994.0000069141,
        819994.6000113189,
        459994.80001983047,
        809994.2000137866,
        649994.8000076115,
        459996.6000061929,
        919995.0000043511,
        469996.4000072181,
        509996.2000055015,
        419996.40000629425,
        1029994.8000033796,
        669993.6000129282,
        539995.4000089169,
        489995.600007534,
        639995.4000092149,
        479995.6000087261,
        679995.6000086069,
        699996.0000044405,
        669995.000007987,
        659995.4000098705,
        519995.60000389814,
        679994.6000057757,
        329997.2000038028,
        479996.200006485,
        509995.8000038266,
        749995.4000048637,
        829995.6000034809,
        569995.8000099063,
        1029994.4000047743,
        659993.0000112951,
        929995.0000048876,
        1159994.4000027478,
        1139994.600005567,
        799994.2000061274,
        849995.400008738,
        359997.0000051558,
        749995.2000080347,
        939994.8000095189,
        649994.8000075221,
        759994.400006175,
        619994.6000089943,
        939995.0000063181,
        979995.2000039518,
        759994.4000172615,
        929994.2000090778,
        519995.8000051379,
        549995.8000026345,
        1059994.8000062704,
        429996.40000858903,
        649995.2000028789,
        679994.2000055313,
        579995.8000065684,
        359997.00000444055,
        669995.200006187,
        619995.4000053406,
        549995.6000027657,
        499995.7999997139,
        359997.2000050247,
        619995.6000048518,
        509996.4000057876,
        489996.20000210404,
        529996.4000048637,
        1189994.4000097811,
        1159992.8000104427,
        839995.8000059128,
        539996.2000050545,
        829995.2000066638,
        529996.4000044465,
        409996.60000661016,
        859995.0000011027,
    ]

    assert len(areas) == len(areas_check)
    assert np.allclose(areas, areas_check), str(areas)
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
    ec.saveAreas(
        join(RESULTDIR, "saveAreas.shp"),
        srs=4326,
        savePolygons=True,
        data=np.arange(len(ec._areas)),
    )
    df_saveAreas = gk.vector.extractFeatures(join(RESULTDIR, "saveAreas.shp"))

    df_inRamAreas = ec.saveAreas(
        srs=4326,
        savePolygons=True,
        data=np.arange(len(ec._areas)),
    )
    # assert that values retrieved from saved file match
    assert np.isclose(df_saveAreas.area_m2.sum(), 175768748.40184686)
    assert np.isclose(df_saveAreas.area_m2.mean(), 612434.6634210692)
    assert np.isclose(df_saveAreas.area_m2.std(), 218353.60307113524)
    assert len(df_saveAreas) == 287

    # assert that values from df stored in variable match
    assert np.isclose(df_inRamAreas.area_m2.sum(), 175768748.40184686)
    assert np.isclose(df_inRamAreas.area_m2.mean(), 612434.6634210692)
    assert np.isclose(df_inRamAreas.area_m2.std(), 218353.60307113524)
    assert len(df_inRamAreas) == 287


def test_percentAvailableAreaGeometries():
    # make a prior source
    pr = gl.core.priors.PriorSource(priorSample)
    ec = gl.ExclusionCalculator(aachenShape)
    ec.excludePrior(pr, value=(400, None))
    ec.distributeItems(separation=1000, outputSRS=3035)
    ec.distributeAreas()
    assert np.isclose(ec.percentAvailableAreaGeometries, 24.740465043104006)


if __name__ == "__main__":
    test_ExclusionCalculator_distributeAreas()
