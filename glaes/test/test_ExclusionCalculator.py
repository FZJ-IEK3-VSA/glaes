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
