import warnings
from os.path import join, dirname
from osgeo import gdal
import numpy as np
import geokit as gk
import pytest

import glaes as gl
from glaes.core.priors import PriorSource, PriorSet

TESTDIR = dirname(__file__)
DATADIR = join(TESTDIR, "data")
RESULTDIR = join(TESTDIR, "results")

aachenShape = gl._test_data_["aachenShapefile.shp"]
clcRaster = gl._test_data_["clc-aachen_clipped.tif"]
priorSample = gl._test_data_["roads_prior_clip.tif"]
cddaVector = gl._test_data_["CDDA_aachenClipped.shp"]


def test_Prior___init__():
    # Test success
    p = PriorSource(priorSample)

    assert p.path == priorSample

    # Test edge extraction
    assert len(p.edgeStr) == 37
    assert p.edgeStr[14] == "<=1600.00"

    # Test edge conversion
    assert len(p.edges) == 37
    assert p.edges[14] == 1600.00

    # Test value estimation
    assert len(p.values) == 37
    assert p.values[14] == 1500.00

    # Test untouched
    assert np.isclose(p.untouchedTight, 40000.001)
    assert np.isclose(p.untouchedWide, 100000000040000.0)

    # Test nodata
    assert np.isclose(p.noData, -999999)

    # Test fail
    try:
        p = PriorSource(gl._test_data_["elevation.tif"])
        assert False
    except PriorSource._LoadFail:
        assert True


@pytest.mark.skip(reason="Todo")
def test_Prior_containsValue():
    print("Prior_containsValue is trivial")


@pytest.mark.skip(reason="Todo")
def test_Prior_valueOnEdge():
    print("Prior_valueOnEdge is trivial")


def test_Prior_generateRaster():
    p = PriorSource(priorSample)

    ext = gk.Extent.load(aachenShape)
    r = p.generateRaster(extent=ext, output=join(RESULTDIR, "generatedRaster1.tif"))
    mat = gk.raster.extractMatrix(r)
    assert np.isclose(269719875, mat.sum())
    assert np.isclose(1287.87195567, mat.std())

    ri = gk.raster.rasterInfo(r)
    assert np.isclose(ri.dx, 100)
    assert np.isclose(ri.dy, 100)
    assert np.isclose(ri.bounds, (4035500.0, 3048700.0, 4069500.0, 3101000.0)).all()
    assert ri.dtype == gdal.GDT_Float64


def test_Prior_generateVector():
    # generateVector(s, extent, value, output=None)

    p = PriorSource(gl._test_data_["roads_prior_clip.tif"])
    ext = gk.Extent.load(aachenShape)

    # Test an on-edge generation
    v = p.generateVector(
        ext, value=4000, output=join(RESULTDIR, "generatedVector1.shp")
    )
    g = gk.vector.extractFeature(v, onlyGeom=True)
    # Tests below are failing for 3.0.0<=gdal<3.4.0 due to problems when
    # polygonizing
    assert np.isclose(g.Area(), 1684940000.0)

    # Test an off-edge generation
    v = p.generateVector(
        ext, value=5500, output=join(RESULTDIR, "generatedVector2.shp")
    )
    g = gk.vector.extractFeature(v, onlyGeom=True)
    # Tests below are failing for 3.0.0<=gdal<3.4.0 due to problems when
    # polygonizing
    assert np.isclose(g.Area(), 1851537325.6536)


@pytest.mark.skip(reason="Todo")
def test_Prior_extractValues():
    print("Prior_extractValues is trivial")


def test_PriorSet___init__():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = PriorSet(DATADIR)

    keys = list(ps._sources.keys())
    assert len(keys) == 2, "Prior count"
    assert "roads_main_proximity" in keys, "Prior inclusion"
    assert "settlement_proximity" in keys, "Prior inclusion"


@pytest.mark.skip(reason="Todo")
def test_PriorSet_loadDirectory():
    print("PriorSet__loadDirectory is implicity tested")


@pytest.mark.skip(reason="Todo")
def test_PriorSet_regionIsOkay():
    print("PriorSet_regionIsOkay not tested")


@pytest.mark.skip(reason="Todo")
def test_PriorSet_sources():
    print("PriorSet_sources not tested")


@pytest.mark.skip(reason="Todo")
def test_PriorSet_listKeys():
    print("PriorSet_listKeys not tested")


@pytest.mark.skip(reason="Todo")
def test_PriorSet___getitem__():
    print("PriorSet___getitem__ not tested")


@pytest.mark.skip(reason="Todo")
def test_PriorSet_combinePriors():
    print("PriorSet_combinePriors not tested")


@pytest.mark.skip(reason="Todo")
def test_setPriorDirectory():
    print("setPriorDirectory not tested")
