from helpers import *
from glaes.core.priors import *


def test_Prior___init__():
    # Test success
    p = PriorSource("data/roads_prior_clip.tif")

    if not p.path == "data/roads_prior_clip.tif": 
        raise RuntimeError("Path mismatch")
    
    if not len(p.edgeStr) == 37 and p.edgeStr[14] =="<=1600.00": 
        raise RuntimeError("edge extraction")

    if not len(p.edges) == 37 and p.edges[14]==1600.00: 
        raise RuntimeError("edge conversion")

    if not len(p.values) == 37 and p.values[14] ==1500.00: 
        raise RuntimeError("value estimation")

    if not np.isclose(p.untouchedTight, 40000.001): 
        raise RuntimeError("untouchedTight")

    if not np.isclose(p.untouchedWide, 100000000040000.0): 
        raise RuntimeError("untouchedWide")

    if not np.isclose(p.noData, -999999):
        raise RuntimeError("noDataValue")

    # Test fail
    try:
        p = PriorSource("data/elevation.tif")
        raise RuntimeError("Prior fail")
    except PriorSource._LoadFail as e:
        pass

    print("Prior___init__ passed")

def test_Prior_containsValue():
    print("Prior_containsValue is trivial")

def test_Prior_valueOnEdge():
    print("Prior_valueOnEdge is trivial")

def test_Prior_generateRaster():
    p = PriorSource(gl._test_data_["roads_prior_clip.tif"])

    ext = gk.Extent.load(gk._test_data_["aachenShapefile.shp"])
    r = p.generateRaster( extent=ext, output="results/generatedRaster1.tif" )
    mat = gk.raster.extractMatrix(r)
    compareF(269719875, mat.sum()) 
    compareF(1287.87195567, mat.std())

    ri = gk.raster.rasterInfo(r)
    compareF(ri.dx, 100)
    compareF(ri.dy, 100)
    compare(ri.bounds, (4035500.0, 3048700.0, 4069500.0, 3101000.0))
    compare(ri.dtype, gdal.GDT_Float64)


    print("Prior_generateRaster passed")

def test_Prior_generateVector():
    #generateVector(s, extent, value, output=None)
    
    p = PriorSource(gl._test_data_["roads_prior_clip.tif"])
    ext = gk.Extent.load(gk._test_data_["aachenShapefile.shp"])

    # Test an on-edge generation
    v = p.generateVector(ext, value=4000, output="results/generatedVector1.shp")
    g = gk.vector.extractFeature(v, onlyGeom=True)

    compareF(g.Area(), 1684940000.0)


    # Test an off-edge generation
    v = p.generateVector(ext, value=5500, output="results/generatedVector2.shp")
    g = gk.vector.extractFeature(v, onlyGeom=True)

    compareF(g.Area(), 1851537325.6536)


    print("Prior_generateVectorFromEdge passed")

def test_Prior_extractValues():
    print("Prior_extractValues is trivial")

def test_PriorSet___init__():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = PriorSet("data")

    keys = list(ps._sources.keys())
    compareF(len(keys), 2, "Prior count")
    compareF("roads_main_proximity" in keys, True, "Prior inclusion")
    compareF("settlement_proximity" in keys, True, "Prior inclusion")

    print("PriorSet___init__ passed")

def test_PriorSet_loadDirectory():
    print("PriorSet__loadDirectory is implicity tested")

def test_PriorSet_regionIsOkay():
    print("PriorSet_regionIsOkay not tested")

def test_PriorSet_sources():
    print("PriorSet_sources not tested")

def test_PriorSet_listKeys():
    print("PriorSet_listKeys not tested")

def test_PriorSet___getitem__():
    print("PriorSet___getitem__ not tested")

def test_PriorSet_combinePriors():
    print("PriorSet_combinePriors not tested")

def test_setPriorDirectory():
    print("setPriorDirectory not tested")

if __name__ == "__main__":
    test_Prior___init__()
    test_Prior_containsValue()
    test_Prior_valueOnEdge()
    test_Prior_generateRaster()
    test_Prior_generateVector()
    test_Prior_extractValues()
    test_PriorSet___init__()
    test_PriorSet_loadDirectory()
    test_PriorSet_regionIsOkay()
    test_PriorSet_combinePriors()
