from helpers import *

def test_ExclusionCalculatorInitialize():
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
	# Test with the default definitions
	ec = gl.ExclusionCalculator(aachenShape)

if __name__ == "__main__":
	test_ExclusionCalculatorInitialize()

