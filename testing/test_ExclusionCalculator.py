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

	if not (isclose(ec.availability.mean(), 0.32497469, 1e-6) and isclose(ec.availability.std(), 0.46836540, 1e-6)):
		raise RuntimeError("excludeRasterType: Single value exclusion - FAIL")

	# exclude value range
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludeRasterType(clcRaster, (5,12))

	if not (isclose(ec.availability.mean(), 0.31626365, 1e-6) and isclose(ec.availability.std(), 0.46501717, 1e-6)):
		raise RuntimeError("excludeRasterType: Value range exclusion - FAIL")
	
	# exclude value maximum
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludeRasterType(clcRaster, (None,12))

	if not (isclose(ec.availability.mean(), 0.23848273, 1e-6) and isclose(ec.availability.std(), 0.42615572, 1e-6)):
		raise RuntimeError("excludeRasterType: Value maximum exclusion - FAIL")
	
	ecMax12 = gl.ExclusionCalculator(aachenShape)
	ecMax12.excludeRasterType(clcRaster, valueMax=12)

	if not np.abs(ecMax12.availability-ec.availability).sum()/ec.availability.size < 0.000001:
		raise RuntimeError("excludeRasterType: valueMax argument - FAIL")

	# exclude value minimum
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludeRasterType(clcRaster, (13,None))

	if not (isclose(ec.availability.mean(), 0.16048251, 1e-6) and isclose(ec.availability.std(), 0.36705297, 1e-6)):
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

	if not (isclose(ec.availability.mean(), 0.30511191, 1e-6) and isclose(ec.availability.std(), 0.46045485, 1e-6)):
		raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

	# exclude all features directly, new srs
	ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelSize=0.005)
	ec.excludeVectorType(cddaVector)

	if not (isclose(ec.availability.mean(), 0.32915172, 1e-6) and isclose(ec.availability.std(), 0.46990514, 1e-6)):
		raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

	# exclude a selection of features
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludeVectorType(cddaVector, where="YEAR>2000")

	if not (isclose(ec.availability.mean(), 0.346693277, 1e-6) and isclose(ec.availability.std(), 0.475917101, 1e-6)):
		raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

	# exclude a selection of features with buffer
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludeVectorType(cddaVector, where="YEAR>2000", buffer=400)

	if not (isclose(ec.availability.mean(), 0.310994267, 1e-6) and isclose(ec.availability.std(), 0.462900430, 1e-6)):
		raise RuntimeError("excludeVectorType: Single value exclusion, new srs - FAIL")

def test_excludePrior():
	# make a prior source 
	pr = gl.core.priors.PriorSource(priorSample)

	# test same srs
	ec = gl.ExclusionCalculator(aachenShape)
	ec.excludePrior(pr, valueMin=400)

	if not (isclose(ec.availability.mean(), 0.099113658, 1e-6) and isclose(ec.availability.std(), 0.290466696, 1e-6)):
		raise RuntimeError("excludePrior: Same srs - FAIL")

	# test different srs and resolution
	ec = gl.ExclusionCalculator(aachenShape, srs='latlon', pixelSize=0.001)
	ec.excludePrior(pr, valueMin=400)

	if not (isclose(ec.availability.mean(), 0.110114217, 1e-6) and isclose(ec.availability.std(), 0.304726034, 1e-6)):
		raise RuntimeError("excludePrior: Different srs - FAIL")

def test_multiple_exclusions():
	pr = gl.core.priors.PriorSource(priorSample)
	ec = gl.ExclusionCalculator(aachenShape)

	# apply exclusions
	ec.excludePrior(pr, valueMax=400)	
	ec.excludeVectorType(cddaVector, where="YEAR>2000")
	ec.excludeRasterType(clcRaster, valueMax=12)

	if not (isclose(ec.availability.mean(), 0.151972011, 1e-6) and isclose(ec.availability.std(), 0.356195122, 1e-6)):
		raise RuntimeError("Multiple Exclusions - FAIL")

	def test_distributeItems():
		print("MAKE TESTS FOR distributeItems!!!")

	def test_distributeAreas():
		print("MAKE TESTS FOR distributeAreas!!!")

if __name__ == "__main__":
	test_ExclusionCalculator()
	test_excludeRasterType()
	test_excludeVectorType()
	test_excludePrior()
	test_multiple_exclusions()
	test_distributeItems()
	test_distributeAreas()

