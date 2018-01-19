import os as _os

testdata_aachen = _os.path.join(_os.path.dirname(__file__), "..",  "..", "testing", "data", "aachenShapefile.shp")

class GlaesError(Exception): pass