from distutils.core import setup

setup(
    name='glaes',
    version='1.0.0',
    author='Severin Ryberg',
    url='http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html',
    packages = ["glaes"],
    install_requires = [
        "gdal>=2.1.0",
        "numpy>=1.11.2",
        "geokit>=1.0.1",
        "pandas",
        "scipy",
        "descartes"
    ]
)
