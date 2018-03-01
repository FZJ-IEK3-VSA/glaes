from distutils.core import setup

setup(
    name='glaes',
    version='1.0.0',
    author='Severin Ryberg',
    url='https://github.com/FZJ-IEK3-VSA/glaes',
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
