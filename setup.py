from setuptools import setup, find_packages

setup(
    name="glaes",
    version="1.3.0",
    author="GLAES Developer Team",
    url="https://github.com/FZJ-IEK3-VSA/glaes",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geokit>=1.4.0",
        "gdal==3.4.*",
        "numpy",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
    ],
)
