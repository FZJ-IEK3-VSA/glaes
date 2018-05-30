from setuptools import setup, find_packages

setup(
    name='glaes',
    version='1.1.1',
    author='Severin Ryberg',
    url='https://github.com/FZJ-IEK3-VSA/glaes',
    packages = find_packages(),
    include_package_data=True,
    install_requires = [
        "geokit>=1.1.1",
        "gdal>=2.0.0",
        "numpy>=1.13.2",
        "descartes>=1.1.0",
        "pandas>=0.22.0",
        "scipy>=1.0.0",
        "matplotlib>=2.1.1",
    ]
)
