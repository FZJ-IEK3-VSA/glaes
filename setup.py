from setuptools import setup, find_packages

setup(
    name='glaes',
    version='1.1.3',
    author='Severin Ryberg',
    url='https://github.com/FZJ-IEK3-VSA/glaes',
    packages = find_packages(),
    include_package_data=True,
    install_requires = [
        "geokit>=1.1.3",
        "gdal==2.4.1",
        "numpy",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
    ]
)
