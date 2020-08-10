from setuptools import setup, find_packages

setup(
    name='glaes',
    version='1.2.0',
    author='Severin Ryberg',
    url='https://github.com/FZJ-IEK3-VSA/glaes',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "geokit>=1.2.8",
        "gdal>2.0.0,<3.0.0",
        "numpy",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
    ]
)
