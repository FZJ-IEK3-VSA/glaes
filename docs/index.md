<p align="left">
  <a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa" height="60px"></a>
</p>


# Geospatial Land Availability for Energy Systems (GLAES)

GLAES is a framework for conducting land eligibility analyses and is designed to easily incorporate disparate geospatial information from a variety of sources into a unified solution.
Currently, the main purpose of GLAES is performing land eligibility (LE) analyses which, in short, are used to determine which areas within a region are deemed 'eligible' for some purpose (such as placing a wind turbine).

Although initially intended to operate in the context of distributed renewable energy systems, such as onshore wind and open-field solar parks, the work flow of GLAES is applicable to any context where a constrained indication of land is desired.

To use GLAES, first [install GLAES](installation.md) and then [get started](getting-started.md).

GLAES is open-source available on [GitHub](https://github.com/FZJ-IEK3-VSA/glaes) and open for collaboration, help requests, etc.
In case you use GLAES in a scientific publication, we kindly request you to cite our publications listed in the [Further Reading](further-reading.md) section.

## Features

- Standardized approach to land eligibility analyses
- Applicable in any geographic region and at any resolution
- Can flexibly incorporate most geospatial datasets: including the common .shp and .tif formats
- Simple visualization and storage of results as common image or raster dataset
- Simple integration of results into other analysis (via numpy array)

## European Priors

A number of precomputed (Prior) datasets which constitute the most commonly considered criteria used for LE analyses have been constructed for the European context.
These datasets are formatted to be used directly with the GLAES framework and, in doing so, drastically reduce the time requirements, data management, and overall complexity of conducting these analyses.

The Prior datasets can be downloaded from [Mendeley Data](https://data.mendeley.com/datasets/trvfb3nwt2) and installed by unzipping the files in the repo directory `glaes/data/priors`.
