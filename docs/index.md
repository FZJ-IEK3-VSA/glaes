<p align="left">
  <a href="https://www.fz-juelich.de/en/ice/ice-2">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg#only-light" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header-dark.svg#only-dark" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa">
  </a>
</p>

# ETHOS.GLAES - Geospatial Land Availability for Energy Systems

**Land eligibility analysis for energy infrastructure — which areas qualify, and why.**

ETHOS.GLAES is a framework for conducting land eligibility analyses and is designed to easily incorporate disparate geospatial information from a variety of sources into a unified solution.
Currently, the main purpose of ETHOS.GLAES is performing land eligibility (LE) analyses which, in short, are used to determine which areas within a region are deemed 'eligible' for some purpose (such as placing a wind turbine).

Although initially intended to operate in the context of distributed renewable energy systems, such as onshore wind and open-field solar parks, the workflow of ETHOS.GLAES is applicable to any context where a constrained indication of land is desired.

To use ETHOS.GLAES, first [install it](installation.md) and then [get started](getting-started.md).

ETHOS.GLAES is open-source available on [GitHub](https://github.com/FZJ-IEK3-VSA/glaes) and open for collaboration, help requests, etc.
In case you use ETHOS.GLAES in a scientific publication, we kindly request you to cite our publications listed in the [Further Reading](further-reading.md) section.

ETHOS.GLAES is part of the [Energy Transformation PatHway Optimization Suite (ETHOS) at ICE-2](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services). It is built on top of [ETHOS.GeoKit](https://github.com/FZJ-IEK3-VSA/geokit), which handles the underlying raster and vector operations.

## Features

- Standardized approach to land eligibility analyses
- Applicable in any geographic region and at any resolution
- Can flexibly incorporate most geospatial datasets: including the common .shp and .tif formats
- Simple visualization and storage of results as common image or raster dataset
- Simple integration of results into other analyses (via numpy array)

## European Priors

A number of precomputed (Prior) datasets which constitute the most commonly considered criteria used for LE analyses have been constructed for the European context.
These datasets are formatted to be used directly with the ETHOS.GLAES framework and, in doing so, drastically reduce the time requirements, data management, and overall complexity of conducting these analyses.

The Prior datasets can be downloaded from [Mendeley Data](https://data.mendeley.com/datasets/trvfb3nwt2) and installed by unzipping the files in the repo directory `glaes/data/priors`.
