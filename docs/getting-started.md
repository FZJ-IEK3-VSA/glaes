# Getting Started

## A Simple Land Eligibility Workflow

**Objective:** Determine land eligibility for photovoltaic (PV) modules in the
[Aachen administration region](https://en.wikipedia.org/wiki/Aachen_(district)) considering that:

1. PV modules should not cover agricultural areas (because people need to eat)
2. PV modules should not be within 200 meters of a major roadway (because they may get dirty)
3. PV modules should not be within 1000 meters of a settlement area (because they are too shiny)

```python
from glaes import ExclusionCalculator

ec = ExclusionCalculator(aachenRegion, srs=3035, pixelRes=100)
ec.excludePrior("agriculture_proximity", value=0)
ec.excludePrior("settlement_proximity", value=(None, 1000))
ec.excludePrior("roads_main_proximity", value=(None, 200))
ec.draw()
```

## European Priors

GLAES provides precomputed Prior datasets for the European context covering the most commonly considered criteria for land eligibility analyses. These can be downloaded from [Mendeley Data](https://data.mendeley.com/datasets/trvfb3nwt2) and placed in `glaes/data/priors`.

Using Priors is as simple as calling `excludePrior()` with the prior name and desired value threshold.

## Next Steps

For more detailed examples, see:

- [Basic Workflow](notebooks/00_basic_workflow.ipynb) - A complete walkthrough of the GLAES workflow
- [Placement Algorithm](notebooks/01_Placement_algorithm.ipynb) - Using GLAES for turbine/module placement
