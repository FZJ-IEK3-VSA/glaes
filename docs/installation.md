# Installation

## Recommended Installation

The recommended way to install GLAES is to use the conda package manager. This will ensure that all dependencies are installed correctly and that the package is compatible with your system.

Using the conda package manager of your choice (we recommend [mambaforge](https://github.com/conda-forge/miniforge)), you can install GLAES with the following command:

```bash
conda install -c conda-forge glaes
```

If you are installing GLAES into an environment using an `environment.yml` file, make sure to add the conda-forge channel:

```yaml
channels:
  - conda-forge
  - defaults
dependencies:
  - conda-forge::glaes
```

!!! warning
    We **highly recommend** installing GLAES into a new, empty environment, as the dependencies of GLAES may conflict with other packages you have installed.

## Alternative Installation

The primary dependencies of GLAES are:

1. `gdal>2.0.0,<3.0.0`
2. [GeoKit](https://github.com/FZJ-IEK3-VSA/geokit) >= 1.2.4

If you can install these modules on your own, then the GLAES module should be easily installable with:

```bash
pip install git+https://github.com/FZJ-IEK3-VSA/glaes.git#egg=glaes
```

## Development Installation

1. Clone a local copy of the repository:

    ```bash
    git clone https://github.com/FZJ-IEK3-VSA/glaes.git
    cd glaes
    ```

2. Create a development environment:

    ```bash
    conda env create --file requirements-dev.yml
    conda activate glaes
    ```

3. Install GLAES in editable mode:

    ```bash
    pip install -e . --no-deps
    ```
