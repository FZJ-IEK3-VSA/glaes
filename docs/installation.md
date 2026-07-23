# Installation

## Recommended Installation

The recommended way to install GLAES is to use the conda package manager. This will ensure that all dependencies are installed correctly and that the package is compatible with your system.

Using the conda package manager of your choice (we recommend [miniforge](https://github.com/conda-forge/miniforge)), you can install GLAES with the following command:

```bash
conda install -c conda-forge glaes
```

If you are installing GLAES into an environment using an `environment.yml` file, make sure to add the conda-forge channel to the file:

```yaml
channels:
  - conda-forge
dependencies:
  - conda-forge::glaes
```

!!! note
    `conda` and `mamba` can be used interchangeably in all commands below.

## Development Installation

GLAES is closely linked to [GeoKit](https://github.com/FZJ-IEK3-VSA/geokit). If you intend to develop GLAES, it is also recommended to install GeoKit in development mode into the same environment.

1. Clone a local copy of both repositories:

    ```bash
    git clone https://github.com/FZJ-IEK3-VSA/glaes.git
    git clone https://github.com/FZJ-IEK3-VSA/geokit.git
    ```

2. Copy all dependencies from both `requirements.yml` files into a new `requirements-combined.yml`.

3. Create the new environment with all conda-forge dependencies:

    ```bash
    conda env create --file requirements-combined.yml -n glaes_dev_env
    ```

4. Activate the environment:

    ```bash
    conda activate glaes_dev_env
    ```

5. Install both libraries in editable mode:

    ```bash
    pip install -e ./geokit --no-deps
    pip install -e ./glaes --no-deps
    ```
