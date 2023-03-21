# Dataset pipeline for H-21cm cube
For the production of epoch of reionisation (EoR) hydrogen 21cm (H-21cm) observational `.fits` cubes. This pipeline utilises a modified/forked [version](https://github.com/SimonP2207/tools21cm) of the python implementation of the [`21cmfast`](https://github.com/sambit-giri/tools21cm) library.
## Installing
### Conda Environment
In order to use this package, we have provided a conda environment configuration file (`environment.yaml`). You can use this to create your conda virtual environment with which to run the pipeline to produce the H-21cm cube. In order to set up the environment on the command-line:

```bash
conda env create --file environment.yaml
```

This will create an environment named 'sdc3_h21cm_generator' the `envs` directory within conda's base directory. In order to create the virtual environment in a custom path (e.g. `./venv`):

```bash
conda env create -p ./venv --file environment.yaml
```

You can also create a custom-named environment via:

```bash
conda env create -n my_name --file environment.yaml
```

### Singularity Image
Coming soon.

## Operation
### Parameter file
Using the pipeline requires the configuration of a [`.toml`](https://toml.io/en/) file containing all of the required parameters needed for an execution of the pipeline. The full path to this configuration file constitutes the sole command-line argument for the execution of the pipeline.

The parameter file is divided into the following sections:
- `outputs`: Directories and file naming
- `field`: Observational coordinate system
- `correlator`: Frequency setup
- `astro_params`: `21cmfast` EoR realisation parameters
- `user_params`: Various user and `21cmfast` configuration parameters
- `flags`: Flags to pass to `21cmfast`

An example parameter file is provided in this repository in `files/example_params.toml` with annotations in the form of comments for each individual parameter.

### Conda environment
After setting up the virtual environment, it can be activated via:

```bash
conda activate sdc3_h21cm_generator
``` 

Once within the virtual environment, the pipeline can be executed according to the parameters within the parameter file via:

```bash
python /path/to/sdc3_training_datasets/eor_h21cm.py parameter_file.toml
```

### Singularity image
Coming soon.

## Dependencies
Please see the file, `environment.yaml`, for a full list of dependencies.
