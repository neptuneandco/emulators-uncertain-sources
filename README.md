# Emulators-Uncertain-Sources
## Description
Emulators-Uncertain-Sources is a repository designed to facilitate training of neural networks on MODFLOW inputs and outputs for scenarios with changing source timing and location, namely U-FNO and Attention U-Net. It can be adapted for any image training purpose. This was based on the original U-FNOB source code by Meray et al. 2024, but it has been significantly adapted for Morphew et al. 2026. We acknowledge that the work to produce this repository and paper submission was funded by Neptune and Company's internal research and development budget.

## Models
- `U_FNO.py` contains U-FNO code modified from Meray et al. 2024's U-FNOB code. Modifications were mainly to fix broken references; the model's architecture remains mostly untouched.
- `AU_NET.py` contains an attention u-net model used for comparison to U-FNO. AU-Net was used to interrogate the following question: how well does the FNO perform to "traditional" CNNs?

## Data Preparation and Training
A small hdf5 dataset has been provided in this repo so that anyone can try out the code. Email Michael at mmorphew@neptuneinc.org for full data access.

- `data_utils.py` contains common functions used across all notebooks for preparing the dataloaders.
- `training_step_based.ipynb` trains data in a step-based way, ideal for larger datasets for which epochs would not be feasible.
- `hyperparameter_tuning.ipynb` helps find ideal hyperparameters for U_FNO specifically.

## Visualization and Analysis
Visualize and analyze the models and their performance by using the following Jupyter notebooks:

- `testing_performance_plots.ipynb` creates figures 4-12 in Morphew et al. 2026, all of which compare the two models.

## Environment installation and usage
Conda is recommended to run these scripts. An environment can be installed from `requirements.yml` with a command such as: `conda env create --file requirements.yml`

Activate the environment with `conda activate emulators` (This environment name can be changed at the top of `requirements.yml`).

The notebooks were designed for use on CUDA devices (NVIDIA 2080 Ti with 12G VRAM was used for the results shown in Morphew et al. 2026). There is some implmented support for Apple's Metal API for compatible Mac devices with Apple Silicon, but not all PyTorch functionality may be implemented, resulting in the CPU being used as a fallback.

If neither CUDA nor Metal is available, notebooks are instead run on CPU and training will likely be very slow.

As training may take a while, consider using papermill to automate the execution of the jupyter notebooks. 

Example command: `papermill training_step_based.ipynb training_step_based_output.ipynb`


## Contributing
We are open to contributions. Please feel free to submit issues, enhancements, or bug fixes through GitHub.
