# tropical_PODS
Fill in text 

## PODs
Contains the functions and modules necessary to compute the various diagnostics. The main diagnostics
included are:

### POD_name_01
Description

### POD_name_02
Description

## examples
Scripts containing example use cases. These scripts read in data, compute diagnostics and plot the results.
This is a good place to start when first using the diagnostics.

## installation
Update text in this section.

Download or clone the source code from github. Make sure all the required packages are available. If working
on a machine that requires python environments to install packages create a conda environment first:

MyEnv = your chosen name of the environment

`conda create --name MyEnv`

Install the required packages:

`conda install -n MyEnv scipy`

`conda install -n MyEnv xarray`

`conda install -n MyEnv numpy`

`conda install -n MyEnv netCDF4`

`conda install -n MyEnv pyngl`

Activate the conda environment:
`conda activate MyEnv`

Change directory to top level package directory tropical_diagnostics and install package using pip.

`cd tropical_diagnostics/`
`pip install./`

To run scripts from the examples directory copy the script into a directory where you would like to run. Make
sure to activate the conda environment with `conda activate MyEnv`. Adjust all path and filenames if necessary.
