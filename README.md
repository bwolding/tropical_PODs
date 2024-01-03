# tropical_PODS
## Installation

1.) Clone the "tropical_PODS" repo from Github

2.) Install environment using attached YAML file (tropical_diagnostics_environment_11_30_2023.yml)

3.) Download data from NCAR GLADE (/glade/scratch/bwolding/GitHub_Tropical_PODs_Data) or email brandon.wolding@noaa.gov for data

4.) Move all data from the "GitHub_Tropical_PODs_Data" folder to your local repo folder  "/tropical_PODS/data/"

5.) Run a simple test case to make sure everything is working, before moving on to the more complicated plume model. To start, open and run "/tropical_PODS/examples/jupyter_notebook_examples/CSF_precipitation_diagnostic_ERA5_example.ipynb"

6.) If everything is running correctly to this point, then it is time to try to tackle the plume model, which requires additional installation.

7.) Navigate to "/tropical_PODs/plume_model_master_bwolding_mod_06_22_22/" and delete the "thermo_functions_bwolding.cpython-311-darwin.so" file. You will need to create a ".so" file specific to your own architecture. 

8.) In order to build your architecture specific .so file, navigate to "/tropical_PODs/plume_model_master_bwolding_mod_06_22_22/" and then run "python setup.py build_ext --inplace" which should result in a ".so" file in that directory.

9.) Once you see a ".so" file in the "/tropical_PODs/plume_model_master_bwolding_mod_06_22_22/" directory, you should be ready to run the plume model. Navigate to "/tropical_PODS/examples/jupyter_notebook_examples/" and run "plume_model_DYNAMO_NSA_example.ipynb".


## PODs
Contains the functions and modules necessary to compute various diagnostics. The main diagnostics
included are:

### Entraining plume buoyancy
As documented in [Wolding et al. 2022](https://journals.ametsoc.org/view/journals/atsc/81/1/JAS-D-23-0061.1.xml), and citations therewithin. Idealized model for computing plume thermodynamic properties upon lifting and mixing. The notebook "plume_model_DYNAMO_NSA_example.ipynb" provides an example workflow.

The version of the plume model here has been slightly modified from previous versions (e.g. [Ahmed and Neelin 2021](https://journals.ametsoc.org/view/journals/clim/34/10/JCLI-D-20-0384.1.xml)), primarily in how the mixing coefficient is used to update plume properties.  

The most relevant starting points for understanding the plume model are:

1.) The "numerical_plume_model" function in "/tropical_PODS/PODs/POD_utils.py" shows how the mixing coefficient is defined, and has several comments that I have added. 

2.) To really understand the  "numerical_plume_model" function and the indexing conventions, look at how c_mix is defined and used in the "plume_lifting" function of "thermo_functions.pyx" in lines 485 - 495.

### Temporal Co-Evolution of Column Saturation Fraction (CSF) and Precipitation
As documented in [Wolding et al. 2020](https://journals.ametsoc.org/view/journals/atsc/77/5/jas-d-19-0225.1.xml). The notebook "CSF_precipitation_diagnostic_ERA5_example.ipynb" provides an example workflow.

This diagnostic also contains methods for compositing and plotting additional variables in CSF-P space.

### Temporal Co-Evolution of Lower Tropospheric Vertically Integrated Buoyancy (VIB) and Precipitation
As documented in [Wolding et al. 2022](https://journals.ametsoc.org/view/journals/atsc/81/1/JAS-D-23-0061.1.xml). The notebook "B_DIB_precipitation_diagnostic_ERA5_example.ipynb" provides an example workflow.

This diagnostic also contains methods for compositing and plotting additional variables in VIB-P space.


