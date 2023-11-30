import numpy as np
import xarray as xr
from tropical_PODs.plume_model_master_bwolding_mod_06_22_22.thermodynamic_functions import *
from tropical_PODs.plume_model_master_bwolding_mod_06_22_22.thermo_functions_bwolding import plume_lifting_bwolding

def numerical_plume_model(temperature, specific_humidity, launch_level_hPa):
    
    ###########################################################
    # Takes a time x level variable, where time has length >= 2
    # Launch level must be very near 1000 hPa, as 1000 hPa is used to determine mass flux profile
    
    # To really understand this function and the indexing conventions, look at how c_mix is defined and used in the plume_lifting
    # function of thermo_functions.pyx in lines 485 - 495
        
    ############################################
    ####  Run Numerical Plume Calculations  ####
    ############################################
    
    ### Transpose data so it is time x level ##
    
    temperature = temperature.transpose('time','lev')  # Added by Brandon Wolding 06/22/2022
    specific_humidity = specific_humidity.transpose('time','lev') # Added by Brandon Wolding 06/22/2022
    
    ### Sort levels in descending order ##
    
    temperature = temperature.sortby('lev', ascending=False) # Added by Brandon Wolding 06/22/2022
    specific_humidity = specific_humidity.sortby('lev', ascending=False) # Added by Brandon Wolding 06/22/2022
    
    ###   Define inputs   ###

    #temp = temperature.reindex(lev=temperature.lev[::-1]) # Commented out by Brandon Wolding 06/22/2022, removed because I added the sortby() code above
    #sphum = specific_humidity.reindex(lev=specific_humidity.lev[::-1]) # Commented out by Brandon Wolding 06/22/2022, removed because I added the sortby() code above
    
    temp = temperature
    sphum = specific_humidity
    pressure = temperature['lev']
    
    #### Set mixing preference ####
    # Deep-inflow B mixing like in Holloway & Neelin 2009, JAS; Schiro et al. 2016, JAS
    #MIX='DIB'       

    # No mixing case
    # MIX='NOMIX'   

    #print('READING AND PREPARING FILES')

    ### Read pressure levels ###
    lev=np.int_(pressure) ## convert from short to int

    ### Remove some levels with NaNs
    # THIS SECTION COMMENTED OUT BY BRANDON WOLDING 06/25/2021 TO REMOVE TREATMENT OF NANS. SECTION ADDED BELOW
    
    #ind_fin=np.where(np.isfinite(temp[0,:]))[0] # Commented out by Brandon Wolding 06/25/2021
    #temp_fin=temp[:,ind_fin]
    #sphum_fin=sphum[:,ind_fin]
    #sphum_saturated_fin = sphum_saturated[:,ind_fin]
    #lev_fin=lev[ind_fin]
    
    # THIS SECTION ADDED BY BRANDON WOLDING 06/25/2021 TO REPLACE ABOVE SECTION TREATING NANS
    temp_fin=temp
    sphum_fin=sphum
    lev_fin=lev

    ###   Save dims and coordinates for making xArray variables later   ###

    temp_fin_dims = temp_fin.dims
    temp_fin_coords = temp_fin.coords

    ### Get time dimension ###
    time_dim=np.arange(temp.shape[0])

    ### Obtain indices for mid and starting pressure levels
    i450=np.where(lev_fin==450)[0]
    i1000=np.where(lev_fin==launch_level_hPa)[0]
    i850=np.where(lev_fin==850)[0] # Added by Brandon Wolding 06/23/2022. Going to use this index to create c_mic_DIBDBL profile
    ####
    
    ## Launch plume from 1000 hPa ##
    ind_launch=np.zeros((time_dim.size),dtype='int')
    ind_launch[:]=i1000 # Modified by Brandon Wolding 06/22/2021
        
    ### Prescribing mixing coefficients ###
    c_mix_DIB = np.zeros((time_dim.size,lev_fin.size)) 
    c_mix_NOMIX = np.zeros((time_dim.size,lev_fin.size))
#     c_mix_DIBDBL = np.zeros((time_dim.size,lev_fin.size)) # Added by Brandon Wolding 06/22/2022
     
    ## Compute Deep Inflow Mass-Flux ##
    ## The mass-flux (= vertical velocity) is a sine wave from near surface
    ## to 450 mb. Designed to be 1 at the level of launch
    ## and 0 above max. w i.e., no mixing in the upper trop.

    #assert(all(ind_launch>=1)) # Commented out by Brandon Wolding 06/22/2021
    w_mean = np.sin(np.pi*0.5*(lev_fin[ind_launch][:,None]-lev_fin[None,:])/(lev_fin[ind_launch][:,None]-lev_fin[i450])) # Modified by Brandon Wolding 06/22/2021 
    minomind = np.where(w_mean==np.nanmax(w_mean))[0][0]
    c_mix_DIB[:,1:-1]= (w_mean[:,2:] - w_mean[:,:-2])/(w_mean[:,2:]+w_mean[:,:-2]) # This section is meant to match plume_lifting_bwolding, a function within
    # thermo_functions_bwolding.pyx, which was modified from original code by B. Wolding on 06/22/2022 in order to make local mixing instantatneous.
    # In the original version, the plume properties at a given level were determined by mixing occuring at previous level, not mixing  occuring at current level.
    # Essentially mixing occured from previous level, plume thermodynamics were diagnosed at current level, and then mixing at current level occured.
    # In other words, plume properties at the current level were not impacted by mixing at the current level.
    # This will bias CAPE high relative to if mixing occured locally (at current level) and instantaneuosly and then plume properties were diagnosed.
    # In this version, plume mixing happens locally (at current level) and instantaenously.
    # Here the definition of c_mix is such that c_mix(n) indicates the mixing of environmental air at level "n" into the plume. 

    c_mix_DIB[c_mix_DIB<0]=0.0
    c_mix_DIB[np.isinf(c_mix_DIB)]=0.0
    
    ## Compute Deep Inflow Mass-Flux Just in the DBL (up to 850 hPa) ##
    ## Added by Brandon Wolding on 06/23/2022
    
#     c_mix_DIBDBL[:,1:-1]= (w_mean[:,2:] - w_mean[:,:-2])/(w_mean[:,2:]+w_mean[:,:-2])
#     c_mix_DIBDBL[:,(int(i850)+1):]=0 # Set c_mix at all levels HIGHER THAN 850 hPa equal to 0. The +1 increment is very important to avoid resolution sensitivity. We want all resolutions to stop mixing at exactly 850 hPa.
    
    ### Change data type ####
    temp_fin=np.float_(temp_fin)
    sphum_fin=np.float_(sphum_fin)
    
    ### Set output variables ####
    temp_plume_DIB=np.zeros_like(temp_fin)    
    Tv_plume_DIB=np.zeros_like(temp_fin)    

    temp_plume_NOMIX=np.zeros_like(temp_fin)    
    Tv_plume_NOMIX=np.zeros_like(temp_fin)    
    
#     temp_plume_DIBDBL=np.zeros_like(temp_fin) # Added by Brandon Wolding on 06/23/2022
#     Tv_plume_DIBDBL=np.zeros_like(temp_fin) # Added by Brandon Wolding on 06/23/2022 
   
    ## Launch plume ###
    #print('DOING DIB PLUME COMPUTATION')
    #print(np.shape(temp_fin))
    #print(np.shape(sphum_fin))
    #print(np.shape(Tv_plume_DIB))
    #print(np.shape(temp_plume_DIB))
    #print(np.shape(c_mix_DIB))
    #print(np.shape(lev_fin))
    #print(np.shape(ind_launch))
    
    plume_lifting_bwolding(temp_fin, sphum_fin, Tv_plume_DIB, temp_plume_DIB, c_mix_DIB, lev_fin, ind_launch)

    #print('DOING NOMIX PLUME COMPUTATION')
    plume_lifting_bwolding(temp_fin, sphum_fin, Tv_plume_NOMIX, temp_plume_NOMIX, c_mix_NOMIX, lev_fin, ind_launch)
    
    #print('DOING DIBDBL PLUME COMPUTATION')
#     plume_lifting_bwolding(temp_fin, sphum_fin, Tv_plume_DIBDBL, temp_plume_DIBDBL, c_mix_DIBDBL, lev_fin, ind_launch) # Added by Brandon Wolding on 06/23/2022
        
    ##############################################
    ###   Buoyancy and Dilution Calculations   ###
    ##############################################

    ## env. virtual temp. ###
    Tv_env = temp_v_calc(temp_fin, sphum_fin, 0.) ### Environmental virtual temp.
    
    ### thermal buoyancy ####
    #buoy_DIB = 9.8 * (Tv_plume_DIB-Tv_env)/(Tv_env)
    #buoy_NOMIX = 9.8 * (Tv_plume_NOMIX-Tv_env)/(Tv_env)
    
    ## Turn Numpy Arrays into XArray Variables ####
    
    Tv_env = xr.DataArray(Tv_env,dims=temp_fin_dims,coords=temp_fin_coords,name='Tv_env')
    Tv_env.attrs['units'] = '[K]'
    
    Tv_plume_DIB = xr.DataArray(Tv_plume_DIB,dims=temp_fin_dims,coords=temp_fin_coords,name='Tv_plume_DIB')
    Tv_plume_DIB.attrs['units'] = '[K]'
    
    Tv_plume_NOMIX = xr.DataArray(Tv_plume_NOMIX,dims=temp_fin_dims,coords=temp_fin_coords,name='Tv_plume_NOMIX')
    Tv_plume_NOMIX.attrs['units'] = '[K]'

    c_mix_DIB = xr.DataArray(c_mix_DIB,dims=temp_fin_dims,coords=temp_fin_coords,name='c_mix_DIB')
    c_mix_DIB.attrs['units'] = '[K]'
    
#     Tv_plume_DIBDBL = xr.DataArray(Tv_plume_DIBDBL,dims=temp_fin_dims,coords=temp_fin_coords,name='Tv_plume_DIBDBL')
#     Tv_plume_DIBDBL.attrs['units'] = '[K]'

    ## Drop time dimension from c_mix_DIB

    c_mix_DIB = c_mix_DIB.isel(time=0).drop('time')
    
    ## Sort Level into Ascending Order ##
    Tv_env = Tv_env.sortby('lev','ascending')
    Tv_plume_DIB = Tv_plume_DIB.sortby('lev','ascending')
    Tv_plume_NOMIX = Tv_plume_NOMIX.sortby('lev','ascending')
    c_mix_DIB = c_mix_DIB.sortby('lev','ascending')
#     Tv_plume_DIBDBL = Tv_plume_DIBDBL.sortby('lev','ascending')
    
    return Tv_env, Tv_plume_DIB, Tv_plume_NOMIX, c_mix_DIB #, Tv_plume_DIBDBL

def calculate_CAPE(Tv_env, Tv_plume, pressure_model_level_midpoint_Pa, pressure_model_level_interface_Pa, max_pressure_integral_array_Pa, min_pressure_integral_array_Pa):
    
    # Accepts both integers and arrays as min and max pressure limits
     
    # Define constants
    
    R_d = 287 # [J Kg^-1 K^-1]
    
    # Define virtual temperature difference between plume and environment
    
    delta_Tv_plume_env = Tv_plume - Tv_env
    
    # Set all model interfaces less than minimum pressure equal to minimum pressure, and more than maximum pressure to maximum pressure. This way, when you calculate "dp", these layers will not have mass.
    
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(pressure_model_level_interface_Pa < max_pressure_integral_array_Pa, other = max_pressure_integral_array_Pa)
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(pressure_model_level_interface_Pa > min_pressure_integral_array_Pa, other = min_pressure_integral_array_Pa)

    # Calculate delta natural log pressure for each model level
    
    d_ln_p = pressure_model_level_midpoint_Pa.copy()
    d_ln_p.values = xr.DataArray(xr.apply_ufunc(np.log, pressure_model_level_interface_Pa.isel(ilev = slice(1, len(pressure_model_level_interface_Pa.ilev))).values) - xr.apply_ufunc(np.log, pressure_model_level_interface_Pa.isel(ilev = slice(0, -1)).values)) # Slice indexing is (inclusive start, exclusive stop)
    
    # Set dp = nan at levels missing data so mass of those levels not included in calculation of dp_total
    
    d_ln_p = d_ln_p.where(~xr.apply_ufunc(np.isnan, delta_Tv_plume_env), drop=False, other=np.nan)
    
    # Mass weight each layer
    
    CAPE = R_d * (Tv_plume - Tv_env) * d_ln_p
    
    # Integrate over levels
    
    CAPE = CAPE.sum('lev', min_count=1)
    d_ln_p_total = d_ln_p.sum('lev', min_count=1)
    
    # Set ci_variable to nan wherever d_ln_p_total is zero or nan
    
    CAPE = CAPE.where(~(d_ln_p_total==0), drop = False, other=np.nan)
    CAPE = CAPE.where(~xr.apply_ufunc(np.isnan, d_ln_p_total), drop = False, other=np.nan)
    
    return CAPE

### Limit list of files to select years

def limit_files_to_select_years(list_of_files, list_of_desired_years):
    
    year_limited_files = []
    
    for year in list_of_desired_years:
                
        # Define year strings #
        
        current_year_string = str(year)
                
        while len(current_year_string) < 4:
            
            current_year_string = '0' + current_year_string
        
        for string in list_of_files:
            
            if (current_year_string in string):

                year_limited_files += [string]
        
    return year_limited_files

### Calculate saturation specific humidity

def calculate_saturation_specific_humidity(pressure, temperature):
    
    # Calculate Relative Humidity Based on Outdated WMO 1987 adapatation of Goff and Gratch Saturation Vapor Pressure (SVP)
    # Units of pressure are [Pa], temperature are [K]
    # Testing against MetPy function gives agreement within ~1% for CSF calculations.
    
    T_0 = 273.16
    
    log10SVP = 10.79574 * (1 - T_0 / temperature) - 5.028 * xr.apply_ufunc(np.log10,(temperature / T_0)) + 1.50475 * (10 ** -4) * (1 - 10 ** (-8.2969 * (temperature / (T_0 - 1)))) \
               + 0.42873 * (10 ** -3) * (10 ** (4.76955 * ((1 - T_0) / temperature))) + 0.78614 + 2.0
        
    SVP = 10 ** (log10SVP)
    
    eta = 0.62198 # Ratios of molecular weights of water and dry air
    
    saturation_specific_humidity = eta * SVP / pressure
    
    return saturation_specific_humidity

### Column integrate a variable that has nan values

def mass_weighted_vertical_integral_w_nan(variable_to_integrate, pressure_model_level_midpoint_Pa, pressure_model_level_interface_Pa, max_pressure_integral_array_Pa, min_pressure_integral_array_Pa):
    
    # Accepts both integers and arrays as min and max pressure limits
    
    # Define constants
    
    g = 9.8 # [m s^-2]
    
    # Set all model interfaces less than minimum pressure equal to minimum pressure, and more than maximum pressure to maximum pressure. This way, when you calculate "dp", these layers will not have mass.
    
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(pressure_model_level_interface_Pa < max_pressure_integral_array_Pa, other = max_pressure_integral_array_Pa)
    pressure_model_level_interface_Pa = pressure_model_level_interface_Pa.where(pressure_model_level_interface_Pa > min_pressure_integral_array_Pa, other = min_pressure_integral_array_Pa)

    # Calculate delta pressure for each model level
    
    dp = pressure_model_level_midpoint_Pa.copy()
    dp.values = xr.DataArray(pressure_model_level_interface_Pa.isel(ilev = slice(1, len(pressure_model_level_interface_Pa.ilev))).values - pressure_model_level_interface_Pa.isel(ilev = slice(0, -1)).values) # Slice indexing is (inclusive start, exclusive stop)
    
    # Set dp = nan at levels missing data so mass of those levels not included in calculation of dp_total
    
    dp = dp.where(~xr.apply_ufunc(np.isnan,variable_to_integrate), drop=False, other=np.nan)

    # Mass weight each layer
    
    ci_variable = variable_to_integrate * dp / g
    
    # Integrate over levels
    
    ci_variable = ci_variable.sum('lev', min_count=1)
    dp_total = dp.sum('lev', min_count=1)
    
    # Set ci_variable to nan wherever dp_total is zero or nan
    
    ci_variable = ci_variable.where(~(dp_total==0), drop = False, other=np.nan)
    ci_variable = ci_variable.where(~xr.apply_ufunc(np.isnan,dp_total), drop = False, other=np.nan)
    
    # Calculate mass weighted vertical average over layer integrated over
    
    mwa_variable = ci_variable * g / dp_total
    
    return ci_variable, dp_total, mwa_variable

### Calculate backwards, forwards and center differences of a variable

def calculate_backward_forward_center_difference(variable_to_difference):
    
    # Leading (backwards differenced)
    backwards_differenced_variable = variable_to_difference.isel(time = slice(1, len(variable_to_difference.time) + 1)).copy() # Careful to assign backwards differenced data to correct time step
    backwards_differenced_variable.values = variable_to_difference.isel(time = slice(1, len(variable_to_difference.time))).values - variable_to_difference.isel(time = slice(0, -1)).values # Slice indexing is (inclusive start, exclusive stop)

    # Lagging (forwards differenced)
    forwards_differenced_variable = variable_to_difference.isel(time = slice(0, -1)).copy() # Careful to assign forwards differenced data to correct time step
    forwards_differenced_variable.values = variable_to_difference.isel(time = slice(1, len(variable_to_difference.time))).values - variable_to_difference.isel(time = slice(0, -1)).values

    # Centered (center differenced)
    center_differenced_variable = variable_to_difference.isel(time = slice(1, -1)).copy() # Careful to assign center differenced data to correct time step
    center_differenced_variable.values = variable_to_difference.isel(time = slice(2, len(variable_to_difference.time))).values - variable_to_difference.isel(time = slice(0, -2)).values

    return backwards_differenced_variable, forwards_differenced_variable, center_differenced_variable

### Bin by one variable (Updated 09/27/2022 to handle NaN values in "variables_to_be_binned")

def bin_by_one_variable(variable_to_be_binned, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector):
    
    # BV1 is the variable to be used to create the bins
    
    # Define bins

    BV1_bin_midpoint = (lower_BV1_bin_limit_vector + upper_BV1_bin_limit_vector) / 2

    lower_BV1_bin_limit_DA = xr.DataArray(lower_BV1_bin_limit_vector, coords=[BV1_bin_midpoint], dims=['BV1_bin_midpoint'])

    upper_BV1_bin_limit_DA = xr.DataArray(upper_BV1_bin_limit_vector, coords=[BV1_bin_midpoint], dims=['BV1_bin_midpoint'])

    number_of_BV1_bins = len(BV1_bin_midpoint)

    # Instantiate composite variable

    coords = {'BV1_bin_midpoint': BV1_bin_midpoint}

    dims = ['BV1_bin_midpoint']

    bin_number_of_samples = xr.DataArray(np.full(len(lower_BV1_bin_limit_DA), np.nan), dims=dims, coords=coords)

    bin_mean_variable = bin_number_of_samples.copy()
    bin_number_pos_variable = bin_number_of_samples.copy()
    
    # Calculate bin mean and number of positive values in each bin

    for BV1_bin in BV1_bin_midpoint:
        
        bin_index = (BV1 >= lower_BV1_bin_limit_DA.where(lower_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin, drop=True).values) & \
                    (BV1 <= upper_BV1_bin_limit_DA.where(upper_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin, drop=True).values)
        
        bin_index_variable_isfinite = xr.apply_ufunc(np.isfinite, variable_to_be_binned.where(bin_index)) # Test that there is valid data where the bin index is
        
        if bin_index_variable_isfinite.sum() > 0:
                
                bin_mean_variable.loc[dict(BV1_bin_midpoint = BV1_bin)] = variable_to_be_binned.where(bin_index_variable_isfinite).mean()
                bin_number_of_samples.loc[dict(BV1_bin_midpoint = BV1_bin)] = bin_index_variable_isfinite.sum()
                bin_number_pos_variable.loc[dict(BV1_bin_midpoint = BV1_bin)] = (variable_to_be_binned.where(bin_index_variable_isfinite) > 0).sum()
            
        else:
                
                bin_mean_variable.loc[dict(BV1_bin_midpoint = BV1_bin)] = np.nan # There is data to determine bins, but the variable to be binned has NaN values
                bin_number_of_samples.loc[dict(BV1_bin_midpoint = BV1_bin)] = 0
                bin_number_pos_variable.loc[dict(BV1_bin_midpoint = BV1_bin)] = 0
                    
    return bin_mean_variable, bin_number_pos_variable, bin_number_of_samples

### Bin by two variables (Updated 05/23/2022 to handle NaN values in "variabiles_to_be_binned")

def bin_by_two_variables(variable_to_be_binned, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector):
    
    # BV1 and BV2 are the variables to be used to create the bins
    
    # Define bins

    BV1_bin_midpoint = (lower_BV1_bin_limit_vector + upper_BV1_bin_limit_vector) / 2

    lower_BV1_bin_limit_DA = xr.DataArray(lower_BV1_bin_limit_vector, coords=[BV1_bin_midpoint], dims=['BV1_bin_midpoint'])

    upper_BV1_bin_limit_DA = xr.DataArray(upper_BV1_bin_limit_vector, coords=[BV1_bin_midpoint], dims=['BV1_bin_midpoint'])

    number_of_BV1_bins = len(BV1_bin_midpoint)

    BV2_bin_midpoint = (lower_BV2_bin_limit_vector + upper_BV2_bin_limit_vector) / 2;

    lower_BV2_bin_limit_DA = xr.DataArray(lower_BV2_bin_limit_vector, coords=[BV2_bin_midpoint], dims=['BV2_bin_midpoint'])

    upper_BV2_bin_limit_DA = xr.DataArray(upper_BV2_bin_limit_vector, coords=[BV2_bin_midpoint], dims=['BV2_bin_midpoint'])

    number_of_BV2_bins = len(BV2_bin_midpoint);

    # Instantiate composite variable

    coords = {'BV2_bin_midpoint' : BV2_bin_midpoint, 'BV1_bin_midpoint': BV1_bin_midpoint}

    dims = ['BV2_bin_midpoint', 'BV1_bin_midpoint']

    bin_number_of_samples = xr.DataArray(np.full((len(lower_BV2_bin_limit_vector), len(lower_BV1_bin_limit_DA)), np.nan), dims=dims, coords=coords)

    bin_mean_variable = bin_number_of_samples.copy()
    bin_number_pos_variable = bin_number_of_samples.copy()
    
    # Calculate bin mean and number of positive values in each bin

    for BV1_bin in BV1_bin_midpoint:
        for BV2_bin in BV2_bin_midpoint:
            
            bin_index = (BV1 >= lower_BV1_bin_limit_DA.where(lower_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin, drop=True).values) & \
                        (BV1 <= upper_BV1_bin_limit_DA.where(upper_BV1_bin_limit_DA.BV1_bin_midpoint == BV1_bin, drop=True).values) & \
                        (BV2 >= lower_BV2_bin_limit_DA.where(lower_BV2_bin_limit_DA.BV2_bin_midpoint == BV2_bin, drop=True).values) & \
                        (BV2 <= upper_BV2_bin_limit_DA.where(upper_BV2_bin_limit_DA.BV2_bin_midpoint == BV2_bin, drop=True).values)
            
            bin_index_variable_isfinite = xr.apply_ufunc(np.isfinite, variable_to_be_binned.where(bin_index)) # Test that there is valid data where the bin index is
                        
            if bin_index_variable_isfinite.sum() > 0:
                
                bin_mean_variable.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = variable_to_be_binned.where(bin_index_variable_isfinite).mean()
                bin_number_of_samples.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = bin_index_variable_isfinite.sum()
                bin_number_pos_variable.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = (variable_to_be_binned.where(bin_index_variable_isfinite) > 0).sum()
            
            else:
                
                bin_mean_variable.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = np.nan # There is data to determine bins, but the variable to be binned has NaN values
                bin_number_of_samples.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = 0
                bin_number_pos_variable.loc[dict(BV2_bin_midpoint = BV2_bin, BV1_bin_midpoint = BV1_bin)] = 0
                    
    return bin_mean_variable, bin_number_pos_variable, bin_number_of_samples

### Calculate One Variable Binned ivar Composites

def calculate_one_variable_binned_ivar_composites(ivar, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, ivar_name_str, ivar_units_str, BV1_name_str, BV1_units_str, year, fname_datasets_for_simulation, log_bins_boolean):
    
    # BV1 = binning variable 1
    
    current_year_string = str(year)
    
    while len(current_year_string) < 4: # make sure "year" of files has 4 digits for consistent file naming convention
        
        current_year_string = '0' + current_year_string
    
    ###################################################################################
    ####  Calculate Backwards, Forwards, and Centered Differences of BV1 and ivar  ####
    ###################################################################################
    
    print('Calculating Differences')
    
    delta_BV1_leading, delta_BV1_lagging, delta_BV1_centered = calculate_backward_forward_center_difference(BV1)
    delta_ivar_leading, delta_ivar_lagging, delta_ivar_centered = calculate_backward_forward_center_difference(ivar)

    ###############################################################
    ####  Limit data to be composited to the year of interest  ####
    ###############################################################
    
    BV1 = BV1.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV1_leading = delta_BV1_leading.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV1_lagging = delta_BV1_lagging.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV1_centered = delta_BV1_centered.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    ivar = ivar.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_leading = delta_ivar_leading.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_lagging = delta_ivar_lagging.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_centered = delta_ivar_centered.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    ######################
    ####  Bin By BV1  ####
    ######################
        
    print('Binning and Compositing')    
    
    bin_mean_ivar, bin_number_pos_ivar, bin_number_of_samples_ivar  = bin_by_one_variable(ivar, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)

    bin_mean_delta_ivar_leading, bin_number_pos_delta_ivar_leading, bin_number_of_samples_ivar_leading  = bin_by_one_variable(delta_ivar_leading, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)
    bin_mean_delta_ivar_lagging, bin_number_pos_delta_ivar_lagging, bin_number_of_samples_ivar_lagging  = bin_by_one_variable(delta_ivar_lagging, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)
    bin_mean_delta_ivar_centered, bin_number_pos_delta_ivar_centered, bin_number_of_samples_ivar_centered  = bin_by_one_variable(delta_ivar_centered, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)
    
    bin_mean_delta_BV1_leading, bin_number_pos_delta_BV1_leading, bin_number_of_samples_BV1_leading  = bin_by_one_variable(delta_BV1_leading, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)
    bin_mean_delta_BV1_lagging, bin_number_pos_delta_BV1_lagging, bin_number_of_samples_BV1_lagging  = bin_by_one_variable(delta_BV1_lagging, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)
    bin_mean_delta_BV1_centered, bin_number_pos_delta_BV1_centered, bin_number_of_samples_BV1_centered  = bin_by_one_variable(delta_BV1_centered, BV1, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #

    bin_number_of_samples_ivar.name = 'bin_number_of_samples_ivar'
    bin_number_of_samples_ivar_leading.name = 'bin_number_of_samples_ivar_leading'
    bin_number_of_samples_ivar_lagging.name = 'bin_number_of_samples_ivar_lagging'
    bin_number_of_samples_ivar_centered.name = 'bin_number_of_samples_ivar_centered'
    bin_number_of_samples_BV1_leading.name = 'bin_number_of_samples_BV1_leading'
    bin_number_of_samples_BV1_lagging.name = 'bin_number_of_samples_BV1_lagging'
    bin_number_of_samples_BV1_centered.name = 'bin_number_of_samples_BV1_centered'

    bin_mean_ivar.name = 'bin_mean_ivar'
    
    bin_mean_delta_ivar_leading.name = 'bin_mean_delta_ivar_leading'
    bin_number_pos_delta_ivar_leading.name = 'bin_number_pos_delta_ivar_leading'
        
    bin_mean_delta_ivar_lagging.name = 'bin_mean_delta_ivar_lagging'
    bin_number_pos_delta_ivar_lagging.name = 'bin_number_pos_delta_ivar_lagging'
        
    bin_mean_delta_ivar_centered.name = 'bin_mean_delta_ivar_centered'
    bin_number_pos_delta_ivar_centered.name = 'bin_number_pos_delta_ivar_centered'
    
    bin_mean_delta_BV1_leading.name = 'bin_mean_delta_BV1_leading'
    bin_number_pos_delta_BV1_leading.name = 'bin_number_pos_delta_BV1_leading'
        
    bin_mean_delta_BV1_lagging.name = 'bin_mean_delta_BV1_lagging'
    bin_number_pos_delta_BV1_lagging.name = 'bin_number_pos_delta_BV1_lagging'
        
    bin_mean_delta_BV1_centered.name = 'bin_mean_delta_BV1_centered'
    bin_number_pos_delta_BV1_centered.name = 'bin_number_pos_delta_BV1_centered'
                                        
    # Add year dimension to all variables #
                                        
    bin_number_of_samples_ivar = bin_number_of_samples_ivar.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_ivar_leading = bin_number_of_samples_ivar_leading.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_ivar_lagging = bin_number_of_samples_ivar_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_ivar_centered = bin_number_of_samples_ivar_centered.assign_coords(year = year).expand_dims('year')

    bin_mean_ivar = bin_mean_ivar.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_leading = bin_mean_delta_ivar_leading.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_leading = bin_number_pos_delta_ivar_leading.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_lagging = bin_mean_delta_ivar_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_lagging = bin_number_pos_delta_ivar_lagging.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_centered = bin_mean_delta_ivar_centered.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_centered = bin_number_pos_delta_ivar_centered.assign_coords(year = year).expand_dims('year')
    
    bin_mean_delta_BV1_leading = bin_mean_delta_BV1_leading.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_leading = bin_number_pos_delta_BV1_leading.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_BV1_lagging = bin_mean_delta_BV1_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_lagging = bin_number_pos_delta_BV1_lagging.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_BV1_centered = bin_mean_delta_BV1_centered.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_centered = bin_number_pos_delta_BV1_centered.assign_coords(year = year).expand_dims('year')
    
    # Merge all neccessary dataarrays to a single dataset #

    output_dataset = xr.merge([bin_number_of_samples_ivar, bin_number_of_samples_ivar_leading, bin_number_of_samples_ivar_lagging, bin_number_of_samples_ivar_centered, \
                               bin_number_of_samples_BV1_leading, bin_number_of_samples_BV1_lagging, bin_number_of_samples_BV1_centered, \
                               bin_mean_ivar, bin_mean_delta_ivar_leading, bin_number_pos_delta_ivar_leading, \
                               bin_mean_delta_ivar_lagging, bin_number_pos_delta_ivar_lagging, \
                               bin_mean_delta_ivar_centered, bin_number_pos_delta_ivar_centered,\
                               bin_mean_delta_BV1_leading, bin_number_pos_delta_BV1_leading, \
                               bin_mean_delta_BV1_lagging, bin_number_pos_delta_BV1_lagging, \
                               bin_mean_delta_BV1_centered, bin_number_pos_delta_BV1_centered])
    # Add desired attributes #

    output_dataset.attrs['Comments'] = 'Binning variable 1 (BV1) is ' + BV1_name_str + ' with units ' + BV1_units_str + \
                                       ', bin mean ivar is ' + ivar_name_str + ' with units of ' + ivar_units_str

    # Output dataset to NetCDF #
    
    if log_bins_boolean:
        
                output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_log_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')

### Calculate Two Variable Binned Co-Evolution Composites

def calculate_two_variable_binned_coevolution_composites(BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector, BV1_name_str, BV1_units_str, BV2_name_str, BV2_units_str, year, fname_datasets_for_simulation, log_bins_boolean):
    
    # BV1 = binning variable 1, and corresponds to CSF in old code
    # BV2 = binning variable 2, and correspondgs to P in old code
    
    current_year_string = str(year)
    
    while len(current_year_string) < 4: # make sure "year" of files has 4 digits for consistent file naming convention
        
        current_year_string = '0' + current_year_string
    
    ##################################################################################
    ####  Calculate Backwards, Forwards, and Centered Differences of BV1 and BV2  ####
    ##################################################################################
    
    print('Calculating Differences')
    
    delta_BV1_leading, delta_BV1_lagging, delta_BV1_centered = calculate_backward_forward_center_difference(BV1)
    delta_BV2_leading, delta_BV2_lagging, delta_BV2_centered = calculate_backward_forward_center_difference(BV2)

    ###############################################################
    ####  Limit data to be composited to the year of interest  ####
    ###############################################################
    
    BV2 = BV2.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    BV1 = BV1.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    delta_BV2_leading = delta_BV2_leading.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV2_lagging = delta_BV2_lagging.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV2_centered = delta_BV2_centered.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    delta_BV1_leading = delta_BV1_leading.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV1_lagging = delta_BV1_lagging.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_BV1_centered = delta_BV1_centered.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    ##################################
    ####  Bin By Both BV1 and BV2 ####
    ##################################
        
    print('Binning and Compositing')

    _, _, bin_number_of_samples  = bin_by_two_variables(BV2, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_BV1_leading, bin_number_pos_delta_BV1_leading, bin_number_of_samples_leading  = bin_by_two_variables(delta_BV1_leading, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_BV2_leading, bin_number_pos_delta_BV2_leading, _  = bin_by_two_variables(delta_BV2_leading, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_BV1_lagging, bin_number_pos_delta_BV1_lagging, bin_number_of_samples_lagging  = bin_by_two_variables(delta_BV1_lagging, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_BV2_lagging, bin_number_pos_delta_BV2_lagging, _  = bin_by_two_variables(delta_BV2_lagging, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_BV1_centered, bin_number_pos_delta_BV1_centered, bin_number_of_samples_centered  = bin_by_two_variables(delta_BV1_centered, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)
    bin_mean_delta_BV2_centered, bin_number_pos_delta_BV2_centered, _  = bin_by_two_variables(delta_BV2_centered, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #

    bin_number_of_samples.name = 'bin_number_of_samples'
    bin_number_of_samples_leading.name = 'bin_number_of_samples_leading'
    bin_number_of_samples_lagging.name = 'bin_number_of_samples_lagging'
    bin_number_of_samples_centered.name = 'bin_number_of_samples_centered'
        
    bin_mean_delta_BV1_leading.name = 'bin_mean_delta_BV1_leading'
    bin_mean_delta_BV2_leading.name = 'bin_mean_delta_BV2_leading'
    bin_number_pos_delta_BV1_leading.name = 'bin_number_pos_delta_BV1_leading'
    bin_number_pos_delta_BV2_leading.name = 'bin_number_pos_delta_BV2_leading'
        
    bin_mean_delta_BV1_lagging.name = 'bin_mean_delta_BV1_lagging'
    bin_mean_delta_BV2_lagging.name = 'bin_mean_delta_BV2_lagging'
    bin_number_pos_delta_BV1_lagging.name = 'bin_number_pos_delta_BV1_lagging'
    bin_number_pos_delta_BV2_lagging.name = 'bin_number_pos_delta_BV2_lagging'
        
    bin_mean_delta_BV1_centered.name = 'bin_mean_delta_BV1_centered'
    bin_mean_delta_BV2_centered.name = 'bin_mean_delta_BV2_centered'
    bin_number_pos_delta_BV1_centered.name = 'bin_number_pos_delta_BV1_centered'
    bin_number_pos_delta_BV2_centered.name = 'bin_number_pos_delta_BV2_centered'
                                        
    # Add year dimension to all variables #
                                        
    bin_number_of_samples = bin_number_of_samples.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_leading = bin_number_of_samples_leading.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_lagging = bin_number_of_samples_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_centered = bin_number_of_samples_centered.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_BV1_leading = bin_mean_delta_BV1_leading.assign_coords(year = year).expand_dims('year')
    bin_mean_delta_BV2_leading = bin_mean_delta_BV2_leading.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_leading = bin_number_pos_delta_BV1_leading.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV2_leading = bin_number_pos_delta_BV2_leading.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_BV1_lagging = bin_mean_delta_BV1_lagging.assign_coords(year = year).expand_dims('year')
    bin_mean_delta_BV2_lagging = bin_mean_delta_BV2_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_lagging = bin_number_pos_delta_BV1_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV2_lagging = bin_number_pos_delta_BV2_lagging.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_BV1_centered = bin_mean_delta_BV1_centered.assign_coords(year = year).expand_dims('year')
    bin_mean_delta_BV2_centered = bin_mean_delta_BV2_centered.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV1_centered = bin_number_pos_delta_BV1_centered.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_BV2_centered = bin_number_pos_delta_BV2_centered.assign_coords(year = year).expand_dims('year')
    
    # Merge all neccessary dataarrays to a single dataset #

    output_dataset = xr.merge([bin_number_of_samples, bin_number_of_samples_leading, bin_number_of_samples_lagging, bin_number_of_samples_centered, \
                               bin_mean_delta_BV1_leading, bin_mean_delta_BV2_leading, bin_number_pos_delta_BV1_leading, bin_number_pos_delta_BV2_leading, \
                               bin_mean_delta_BV1_lagging, bin_mean_delta_BV2_lagging, bin_number_pos_delta_BV1_lagging, bin_number_pos_delta_BV2_lagging, \
                               bin_mean_delta_BV1_centered, bin_mean_delta_BV2_centered, bin_number_pos_delta_BV1_centered, bin_number_pos_delta_BV2_centered])
    # Add desired attributes #

    output_dataset.attrs['Comments'] = 'Binning variable 1 (BV1) is ' + BV1_name_str + ' with units ' + BV1_units_str + \
                                        ', binning variable 2 (BV2) is ' + BV2_name_str + ' with units ' + BV2_units_str

    # Output dataset to NetCDF #
    
    if log_bins_boolean:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_' + BV2_name_str + '_log_binned_coevolution_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_' + BV2_name_str + '_binned_coevolution_composite' + '_' + current_year_string + '.nc', 'w')

### Calculate Two Variable Binned ivar Composites

def calculate_two_variable_binned_ivar_composites(ivar, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector, ivar_name_str, ivar_units_str, BV1_name_str, BV1_units_str, BV2_name_str, BV2_units_str, year, fname_datasets_for_simulation, log_bins_boolean):
    
    # BV1 = binning variable 1, and corresponds to CSF in old code
    # BV2 = binning variable 2, and correspondgs to P in old code
    
    current_year_string = str(year)
    
    while len(current_year_string) < 4: # make sure "year" of files has 4 digits for consistent file naming convention
        
        current_year_string = '0' + current_year_string
    
    ##################################################################################
    ####  Calculate Backwards, Forwards, and Centered Differences of BV1 and BV2  ####
    ##################################################################################
    
    print('Calculating Differences')
    
    delta_ivar_leading, delta_ivar_lagging, delta_ivar_centered = calculate_backward_forward_center_difference(ivar)

    ###############################################################
    ####  Limit data to be composited to the year of interest  ####
    ###############################################################
    
    BV2 = BV2.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    BV1 = BV1.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    ivar = ivar.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_leading = delta_ivar_leading.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_lagging = delta_ivar_lagging.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
    delta_ivar_centered = delta_ivar_centered.sel(time = slice(current_year_string+'-01-01', current_year_string+'-12-31'))
        
    ##################################
    ####  Bin By Both BV1 and BV2 ####
    ##################################
        
    print('Binning and Compositing')    
    
    bin_mean_ivar, bin_number_pos_ivar, bin_number_of_samples_ivar  = bin_by_two_variables(ivar, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_ivar_leading, bin_number_pos_delta_ivar_leading, bin_number_of_samples_leading  = bin_by_two_variables(delta_ivar_leading, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_ivar_lagging, bin_number_pos_delta_ivar_lagging, bin_number_of_samples_lagging  = bin_by_two_variables(delta_ivar_lagging, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    bin_mean_delta_ivar_centered, bin_number_pos_delta_ivar_centered, bin_number_of_samples_centered  = bin_by_two_variables(delta_ivar_centered, BV1, BV2, lower_BV1_bin_limit_vector, upper_BV1_bin_limit_vector, lower_BV2_bin_limit_vector, upper_BV2_bin_limit_vector)

    ####  Output Data as NetCDF  ####

    # Name variables #

    bin_number_of_samples_ivar.name = 'bin_number_of_samples_ivar'
    bin_number_of_samples_leading.name = 'bin_number_of_samples_leading'
    bin_number_of_samples_lagging.name = 'bin_number_of_samples_lagging'
    bin_number_of_samples_centered.name = 'bin_number_of_samples_centered'

    bin_mean_ivar.name = 'bin_mean_ivar'
        
    bin_mean_delta_ivar_leading.name = 'bin_mean_delta_ivar_leading'
    bin_number_pos_delta_ivar_leading.name = 'bin_number_pos_delta_ivar_leading'
        
    bin_mean_delta_ivar_lagging.name = 'bin_mean_delta_ivar_lagging'
    bin_number_pos_delta_ivar_lagging.name = 'bin_number_pos_delta_ivar_lagging'
        
    bin_mean_delta_ivar_centered.name = 'bin_mean_delta_ivar_centered'
    bin_number_pos_delta_ivar_centered.name = 'bin_number_pos_delta_ivar_centered'
                                        
    # Add year dimension to all variables #
                                        
    bin_number_of_samples_ivar = bin_number_of_samples_ivar.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_leading = bin_number_of_samples_leading.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_lagging = bin_number_of_samples_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_of_samples_centered = bin_number_of_samples_centered.assign_coords(year = year).expand_dims('year')

    bin_mean_ivar = bin_mean_ivar.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_leading = bin_mean_delta_ivar_leading.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_leading = bin_number_pos_delta_ivar_leading.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_lagging = bin_mean_delta_ivar_lagging.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_lagging = bin_number_pos_delta_ivar_lagging.assign_coords(year = year).expand_dims('year')
        
    bin_mean_delta_ivar_centered = bin_mean_delta_ivar_centered.assign_coords(year = year).expand_dims('year')
    bin_number_pos_delta_ivar_centered = bin_number_pos_delta_ivar_centered.assign_coords(year = year).expand_dims('year')
    
    # Merge all neccessary dataarrays to a single dataset #

    output_dataset = xr.merge([bin_number_of_samples_ivar, bin_number_of_samples_leading, bin_number_of_samples_lagging, bin_number_of_samples_centered, \
                               bin_mean_ivar, bin_mean_delta_ivar_leading, bin_number_pos_delta_ivar_leading, \
                               bin_mean_delta_ivar_lagging, bin_number_pos_delta_ivar_lagging, \
                               bin_mean_delta_ivar_centered, bin_number_pos_delta_ivar_centered])
    # Add desired attributes #

    output_dataset.attrs['Comments'] = 'Binning variable 1 (BV1) is ' + BV1_name_str + ' with units ' + BV1_units_str + \
                                        ', binning variable 2 (BV2) is ' + BV2_name_str + ' with units ' + BV2_units_str + \
                                        ', bin mean ivar is ' + ivar_name_str + ' with units of ' + ivar_units_str

    # Output dataset to NetCDF #
    
    if log_bins_boolean:
        
                output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_' + BV2_name_str + '_log_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + BV1_name_str + '_' + BV2_name_str + '_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')

### Process Multiyear One Variable Binned ivar Composites

def process_multiyear_one_variable_binned_ivar_composites(list_of_files):
    
    one_variable_binned_ivar_composites = xr.open_mfdataset(list_of_files, combine="by_coords")
    
    # Calculate the bin means over all years #

    more_than_zero_obs_mask_ivar = one_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year') > 0
    
    more_than_zero_obs_mask_ivar_leading= one_variable_binned_ivar_composites.bin_number_of_samples_ivar_leading.sum('year') > 0
    
    more_than_zero_obs_mask_ivar_lagging = one_variable_binned_ivar_composites.bin_number_of_samples_ivar_lagging.sum('year') > 0
    
    more_than_zero_obs_mask_ivar_centered = one_variable_binned_ivar_composites.bin_number_of_samples_ivar_centered.sum('year') > 0
    
    more_than_zero_obs_mask_BV1_leading= one_variable_binned_ivar_composites.bin_number_of_samples_BV1_leading.sum('year') > 0
    
    more_than_zero_obs_mask_BV1_lagging = one_variable_binned_ivar_composites.bin_number_of_samples_BV1_lagging.sum('year') > 0
    
    more_than_zero_obs_mask_BV1_centered = one_variable_binned_ivar_composites.bin_number_of_samples_BV1_centered.sum('year') > 0
    
    one_variable_binned_ivar_composites['bin_mean_ivar'] = (one_variable_binned_ivar_composites.bin_mean_ivar * one_variable_binned_ivar_composites.bin_number_of_samples_ivar).sum('year').where(more_than_zero_obs_mask_ivar, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year').where(more_than_zero_obs_mask_ivar, other=np.nan) 

    one_variable_binned_ivar_composites['bin_mean_delta_ivar_leading'] = (one_variable_binned_ivar_composites.bin_mean_delta_ivar_leading * one_variable_binned_ivar_composites.bin_number_of_samples_ivar_leading).sum('year').where(more_than_zero_obs_mask_ivar_leading, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_ivar_leading.sum('year').where(more_than_zero_obs_mask_ivar_leading, other=np.nan) 

    one_variable_binned_ivar_composites['bin_mean_delta_ivar_lagging'] = (one_variable_binned_ivar_composites.bin_mean_delta_ivar_lagging * one_variable_binned_ivar_composites.bin_number_of_samples_ivar_lagging).sum('year').where(more_than_zero_obs_mask_ivar_lagging, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_ivar_lagging.sum('year').where(more_than_zero_obs_mask_ivar_lagging, other=np.nan) 

    one_variable_binned_ivar_composites['bin_mean_delta_ivar_centered'] = (one_variable_binned_ivar_composites.bin_mean_delta_ivar_centered * one_variable_binned_ivar_composites.bin_number_of_samples_ivar_centered).sum('year').where(more_than_zero_obs_mask_ivar_centered, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_ivar_centered.sum('year').where(more_than_zero_obs_mask_ivar_centered, other=np.nan) 
    
    one_variable_binned_ivar_composites['bin_mean_delta_BV1_leading'] = (one_variable_binned_ivar_composites.bin_mean_delta_BV1_leading * one_variable_binned_ivar_composites.bin_number_of_samples_BV1_leading).sum('year').where(more_than_zero_obs_mask_BV1_leading, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_BV1_leading.sum('year').where(more_than_zero_obs_mask_BV1_leading, other=np.nan) 

    one_variable_binned_ivar_composites['bin_mean_delta_BV1_lagging'] = (one_variable_binned_ivar_composites.bin_mean_delta_BV1_lagging * one_variable_binned_ivar_composites.bin_number_of_samples_BV1_lagging).sum('year').where(more_than_zero_obs_mask_BV1_lagging, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_BV1_lagging.sum('year').where(more_than_zero_obs_mask_BV1_lagging, other=np.nan) 

    one_variable_binned_ivar_composites['bin_mean_delta_BV1_centered'] = (one_variable_binned_ivar_composites.bin_mean_delta_BV1_centered * one_variable_binned_ivar_composites.bin_number_of_samples_BV1_centered).sum('year').where(more_than_zero_obs_mask_BV1_centered, other=np.nan) \
                                                                   / one_variable_binned_ivar_composites.bin_number_of_samples_BV1_centered.sum('year').where(more_than_zero_obs_mask_BV1_centered, other=np.nan) 

    # Sum number of observations in each bin over all years #
    
    one_variable_binned_ivar_composites['bin_number_of_samples_ivar'] = one_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_ivar_leading'] = one_variable_binned_ivar_composites.bin_number_of_samples_ivar_leading.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_ivar_lagging'] = one_variable_binned_ivar_composites.bin_number_of_samples_ivar_lagging.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_ivar_centered'] = one_variable_binned_ivar_composites.bin_number_of_samples_ivar_centered.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_BV1_leading'] = one_variable_binned_ivar_composites.bin_number_of_samples_BV1_leading.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_BV1_lagging'] = one_variable_binned_ivar_composites.bin_number_of_samples_BV1_lagging.sum('year')
    one_variable_binned_ivar_composites['bin_number_of_samples_BV1_centered'] = one_variable_binned_ivar_composites.bin_number_of_samples_BV1_centered.sum('year')

    one_variable_binned_ivar_composites['bin_number_pos_delta_ivar_leading'] = one_variable_binned_ivar_composites.bin_number_pos_delta_ivar_leading.sum('year')

    one_variable_binned_ivar_composites['bin_number_pos_delta_ivar_lagging'] = one_variable_binned_ivar_composites.bin_number_pos_delta_ivar_lagging.sum('year')

    one_variable_binned_ivar_composites['bin_number_pos_delta_ivar_centered'] = one_variable_binned_ivar_composites.bin_number_pos_delta_ivar_centered.sum('year')
    
    one_variable_binned_ivar_composites['bin_number_pos_delta_BV1_leading'] = one_variable_binned_ivar_composites.bin_number_pos_delta_BV1_leading.sum('year')

    one_variable_binned_ivar_composites['bin_number_pos_delta_BV1_lagging'] = one_variable_binned_ivar_composites.bin_number_pos_delta_BV1_lagging.sum('year')

    one_variable_binned_ivar_composites['bin_number_pos_delta_BV1_centered'] = one_variable_binned_ivar_composites.bin_number_pos_delta_BV1_centered.sum('year')
    
    # Remove year dimension
        
    one_variable_binned_ivar_composites = one_variable_binned_ivar_composites.squeeze()
        
    return one_variable_binned_ivar_composites 

def process_multiyear_two_variable_binned_coevolution_composites(list_of_files):
    
    two_variable_binned_coevolution_composites = xr.open_mfdataset(list_of_files, combine="by_coords")
            
    # Calculate the bin means over all years #

    more_than_zero_obs_mask = two_variable_binned_coevolution_composites.bin_number_of_samples.sum('year') > 0
    
    more_than_zero_obs_mask_leading= two_variable_binned_coevolution_composites.bin_number_of_samples_leading.sum('year') > 0
    
    more_than_zero_obs_mask_lagging = two_variable_binned_coevolution_composites.bin_number_of_samples_lagging.sum('year') > 0
    
    more_than_zero_obs_mask_centered = two_variable_binned_coevolution_composites.bin_number_of_samples_centered.sum('year') > 0

    two_variable_binned_coevolution_composites['bin_mean_delta_BV1_leading'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV1_leading * two_variable_binned_coevolution_composites.bin_number_of_samples_leading).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_leading.sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) 

    two_variable_binned_coevolution_composites['bin_mean_delta_BV2_leading'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV2_leading * two_variable_binned_coevolution_composites.bin_number_of_samples_leading).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_leading.sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) 

    two_variable_binned_coevolution_composites['bin_mean_delta_BV1_lagging'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV1_lagging * two_variable_binned_coevolution_composites.bin_number_of_samples_lagging).sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_lagging.sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) 

    two_variable_binned_coevolution_composites['bin_mean_delta_BV2_lagging'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV2_lagging * two_variable_binned_coevolution_composites.bin_number_of_samples_lagging).sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_lagging.sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) 

    two_variable_binned_coevolution_composites['bin_mean_delta_BV1_centered'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV1_centered * two_variable_binned_coevolution_composites.bin_number_of_samples_centered).sum('year').where(more_than_zero_obs_mask_centered, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_centered.sum('year').where(more_than_zero_obs_mask_centered, other=np.nan) 

    two_variable_binned_coevolution_composites['bin_mean_delta_BV2_centered'] = (two_variable_binned_coevolution_composites.bin_mean_delta_BV2_centered * two_variable_binned_coevolution_composites.bin_number_of_samples_centered).sum('year').where(more_than_zero_obs_mask_centered, other=np.nan) \
                                                                   / two_variable_binned_coevolution_composites.bin_number_of_samples_centered.sum('year').where(more_than_zero_obs_mask_centered, other=np.nan)
    
    # Sum number of observations in each bin over all years #
    
    two_variable_binned_coevolution_composites['bin_number_of_samples'] = two_variable_binned_coevolution_composites.bin_number_of_samples.sum('year')
    two_variable_binned_coevolution_composites['bin_number_of_samples_leading'] = two_variable_binned_coevolution_composites.bin_number_of_samples_leading.sum('year')
    two_variable_binned_coevolution_composites['bin_number_of_samples_lagging'] = two_variable_binned_coevolution_composites.bin_number_of_samples_lagging.sum('year')
    two_variable_binned_coevolution_composites['bin_number_of_samples_centered'] = two_variable_binned_coevolution_composites.bin_number_of_samples_centered.sum('year')

    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV1_leading'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV1_leading.sum('year')
    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV2_leading'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV2_leading.sum('year')

    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV1_lagging'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV1_lagging.sum('year')
    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV2_lagging'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV2_lagging.sum('year')

    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV1_centered'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV1_centered.sum('year')
    two_variable_binned_coevolution_composites['bin_number_pos_delta_BV2_centered'] = two_variable_binned_coevolution_composites.bin_number_pos_delta_BV2_centered.sum('year')
    
    # Remove year dimension
        
    two_variable_binned_coevolution_composites = two_variable_binned_coevolution_composites.squeeze()
        
    return two_variable_binned_coevolution_composites

### Process Multiyear Two Variable Binned ivar Composites

def process_multiyear_two_variable_binned_ivar_composites(list_of_files):
    
    two_variable_binned_ivar_composites = xr.open_mfdataset(list_of_files, combine="by_coords")
            
    # Calculate the bin means over all years #

    more_than_zero_obs_mask_ivar = two_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year') > 0
    
    more_than_zero_obs_mask_leading= two_variable_binned_ivar_composites.bin_number_of_samples_leading.sum('year') > 0
    
    more_than_zero_obs_mask_lagging = two_variable_binned_ivar_composites.bin_number_of_samples_lagging.sum('year') > 0
    
    more_than_zero_obs_mask_centered = two_variable_binned_ivar_composites.bin_number_of_samples_centered.sum('year') > 0
    
    two_variable_binned_ivar_composites['bin_mean_ivar'] = (two_variable_binned_ivar_composites.bin_mean_ivar * two_variable_binned_ivar_composites.bin_number_of_samples_ivar).sum('year').where(more_than_zero_obs_mask_ivar, other=np.nan) \
                                                                   / two_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year').where(more_than_zero_obs_mask_ivar, other=np.nan) 

    two_variable_binned_ivar_composites['bin_mean_delta_ivar_leading'] = (two_variable_binned_ivar_composites.bin_mean_delta_ivar_leading * two_variable_binned_ivar_composites.bin_number_of_samples_leading).sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) \
                                                                   / two_variable_binned_ivar_composites.bin_number_of_samples_leading.sum('year').where(more_than_zero_obs_mask_leading, other=np.nan) 

    two_variable_binned_ivar_composites['bin_mean_delta_ivar_lagging'] = (two_variable_binned_ivar_composites.bin_mean_delta_ivar_lagging * two_variable_binned_ivar_composites.bin_number_of_samples_lagging).sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) \
                                                                   / two_variable_binned_ivar_composites.bin_number_of_samples_lagging.sum('year').where(more_than_zero_obs_mask_lagging, other=np.nan) 

    two_variable_binned_ivar_composites['bin_mean_delta_ivar_centered'] = (two_variable_binned_ivar_composites.bin_mean_delta_ivar_centered * two_variable_binned_ivar_composites.bin_number_of_samples_centered).sum('year').where(more_than_zero_obs_mask_centered, other=np.nan) \
                                                                   / two_variable_binned_ivar_composites.bin_number_of_samples_centered.sum('year').where(more_than_zero_obs_mask_centered, other=np.nan) 
    
    # Sum number of observations in each bin over all years #
    
    two_variable_binned_ivar_composites['bin_number_of_samples_ivar'] = two_variable_binned_ivar_composites.bin_number_of_samples_ivar.sum('year')
    two_variable_binned_ivar_composites['bin_number_of_samples_leading'] = two_variable_binned_ivar_composites.bin_number_of_samples_leading.sum('year')
    two_variable_binned_ivar_composites['bin_number_of_samples_lagging'] = two_variable_binned_ivar_composites.bin_number_of_samples_lagging.sum('year')
    two_variable_binned_ivar_composites['bin_number_of_samples_centered'] = two_variable_binned_ivar_composites.bin_number_of_samples_centered.sum('year')

    two_variable_binned_ivar_composites['bin_number_pos_delta_ivar_leading'] = two_variable_binned_ivar_composites.bin_number_pos_delta_ivar_leading.sum('year')

    two_variable_binned_ivar_composites['bin_number_pos_delta_ivar_lagging'] = two_variable_binned_ivar_composites.bin_number_pos_delta_ivar_lagging.sum('year')

    two_variable_binned_ivar_composites['bin_number_pos_delta_ivar_centered'] = two_variable_binned_ivar_composites.bin_number_pos_delta_ivar_centered.sum('year')
    
    # Remove year dimension
        
    two_variable_binned_ivar_composites = two_variable_binned_ivar_composites.squeeze()
        
    return two_variable_binned_ivar_composites
