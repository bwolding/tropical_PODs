import numpy as np
import xarray as xr

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
        
        bin_index_variable_isfinite = xr.ufuncs.isfinite(variable_to_be_binned.where(bin_index)) # Test that there is valid data where the bin index is
        
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
            
            bin_index_variable_isfinite = xr.ufuncs.isfinite(variable_to_be_binned.where(bin_index)) # Test that there is valid data where the bin index is
                        
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
        
                output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_log_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')

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
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_' + BV2_name_str + '_log_binned_coevolution_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_' + BV2_name_str + '_binned_coevolution_composite' + '_' + current_year_string + '.nc', 'w')

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
        
                output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_' + BV2_name_str + '_log_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')
        
    else:
        
        output_dataset.to_netcdf(fname_datasets_for_simulation + '_'+ BV1_name_str + '_' + BV2_name_str + '_binned_' + ivar_name_str + '_composite' + '_' + current_year_string + '.nc', 'w')

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



