## List of variables and functions used in the POD package

## Dataset names and attributes used in code


## Dimension varaibles

| Var name | description | Units | Order |
| ---------| ----------- | ----- | ----- |
| full_lat | latitude | degrees | increasing |
| full_lon | longitude | degrees | increasing |


## physical variables that are used in computations

| Var name | description | Units | Dimensions |
| -------- | ----------- | ----- | ---------- |
| PS | surface pressure | Pa | time x lat x lon |
| Q | specific humidity | Kg/Kg | time x lat x lon x level |
| T | air temperature | K | time x lat x lon x level |
| precipitation_rate | surface precipitation rate  | mm/day | time x lat x lon |
| saturation_specific_humidity | saturation specific humidity | Kg/Kg | time x lat x lon x level |
| ci_q | mass weighted vertical integral of specific humidity | Kg/Kg | time x lat x lon |
| ci_q_sat | mass weighted vertical integral of saturation specific humidity | Kg/Kg | time x lat x lon |




## Internal variables being used by different functions

| Var name | description | functions that use it |
| -------- | ----------- | --------------------- |
| ivar | physical variable that you want to composite | calculate_one_variable_binned_ivar_comopsites |
| BV1 | binning variable 1 - ivar is binned over this | calculate_one_variable_binned_ivar_comopsites |
| lower_BV1_bin_limit_vector | lower boundaries of BV1 bins | calculate_one_variable_binned_ivar_comopsites |
| upper_BV1_bin_limit_vector | upper boundaries of BV1 bins | calculate_one_variable_binned_ivar_comopsites |
| BV2 | binning variable 2 - ivar is binned over this and BV1 | calculate_two_variable_binned_ivar_comopsites |
| lower_BV2_bin_limit_vector | lower boundaries of BV2 bins | calculate_two_variable_binned_ivar_comopsites |
| upper_BV2_bin_limit_vector | upper boundaries of BV2 bins | calculate_two_variable_binned_ivar_comopsites |
| bin_mean_ivar | ivar as function of BV1 bins | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_ivar_leading | bin mean forward difference of ivar | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_ivar_lagging | bin mean backward difference of ivar | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_ivar_centered | bin mean centered difference of ivar | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_BV1_leading | bin mean forward difference of BV1 | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_BV1_lagging | bin mean backward difference of BV1 | calculate_one_variable_binned_ivar_comopsites |
| bin_mean_delta_BV1_centered | bin mean centered difference of BV1 | calculate_one_variable_binned_ivar_comopsites |




## Code structure variables - what user might want to edit

| start_year | starting year for time period being analyzed |
| ---------- | -------------------------------------------- |
| end_year | ending year for time period being analyzed |
