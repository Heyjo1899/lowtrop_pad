
from lowtrop_pad.profile_analysis import (
    calculate_and_save_differences,
    calculate_and_save_absolute_differences,
    calculate_mean_differences,
    extract_times_from_merged_profiles,
    time_array_of_profile_difference,
    resample_and_interpolate_time_arrays
)
# Calculating differences xq2-carra and xq2-era5 for all profiles and store also seperated after surfaces
if False:
    calculate_and_save_differences(
        input_directory = r'data\merged_interpol_profiles', 
        output_directory = r'results\differences_xq2_reanalyis'
        )

# Calculating absolute differences xq2-carra and xq2-era5 for all profiles and store also seperated after surfaces
if False:
    calculate_and_save_absolute_differences(
        input_directory = r'data\merged_interpol_profiles',
        output_directory= r'results\absolute_differences_xq2_reanalysis'
    )

# Calculating mean differences and mean absolute differences xq2-carra and xq2-era5 for all profiles and  also seperated after surfaces
if False:
    calculate_mean_differences(
        input_directory = r'results\differences_xq2_reanalyis',
        output_directory = r'results\mean_differences_xq2_reanalysis'
    )

if False:
    calculate_mean_differences(
        input_directory = r'results\absolute_differences_xq2_reanalysis',
        output_directory = r'results\mean_absolute_differences_xq2_reanalysis'
    )
    
# Extract times from merged interpolated profile files and Replace column names with times in the differences files
if False:
    profile_time_map = extract_times_from_merged_profiles(
        profile_directory = r'data\merged_interpol_profiles'
        )
    # absolute differences
    time_array_of_profile_difference(
        profile_time_map = profile_time_map,
        input_directory = r'results\absolute_differences_xq2_reanalysis',
        output_directory=r'results\absolute_differences_xq2_reanalysis_time_arrays'
        )

    # differences
    time_array_of_profile_difference(
        profile_time_map = profile_time_map,
        input_directory = r'results\differences_xq2_reanalyis',
        output_directory=r'results\differences_xq2_reanalysis_time_arrays'
        )

if True: 
    resample_and_interpolate_time_arrays(
        input_directory = r'results\differences_xq2_reanalysis_time_arrays',
        output_directory = r'results\differences_xq2_reanalysis_time_arrays_resampled',
        resample_interval='1H', interpolation_method='linear'
    )
    resample_and_interpolate_time_arrays(
        input_directory = r'results\absolute_differences_xq2_reanalysis_time_arrays',
        output_directory = r'results\absolute_differences_xq2_reanalysis_time_arrays_resampled',
            resample_interval='1H', interpolation_method='linear'
    )