from lowtrop_pad.profile_analysis import (
    calculate_and_save_differences,
    calculate_and_save_absolute_differences,
    calculate_mean_differences,
    extract_times_from_merged_profiles,
    time_array_of_profile_difference,
    resample_and_interpolate_time_arrays,
    single_profiles_to_array, 
    identify_air_mass_changes,
    categorize_by_wind_direction,
)

if True:
    identify_air_mass_changes(
        input_directory=r"data\merged_interpol_profiles",
        met_station_directory = r'data\met_stations\Asiaq_met_VRS.csv',
        time_delta = '2h',
        temp_diff = 3,
        output_directory=r"results\conditional_profiles")
if True:
    categorize_by_wind_direction(
    input_directory=r"data\merged_interpol_profiles",
    met_station_directory=r'data\met_stations\Asiaq_met_VRS.csv',
    time_delta='1h',
    output_directory=r"results\conditional_profiles"
)
    
# Calculating differences xq2-carra and xq2-era5 for all profiles and store also seperated after surfaces
if True:
    calculate_and_save_differences(
        input_directory=r"data\merged_interpol_profiles",
        output_directory=r"results\differences_xq2_reanalyis",
    )

# Calculating absolute differences xq2-carra and xq2-era5 for all profiles and store also seperated after surfaces
if True:
    calculate_and_save_absolute_differences(
        input_directory=r"data\merged_interpol_profiles",
        output_directory=r"results\absolute_differences_xq2_reanalysis",
    )

# Calculating mean differences and mean absolute differences xq2-carra and xq2-era5 for all profiles and  also seperated after surfaces
if True:
    calculate_mean_differences(
        input_directory=r"results\differences_xq2_reanalyis",
        output_directory=r"results\mean_differences_xq2_reanalysis",
    )

if True:
    calculate_mean_differences(
        input_directory=r"results\absolute_differences_xq2_reanalysis",
        output_directory=r"results\mean_absolute_differences_xq2_reanalysis",
    )

# Build array over time for single xq2 profiles, then resample and interpolate them
if True:
    profile_time_map = extract_times_from_merged_profiles(
        profile_directory=r"data\merged_interpol_profiles"
    )

    single_profiles_to_array(profile_time_map = profile_time_map, 
    input_directory=r'data\merged_interpol_profiles',
    variable='xq2_T', 
    output_directory='results/profiles_over_time')

    resample_and_interpolate_time_arrays(input_directory='results/profiles_over_time',
    output_directory='results/profiles_over_time_resampled',
    resample_interval='1h',
    interpolation_method='linear')

# Build array over time for single GRADIENT profiles, then resample and interpolate them
if True:
    profile_time_map = extract_times_from_merged_profiles(
        profile_directory=r"data\gradients_merged_interpol_profiles"
    )

    single_profiles_to_array(profile_time_map = profile_time_map, 
    input_directory=r'data\gradients_merged_interpol_profiles',
    variable='xq2_T_grad', 
    output_directory='results/gradients_over_time')

    resample_and_interpolate_time_arrays(input_directory='results/gradients_over_time',
    output_directory='results/gradients_over_time_resampled',
    resample_interval='1h',
    interpolation_method='linear')
    
# Extract times from merged interpolated profile files and Replace column names with times in the differences files
if True:
    profile_time_map = extract_times_from_merged_profiles(
        profile_directory=r"data\merged_interpol_profiles"
    )
    # absolute differences
    time_array_of_profile_difference(
        profile_time_map=profile_time_map,
        input_directory=r"results\absolute_differences_xq2_reanalysis",
        output_directory=r"results\absolute_differences_xq2_reanalysis_time_arrays",
    )

    # differences
    time_array_of_profile_difference(
        profile_time_map=profile_time_map,
        input_directory=r"results\differences_xq2_reanalyis",
        output_directory=r"results\differences_xq2_reanalysis_time_arrays",
    )

if True:
    resample_and_interpolate_time_arrays(
        input_directory=r"results\differences_xq2_reanalysis_time_arrays",
        output_directory=r"results\differences_xq2_reanalysis_time_arrays_resampled",
        resample_interval="1h",
        interpolation_method="linear",
    )
    resample_and_interpolate_time_arrays(
        input_directory=r"results\absolute_differences_xq2_reanalysis_time_arrays",
        output_directory=r"results\absolute_differences_xq2_reanalysis_time_arrays_resampled",
        resample_interval="1h",
        interpolation_method="linear",
    )
