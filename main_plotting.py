import os
from lowtrop_pad.plotting_profiles import (
    plot_raw_and_smoothed_profiles_of_day,
    plot_xq2_vs_reanalysis_profiles_of_day,
    plot_Asiaq_station_data,
    plot_merged_and_resampled_profiles,
    plot_mean_differences,
    plot_differences_array,
    plot_differences_array_resampled,
    plot_profiles_array_resampled,
)
from lowtrop_pad.plotting_synoptic import plot_era5_synoptic

# Plotting only xq2 profiles of a specific Day raw and smoothed
if False:
    date = "20230814"
    directory_path_smoothed = os.path.join("data", "xq2", "averaged_profiles_20", date)
    directory_path_raw = os.path.join("data", "xq2", "by_flight", date)
    directory_path_smoothed_custom = os.path.join(
        "data", "xq2", "averaged_profiles_custom_3_5_10_20", date
    )

    plot_raw_and_smoothed_profiles_of_day(
        date,
        directory_path_smoothed_custom,
        directory_path_raw,
        varname="T",
        file_ending=".csv",
    )

# Plotting xq2 and raw reanalysis profiles of a specific Day
if False:
    date = "20230815"
    xq_2_path = os.path.join("data/xq2/averaged_profiles_custom_3_5_10_20", date)
    carra_path = os.path.join("data/reanalysis/CARRA_extracted_profiles", date)
    era5_path = os.path.join("data/reanalysis/ERA5_extracted_profiles", date)
    plot_xq2_vs_reanalysis_profiles_of_day(
        date,
        xq_2_path,
        carra_path,
        era5_path,
        x_varname="T",
        y_varname="alt_ag",
        file_ending=".csv",
        savefig=True,
        output_path="plots\\plots_of_day\\",
        output_filename=None,
    )

# Plotting all profiles (not resampled) of each day over LOOP
if False:
    base_xq2_path = "data/xq2/averaged_profiles_custom_3_5_10_20"
    base_carra_path = "data/reanalysis/CARRA_extracted_profiles"
    base_era5_path = "data/reanalysis/ERA5_extracted_profiles"
    output_base_path = "plots\\plots_of_day"

    # Get the list of dates from the xq2 directory
    dates = next(os.walk(base_xq2_path), (None, None, []))[
        1
    ]  # Lists directories inside base_xq2_path

    # Loop over the dates and execute the plotting function for each
    for date in dates:
        xq_2_date_path = os.path.join(base_xq2_path, date)
        carra_date_path = os.path.join(base_carra_path, date)
        era5_date_path = os.path.join(base_era5_path, date)

        # Execute the plotting function for each date
        plot_xq2_vs_reanalysis_profiles_of_day(
            date,
            xq_2_date_path,
            carra_date_path,
            era5_date_path,
            x_varname="T",
            y_varname="alt_ag",
            file_ending=".csv",
            savefig=True,
            output_path=output_base_path,
            output_filename=None,
        )

# Plotting Asiaq station data over specified Period (Atmospheric Monitoring Hut)
if False:
    start_date = "2023-08-21 21:00:00"
    end_date = "2023-08-21 23:00:00"
    plot_Asiaq_station_data(
        "data\\met_stations\\Asiaq_met_VRS.csv", start_date, end_date
    )

# Plotting merged and resampled profiles of all days over loop
if True:
    interpol_path = r"data\merged_interpol_profiles"
    output_dir = r"plots\plots_of_day_merged"

    for root, dirs, files in os.walk(interpol_path):
        for dir in dirs:
            plot_merged_and_resampled_profiles(
                file_path=os.path.join(root, dir),
                x_varname="T",
                y_varname="alt_ag",
                file_ending=".csv",
                output_path=output_dir,
                output_filename=f"{dir}_temperature_profiles.png",
            )

# plotting synoptic setting single plot for a specific date and time
if False:
    date = "20230801"
    time = "12"
    path_to_file1 = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic.nc"
    path_to_file2 = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic_t2m.nc"
    temp_level = 850
    geopotential_level = 500
    output_path = "plots\\synoptic"
    output_filename = (
        f"temp_geopotential_{temp_level}_{geopotential_level}_{time}_{date}.png"
    )

    plot_era5_synoptic(
        date=date,
        time=time,
        path_to_file1=path_to_file1,
        path_to_file2=path_to_file2,
        temp_level=temp_level,
        geopotential_level=geopotential_level,
        output_path=output_path,
        output_filename=output_filename,
    )

# Function to plot synoptic settings for all days and time steps
if False:
    base_path = "data/reanalysis/ERA5_extracted_profiles"  # Path to the folder containing date subfolders
    nc_file_path1 = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic.nc"
    nc_file_path2 = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic_t2m.nc"
    temp_level = 850  # Set the temperature level
    geopotential_level = 500  # Set the geopotential height level
    output_path = "plots/synoptic"

    time_steps = ["06", "12", "18"]

    # Get the list of dates by checking subfolders in the base_path directory
    dates = [
        folder
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]

    # Iterate through each date and time step
    for date in dates:
        for time in time_steps:
            print(f"Processing {date} at {time} UTC...")
            # Define the output file path and name
            output_filename = (
                f"temp_geopotential_{temp_level}_{geopotential_level}_{date}_{time}.png"
            )

            # Plot and save the figure
            plot_era5_synoptic(
                date=date,
                time=time,
                path_to_file1=nc_file_path1,
                path_to_file2=nc_file_path2,
                temp_level=temp_level,
                geopotential_level=geopotential_level,
                output_path=output_path,
                output_filename=output_filename,
            )

# Plotting mean differences of XQ2 and Reanalysis
if True:
    plot_mean_differences(
        input_directory=r"results\mean_differences_xq2_reanalysis",
        output_directory=r"plots\differences_xq2_reanalysis",
        add_std=True
    )

# Plotting mean Absolute differences of XQ2 and Reanalysis
if True:
    plot_mean_differences(
        input_directory=r"results\mean_absolute_differences_xq2_reanalysis",
        output_directory=r"plots\differences_xq2_reanalysis",
        add_std=True
    )

# Plotting differnces of xq2 and reanalysis as time array
if True:
    input_directory = r"results\differences_xq2_reanalysis_time_arrays"
    output_directory = r"plots\delta_T_over_time"

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
        plot_differences_array(file_path, output_directory)
if False:
    input_directory = r"results\differences_xq2_reanalysis_time_arrays"
    output_directory = r"plots\delta_T_over_time"

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
        plot_differences_array(file_path, output_directory)

# Plotting differnces of xq2 and reanylsis as time array RESAMPLED
if True:
    input_directory = r"results\differences_xq2_reanalysis_time_arrays_resampled"
    output_directory = r"plots\delta_T_over_time_resampled"

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
        plot_differences_array_resampled(file_path, output_directory, upper_ylim=480)


if True:
    input_directory = (
        r"results\absolute_differences_xq2_reanalysis_time_arrays_resampled"
    )
    output_directory = r"plots\absolute_delta_T_over_time_resampled"

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
        plot_differences_array_resampled(file_path, output_directory, upper_ylim=480)

# PLotting XQ2 profiles as array over time
if False: 
    resampled_dir = r"results\profiles_over_time_resampled"
    not_resampled_dir = r"results\profiles_over_time"
    output_dir="plots\profiles_over_time_resampled"
 # Define the file matching suffixes (all, ice, lake, tundra, water)
    file_suffixes = ['all_profiles', 'ice_profiles', 'lake_profiles', 'tundra_profiles', 'water_profiles']

    # Loop through the suffixes and process matching files
    for suffix in file_suffixes:
        # Construct file paths
        file_path_resampled = os.path.join(resampled_dir, f"resampled_array_xq2_T_{suffix}.csv")
        file_path_not_resampled = os.path.join(not_resampled_dir, f"array_xq2_T_{suffix}.csv")
        plot_profiles_array_resampled(file_path_resampled, file_path_not_resampled, output_dir=output_dir, upper_ylim=480)

# Plotting XQ2 Gradients over time
if True:
    # Define directories
    resampled_dir = r'results\gradients_over_time_resampled'
    not_resampled_dir = r"results\gradients_over_time"
    output_dir = r"plots\gradients_over_time_resampled"

    # Define the file matching suffixes (all, ice, lake, tundra, water)
    file_suffixes = ['all_profiles', 'ice_profiles', 'lake_profiles', 'tundra_profiles', 'water_profiles']

    # Loop through the suffixes and process matching files
    for suffix in file_suffixes:
        # Construct file paths
        file_path_resampled = os.path.join(resampled_dir, f"resampled_array_xq2_T_grad_{suffix}.csv")
        file_path_not_resampled = os.path.join(not_resampled_dir, f"array_xq2_T_grad_{suffix}.csv")

        plot_profiles_array_resampled(file_path_resampled, file_path_not_resampled, output_dir=output_dir, upper_ylim=480)
