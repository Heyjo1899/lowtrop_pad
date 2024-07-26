import os
from lowtrop_pad.plotting_profiles import (
    plot_raw_and_smoothed_profiles_of_day,
    plot_xq2_vs_reanalysis_profiles_of_day,
    plot_Asiaq_station_data
)
from lowtrop_pad.plotting_synoptic import plot_era5_synoptic

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

# Plotting profiles of a specific Day
if False:
    date = '20230815'
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
        output_path='plots\\plots_of_day\\',
        output_filename=None
    )

# Plotting all profiles of each day over LOOP
if False:
    base_xq2_path = "data/xq2/averaged_profiles_custom_3_5_10_20"
    base_carra_path = "data/reanalysis/CARRA_extracted_profiles"
    base_era5_path = "data/reanalysis/ERA5_extracted_profiles"
    output_base_path = "plots\\plots_of_day"

    # Get the list of dates from the xq2 directory
    dates = next(os.walk(base_xq2_path), (None, None, []))[1]  # Lists directories inside base_xq2_path

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
            output_filename=None
        )

if False:
    start_date = '2023-08-01 10:00:00'
    end_date = '2023-08-01 16:00:00'
    # Plotting Asiaq station data over Field Period (Atmospheric Monitoring Hut) 
    plot_Asiaq_station_data('data\\met_stations\\Asiaq_met_VRS.csv', start_date, end_date)

# plotting synoptic single plot
if False:
    date = '20230801'
    time = '12'
    path_to_file = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic.nc"
    variable_to_plot = "z"
    level = 500
    output_path = "plots\\synoptic"
    output_filename = f'{variable_to_plot}{level}_{time}_{date}.png'

    plot_era5_synoptic(date, time, path_to_file, variable_to_plot,level, output_path, output_filename)

# plotting synoptic all days + timesteps plots
if True: 
    base_path = "data/reanalysis/ERA5_extracted_profiles"  # Path to the folder containing date subfolders
    nc_file_path = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_synoptic.nc"  
    variable_to_plot = "z"  
    level = 500  
    output_path = "plots/synoptic"  

    time_steps = ['06', '12', '18']

    dates = [
        folder
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder))
    ]

    # Iterate through each date and time step
    for date in dates:
        for time in time_steps:
            print(time)
            # Define the output file path and name
            output_filename = f'{variable_to_plot}_{level}_{date}_{time}.png'
           
            # Plot and save the figure
            plot_era5_synoptic(
                date=date,
                time=time,
                path_to_file=nc_file_path,
                variable_to_plot=variable_to_plot,
                level=level,
                output_path=output_path,
                output_filename=output_filename
            )