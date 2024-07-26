import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def split_and_concatenate(file):
    # Remove the file ending
    file = file.replace(".csv", "")
    # Find the first underscore
    first_underscore = file.find('_')
    # Find the last hyphen
    last_hyphen = file.rfind('-')
    # Get the parts before the first underscore and after the last hyphen
    part_before_underscore = file[:first_underscore]
    if part_before_underscore == "avg":
        part_before_underscore = "XQ2"
    part_after_hyphen = file[last_hyphen + 1:]
    # Concatenate the first and last parts
    result = f"{part_before_underscore} {part_after_hyphen}"
    return result

def plot_raw_and_smoothed_profiles_of_day(
    date, directory_path_1, directory_path_2, varname="T", file_ending=".csv"
):
    """
    Plot profiles from the specified directories for a given variable.

    Parameters:
    date (str): Date of which profiles are plotted.
    directory_path_1 (str): Path to first directory containing profiles.
    directory_path_2 (str): Path to second directory containing profiles.
    varname (str): Variable name to plot on X.
    file_ending = (str): Ending of files to plot. When ".csv" all profiles of day will be plotted.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Count total files to determine the number of colors needed
    total_files = sum(
        [
            len(files)
            for r, d, files in os.walk(directory_path_1)
            if any(file.endswith(file_ending) for file in files)
        ]
    )

    # Get a colormap with the same number of colors as the total number of files
    colors = plt.get_cmap("tab10", total_files)

    file_counter_1 = 0  # Initialize a counter for the files processed

    for root, dirs, files in os.walk(directory_path_1):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                parent_directory = os.path.dirname(directory_path_1)
                second_last_directory_name = os.path.basename(parent_directory)
                ax.plot(
                    df[varname],
                    df["alt"],
                    label=f"{second_last_directory_name} {file}",
                    linestyle="solid",
                    linewidth=0.9,
                    color=colors(file_counter_1),
                )
                file_counter_1 += 1

    file_counter_2 = 0
    for root, dirs, files in os.walk(directory_path_2):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                parent_directory = os.path.dirname(directory_path_2)
                second_last_directory_name = os.path.basename(parent_directory)
                ax.plot(
                    df[varname],
                    df["alt"],
                    label=f"{second_last_directory_name} {file}",
                    linestyle="--",
                    color=colors(file_counter_2),
                )
                file_counter_2 += 1

    ax.grid()
    ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_xlabel(varname, fontsize=12)
    ax.set_title(date, fontsize=14)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_xq2_vs_reanalysis_profiles_of_day(
    date,
    xq_2_path,
    carra_path,
    era5_path,
    x_varname="T",
    y_varname="alt_ag",
    file_ending=".csv",
    savefig=True,
    output_path=None,
    output_filename=None,
):
    """
    Plot profiles from the specified directories for a given variable.

    Parameters:
    date (str): Date of which profiles are plotted.
    xq_2_path (str): Path to xq2 directory containing profiles.
    carra_path (str): Path to extracted CARRA profiles.
    era5_path (str): Path to extracted ERA5 profiles.
    x_varname (str): Variable name to plot on X.
    y_varname (str): Variable name to plot on Y.
    file_ending (str): Ending of files to plot.
    savefig (bool): Whether to save the figure (default: False).
    output_path (str): Directory path where the figure should be saved.
    output_filename (str): Name of the output file (default: None, will use date for filename).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Count total files to determine the number of colors needed
    # Get a list of all files in the directory
    files = os.listdir(xq_2_path)

    # Count the number of .csv files
    file_count = sum(1 for file in files if file.endswith(".csv"))

    # Get a colormap with the same number of colors as the total number of files
    colors = plt.get_cmap("turbo", file_count)
    file_counter_1 = 0  # Initialize a counter for the files processed

    for root, dirs, files in os.walk(xq_2_path):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                time = df["time"][0][11:19]
                ax.plot(
                    df[x_varname],
                    df[y_varname],
                    label = f"{time} {split_and_concatenate(file)}",
                    linestyle="dotted",
                    linewidth=1.5,
                    color=colors(file_counter_1),
                )
                file_counter_1 += 1

    file_counter_2 = 0
    for root, dirs, files in os.walk(carra_path):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                time = df["time"][0][11:19]
                ax.plot(
                    df[x_varname],
                    df[y_varname],
                    label = f"{time} {split_and_concatenate(file)}",
                    linestyle="--",
                    linewidth=1.4,
                    color=colors(file_counter_2),
                )
                file_counter_2 += 1

    file_counter_3 = 0
    for root, dirs, files in os.walk(era5_path):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                time = df["time"][0][11:19]
                ax.plot(
                    df[x_varname],
                    df[y_varname],
                    label = f"{time} {split_and_concatenate(file)}",
                    linestyle="solid",
                    linewidth=1.3,
                    color=colors(file_counter_3),
                )
                file_counter_3 += 1

    ax.grid()
    ax.set_ylabel("Altitude above ground [m]", fontsize=12)
    ax.set_xlabel("T [°C]", fontsize=12)
    ax.set_title(date, fontsize=14)
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    # Set output filename
    if output_filename is None:
        output_filename = f"{file_ending}_{date}.png"

    # If savefig is True, save the figure to the given path
    os.makedirs(output_path, exist_ok=True)
    if savefig:
        if output_path is not None:
            full_output_path = os.path.join(output_path, output_filename)
        else:  # Output in current directory
            full_output_path = output_filename
        plt.savefig(full_output_path, format='png', dpi=300)
        print(f"Figure saved as {full_output_path}")
    else:
        plt.show()
    

def plot_Asiaq_station_data(data_string, start_date, end_date):
    
    # Parse the data into a pandas DataFrame
    df = pd.read_csv(data_string, sep=';', parse_dates=['DateTime'], dayfirst=True)
    
    # Filter the data by start and end dates
    mask = (df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)
    filtered_df = df.loc[mask]

    # Define the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # First subplot - Wind Direction and Mean Wind Speed
    ax1.plot(filtered_df['DateTime'], filtered_df['VD(degrees 9m)'], 'g*', label='Wind Direction (degrees)')
    ax1b = ax1.twinx()
    ax1b.plot(filtered_df['DateTime'], filtered_df['VS_Mean(m/s 9m)'], 'b-', label='Mean Wind Speed (m/s)')

    # Second subplot - Temperature and Relative Humidity
    ax2.plot(filtered_df['DateTime'], filtered_df['Temp(oC 9m)'], 'r-', label='Temperature (°C)')
    ax2b = ax2.twinx()
    ax2b.plot(filtered_df['DateTime'], filtered_df['RH(% 9m)'], 'c-', label='Relative Humidity (%)')

    # Third subplot - Pressure and Radiation
    ax3.plot(filtered_df['DateTime'], filtered_df['Pressure(hPa 0m)'], 'm-', label='Pressure (hPa)')
    ax3b = ax3.twinx()
    ax3b.plot(filtered_df['DateTime'], filtered_df['RAD(W/m2 3m)'], 'y-', label='Radiation (W/m2)')
    
    # Set labels for the ax1 and ax1b
    ax1.set_ylabel('Wind Direction (degrees)', color='g')
    ax1b.set_ylabel('Mean Wind Speed (m/s)', color='b')

    # Set labels for ax2 and ax2b
    ax2.set_ylabel('Temperature (°C)', color='r')
    ax2b.set_ylabel('Relative Humidity (%)', color='c')

    # Set labels for ax3 and ax3b
    ax3.set_ylabel('Pressure (hPa)', color='m')
    ax3b.set_ylabel('Radiation (W/m2)', color='y')

    # Improve the x-axis Date formatting
    ax3.xaxis.set_major_formatter(DateFormatter('%d-%m-%Y %H:%M'))

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Legends
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax2b.legend(loc='upper right')
    ax3.legend(loc='upper left')
    ax3b.legend(loc='upper right')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    plt.tight_layout()
    
    output_filename = f"Asiaq_station_data_{start_date[0:10]}_{end_date[0:10]}.png"
    os.makedirs("plots\\met_station_plots", exist_ok=True)
    plt.savefig(f"plots\\met_station_plots\\{output_filename}", dpi=300)