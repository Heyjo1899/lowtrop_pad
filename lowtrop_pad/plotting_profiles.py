import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns


def split_and_concatenate(file):
    # Remove the file ending
    file = file.replace(".csv", "")
    # Find the first underscore
    first_underscore = file.find("_")
    # Find the last hyphen
    last_hyphen = file.rfind("-")
    # Get the parts before the first underscore and after the last hyphen
    part_before_underscore = file[:first_underscore]
    if part_before_underscore == "avg":
        part_before_underscore = "XQ2"
    part_after_hyphen = file[last_hyphen + 1 :]
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
                    label=f"{time} {split_and_concatenate(file)}",
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
                    label=f"{time} {split_and_concatenate(file)}",
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
                    label=f"{time} {split_and_concatenate(file)}",
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
        output_filename = f"Temperature_profiles_{date}.png"

    # If savefig is True, save the figure to the given path
    os.makedirs(output_path, exist_ok=True)
    if savefig:
        if output_path is not None:
            full_output_path = os.path.join(output_path, output_filename)
        else:  # Output in current directory
            full_output_path = output_filename
        plt.savefig(full_output_path, format="png", dpi=300)
        print(f"Figure saved as {full_output_path}")
    else:
        plt.show()


def plot_Asiaq_station_data(data_path, start_date, end_date):
    """
    Plot Asiaq station data for a given date range.
    Parameters:
    data_path (str): Path to the CSV file containing the data.
    start_date (str): Start date in the format "YYYY-MM-DD HH:MM:SS".
    end_date (str): End date in the format "YYYY-MM-DD HH:MM:SS".
    """
    # Parse the data into a pandas DataFrame
    df = pd.read_csv(data_path, sep=";", parse_dates=["DateTime"], dayfirst=True)

    # Filter the data by start and end dates
    mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
    filtered_df = df.loc[mask]

    # Define the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # First subplot - Wind Direction and Mean Wind Speed
    ax1.plot(
        filtered_df["DateTime"],
        filtered_df["VD(degrees 9m)"],
        "g*",
        label="Wind Direction (degrees)",
    )
    ax1b = ax1.twinx()
    ax1b.plot(
        filtered_df["DateTime"],
        filtered_df["VS_Mean(m/s 9m)"],
        "b-",
        label="Mean Wind Speed (m/s)",
    )

    # Second subplot - Temperature and Relative Humidity
    ax2.plot(
        filtered_df["DateTime"],
        filtered_df["Temp(oC 9m)"],
        "r-",
        label="Temperature (°C)",
    )
    ax2b = ax2.twinx()
    ax2b.plot(
        filtered_df["DateTime"],
        filtered_df["RH(% 9m)"],
        "c-",
        label="Relative Humidity (%)",
    )

    # Third subplot - Pressure and Radiation
    ax3.plot(
        filtered_df["DateTime"],
        filtered_df["Pressure(hPa 0m)"],
        "m-",
        label="Pressure (hPa)",
    )
    ax3b = ax3.twinx()
    ax3b.plot(
        filtered_df["DateTime"],
        filtered_df["RAD(W/m2 3m)"],
        "y-",
        label="Radiation (W/m2)",
    )

    # Set labels for the ax1 and ax1b
    ax1.set_ylabel("Wind Direction (degrees)", color="g")
    ax1b.set_ylabel("Mean Wind Speed (m/s)", color="b")

    # Set labels for ax2 and ax2b
    ax2.set_ylabel("Temperature (°C)", color="r")
    ax2b.set_ylabel("Relative Humidity (%)", color="c")

    # Set labels for ax3 and ax3b
    ax3.set_ylabel("Pressure (hPa)", color="m")
    ax3b.set_ylabel("Radiation (W/m2)", color="y")

    # Improve the x-axis Date formatting
    ax3.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y %H:%M"))

    # Rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Legends
    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")
    ax3.legend(loc="upper left")
    ax3b.legend(loc="upper right")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    plt.tight_layout()

    output_filename = f"Asiaq_station_data_{start_date[0:10]}_{end_date[0:10]}.png"
    os.makedirs("plots\\met_station_plots", exist_ok=True)
    plt.savefig(f"plots\\met_station_plots\\{output_filename}", dpi=300)


def plot_merged_and_resampled_profiles(
    file_path, x_varname, y_varname, file_ending, output_path, output_filename
):
    """
    Plots all profiles of day from the merged and interpolated files.
    file_path = path to file
    x_varname = Variable to plot on X-Axis.
    y_varname = Variable to plot on Y-Axis.
    file_ending = file ending of files to load.
    output_path = directory to store the profiles at.
    output_filename = name of the plot for storing.
    """
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different surfaces
    color_map = {
        "tundra": "brown",
        "ice": "magenta",
        "water": "darkblue",
        "lake": "green",
    }

    # Define line styles for different datasets
    line_styles = {
        "xq2_T": "-",
        "carra_T": "--",
        "era5_T": ":",
    }

    # Get a list of all files in the directory
    files = os.listdir(file_path)

    # Loop through the files and plot the data
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(file_ending):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Determine the color based on the file ending
                for key in color_map.keys():
                    if key in file:
                        color = color_map[key]
                        break

                # Plot each temperature profile with different line styles
                for temp_column in ["xq2_T", "carra_T", "era5_T"]:
                    ax.plot(
                        df[temp_column],
                        df[y_varname],
                        linestyle=line_styles[temp_column],
                        linewidth=1.5,
                        color=color,
                    )

    # Add grid
    ax.grid(True)

    # Create custom legend entries for line styles
    handles = []
    labels = []

    for key, style in line_styles.items():
        handles.append(
            plt.Line2D([0, 1], [0, 0], color="black", linestyle=style, linewidth=1.5)
        )
        labels.append(key)

    # Create custom legend entries for colors
    for key, color in color_map.items():
        handles.append(
            plt.Line2D([0, 1], [0, 0], color=color, linestyle="-", linewidth=1.5)
        )
        labels.append(key)

    ax.legend(handles, labels, title="Legend")

    # Add labels and title
    ax.set_xlabel(f"{x_varname} (°C)")
    ax.set_ylabel(f"{y_varname} (m)")
    ax.set_title(f"Temperature Profiles {root[-8:]}")
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, output_filename))


def plot_mean_differences_matrix(
    input_directory, output_directory_plots, output_directory_mean_diff
):
    """
    Loads the differences files including the mean difference.
    Creates 5x5 subplots for each surface and wind direction.
    Saves mean absolute and mean differences in a csv file.
    input_directory (str): Path to the directory containing the mean differences csv files.
    output_directory (str): Path to the output directory to save the plots.
    """

    # Create a dictionary for surfaces and wind directions
    wind_directions = ["north", "east", "south", "west", "all"]
    surfaces = ["tundra", "ice", "water", "lake", "all"]
    all_min, all_max = None, None  # Initialize at the beginning of the function

    # Initialize figures for "carra" and "era5"
    for dataset in ["carra", "era5"]:
        if dataset == "carra":
            MAD_df_carra = pd.DataFrame()  # initializing df to store MAD
        if dataset == "era5":
            MAD_df_era5 = pd.DataFrame()
        fig, axs = plt.subplots(5, 5, figsize=(20, 20))
        fig.suptitle(
            f"Differences for {dataset.upper()}", fontsize=18, fontweight="bold"
        )
        fig.supylabel("Altitude above gorund (m)", fontsize=17, fontweight="bold")
        fig.supxlabel(
            f"Temperature Difference XQ2-{dataset.upper()} (°C)",
            fontsize=17,
            fontweight="bold",
        )

        # Read all files in the input directory and plot the mean differences
        for file in os.listdir(input_directory):
            if file.endswith(".csv") and dataset in file:
                file_path = os.path.join(input_directory, file)
                try:
                    df = pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    continue

                # Get the surface and wind direction from the file name
                surface = file.split("_")[2]
                wind_direction = file.split("_")[-1].split(".")[0]

                # Ensure the surface and wind direction are valid
                if surface not in surfaces or wind_direction not in wind_directions:
                    continue  # Skip if invalid surface or wind direction

                # Get the index of the surface and wind direction
                surface_index = surfaces.index(surface)
                wind_direction_index = wind_directions.index(wind_direction)

                # Plot the mean differences in the right subplot
                axs[surface_index, wind_direction_index].plot(
                    df["mean"], df["alt_ag"], label="MD", linewidth=2.5, color="blue"
                )

                # Counter for the number of lines plotted
                line_count = 0

                # Plot all the single differences profiles with transparency
                for col in df.columns:
                    if col != "alt_ag" and col != "mean":
                        axs[surface_index, wind_direction_index].plot(
                            df[col],
                            df["alt_ag"],
                            label=col,
                            alpha=0.3,
                            linewidth=0.7,
                            color="blue",
                        )
                        line_count += 1  # Increment line count

                # Draw a transparent band between the maximum and minimum row-wise values
                axs[surface_index, wind_direction_index].fill_betweenx(
                    df["alt_ag"],
                    df.drop(columns=["alt_ag", "mean"]).max(axis=1),
                    df.drop(columns=["alt_ag", "mean"]).min(axis=1),
                    alpha=0.2,
                    color="blue",
                )

                # Calculate Mean Absolute Deviation (MAD)
                mean_absolute_deviation = (
                    df.drop(columns=["alt_ag", "mean"]).abs().mean().mean()
                )
                mean_deviation = df.drop(columns=["alt_ag", "mean"]).mean().mean()

                if dataset == "carra":
                    MAD_df_carra = pd.concat(
                        [
                            MAD_df_carra,
                            pd.DataFrame(
                                {
                                    "file": [file.split(".")[0]],
                                    "surface": [surface],
                                    "wind_direction": wind_direction,
                                    "MAD": [mean_absolute_deviation],
                                    "MD": [mean_deviation],
                                    "n": [line_count],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                if dataset == "era5":
                    MAD_df_era5 = pd.concat(
                        [
                            MAD_df_era5,
                            pd.DataFrame(
                                {
                                    "file": [file.split(".")[0]],
                                    "surface": [surface],
                                    "wind_direction": [wind_direction],
                                    "MAD": [mean_absolute_deviation],
                                    "MD": [mean_deviation],
                                    "n": [line_count],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                # Create a dictionary for the bounding box properties
                bbox_props = dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                    facecolor="lightgray",
                    alpha=1,
                )

                # Create a single annotation text
                annotation_text = (
                    f"MAD = {mean_absolute_deviation:.2f}\n" f"n = {line_count}"
                )

                # Annotate with a single box around the combined text
                axs[surface_index, wind_direction_index].annotate(
                    annotation_text,
                    xy=(0.02, 0.88),
                    xycoords="axes fraction",
                    fontsize=10,
                    fontweight="semibold",
                    ha="left",
                    bbox=bbox_props,  # Add the bounding box properties here
                )

                # Set the title of the subplot
                # axs[surface_index, wind_direction_index].set_title(f"{surface} - {wind_direction}")
                # Add a grid to the current subplot
                axs[surface_index, wind_direction_index].grid(True)

                if surface == "all" and wind_direction == "all":
                    all_min = df.drop(columns=["alt_ag", "mean"]).min().min()
                    all_max = df.drop(columns=["alt_ag", "mean"]).max().max()
        # Set consistent x and y limits for all plots
        if all_min is not None and all_max is not None:
            for ax in axs.flat:
                ax.set_xlim(all_min, all_max)
                ax.set_ylim(0, 500)

        # Add surface labels on the Y axis
        for i, surface in enumerate(surfaces):
            if surface == "all":
                axs[i, 0].set_ylabel("all surfaces", fontsize=15, fontweight="semibold")
            else:
                axs[i, 0].set_ylabel(surface, fontsize=15, fontweight="semibold")

        # Add wind direction labels above the top row
        for j, wind_direction in enumerate(wind_directions):
            if wind_direction == "all":
                axs[0, j].set_title("all wind sectors", fontsize=15, fontweight="bold")
            else:
                axs[0, j].set_title(wind_direction, fontsize=15, fontweight="bold")

        # Adjust layout to avoid overlap
        plt.tight_layout(rect=[0.01, 0.0, 1, 0.98])
        plt.grid(True)

        # create output directory if it does not exist
        os.makedirs(output_directory_plots, exist_ok=True)
        os.makedirs(output_directory_mean_diff, exist_ok=True)
        if "no_air_mass_change_profiles" in input_directory:
            plt.savefig(
                os.path.join(
                    output_directory_plots,
                    f"No_Air_Mass_Change_matrix_{dataset}_mean_differences.png",
                ),
                dpi=300,
            )
            # saving MAD values to a csv file
            if dataset == "carra":
                MAD_df_carra.to_csv(
                    os.path.join(
                        output_directory_mean_diff,
                        f"MD_MAD_No_Air_Mass_Change_{dataset}.csv",
                    ),
                    index=False,
                )
            if dataset == "era5":
                MAD_df_era5.to_csv(
                    os.path.join(
                        output_directory_mean_diff,
                        f"MD_MAD_No_Air_Mass_Change_{dataset}.csv",
                    ),
                    index=False,
                )
        else:
            plt.savefig(
                os.path.join(
                    output_directory_plots,
                    f"All_Profiles_matrix_{dataset}_mean_differences.png",
                ),
                dpi=300,
            )
            # saving MAD values to a csv file
            if dataset == "carra":
                MAD_df_carra.to_csv(
                    os.path.join(
                        output_directory_mean_diff, f"MD_MAD_all_profiles_{dataset}.csv"
                    ),
                    index=False,
                )
            if dataset == "era5":
                MAD_df_era5.to_csv(
                    os.path.join(
                        output_directory_mean_diff, f"MD_MAD_all_profiles_{dataset}.csv"
                    ),
                    index=False,
                )
        plt.close()


def plot_mean_differences(input_directory, output_directory, add_std=True):
    """
    Loads the mean differences or mean absolute differences and their standard deviations from the specified directory,
    creates and stores 3 types of plots:
    1. All profiles (mean_*.csv)
    2. All surfaces (columns ending with 'all.csv')
    3. All wind directions (columns starting with 'mean_all')

    Parameters:
    input_directory (str): Path to the directory containing the mean differences or mean absolute differences csv files.
    output_directory (str): Path to the output directory to save the plots.
    add_std (bool): If True, the ±1 standard deviation will be added to the plot as shaded regions.
    """

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Determine file names and plot title
    if "absolute" in input_directory:
        carra_file = os.path.join(
            input_directory, "mean_absolute_differences_carra.csv"
        )
        era5_file = os.path.join(input_directory, "mean_absolute_differences_era5.csv")
        plot_title = "Mean Absolute Temperature Differences (°C)"
        plot_filename = "mean_absolute_differences_plot"
    else:
        carra_file = os.path.join(input_directory, "mean_differences_carra.csv")
        era5_file = os.path.join(input_directory, "mean_differences_era5.csv")
        plot_title = "Mean Temperature Differences (°C)"
        plot_filename = "mean_differences_plot"

    if add_std:
        plot_filename += "_with_std.png"
    else:
        plot_filename += "_no_std.png"

    # Load data
    carra_df = pd.read_csv(carra_file)
    era5_df = pd.read_csv(era5_file)

    # Filter data to start from 2m above ground
    carra_df = carra_df[carra_df["alt_ag"] >= 2]
    era5_df = era5_df[era5_df["alt_ag"] >= 2]

    # Plot for Carra
    plt.figure(figsize=(6, 6))

    # Plot all profiles (mean_*.csv)
    for col in carra_df.columns:
        if col.startswith("mean_") and not col.startswith("std_"):
            first = col.split("_")[1]
            second = col.split("_")[2].split(".")[0]
            final_label = f"{first} {second}"
            plt.plot(carra_df[col], carra_df["alt_ag"], label=final_label)
            if add_std:
                std_col = col.replace("mean_", "std_")
                if std_col in carra_df.columns:
                    plt.fill_betweenx(
                        carra_df["alt_ag"],
                        carra_df[col] - carra_df[std_col],
                        carra_df[col] + carra_df[std_col],
                        alpha=0.2,
                    )

    # Plot settings for Carra (all profiles)
    plt.xlabel("Temperature Difference (°C)")
    plt.ylabel("Altitude Above Ground (m)")
    plt.title(f"XQ2-Carra - All Profiles {plot_title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, f"carra_{plot_filename}_all_profiles.png")
    )
    plt.close()

    # Plot for Era5
    plt.figure(figsize=(6, 6))

    # Plot all surfaces (columns ending with 'all.csv')
    for col in carra_df.columns:
        if col.endswith("_all.csv") and col.startswith("mean"):
            first = col.split("_")[1]
            second = col.split("_")[2].split(".")[0]
            final_label = f"{first} {second}"
            plt.plot(carra_df[col], carra_df["alt_ag"], label=final_label)
            if add_std:
                std_col = col.replace("mean_", "std_")
                if std_col in carra_df.columns:
                    plt.fill_betweenx(
                        carra_df["alt_ag"],
                        carra_df[col] - carra_df[std_col],
                        carra_df[col] + carra_df[std_col],
                        alpha=0.2,
                    )

    # Plot settings for Carra (all surfaces)
    plt.xlabel("Temperature Difference (°C)")
    plt.ylabel("Altitude Above Ground (m)")
    plt.title(f"XQ2-Carra - All Surfaces {plot_title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, f"carra_{plot_filename}_all_surfaces.png")
    )
    plt.close()

    # Plot for wind directions (columns starting with 'mean_all')
    plt.figure(figsize=(6, 6))

    for col in carra_df.columns:
        if col.startswith("mean_all"):
            first = col.split("_")[1]
            second = col.split("_")[2].split(".")[0]
            final_label = f"{first} {second}"
            plt.plot(carra_df[col], carra_df["alt_ag"], label=final_label)
            if add_std:
                std_col = col.replace("mean_", "std_")
                if std_col in carra_df.columns:
                    plt.fill_betweenx(
                        carra_df["alt_ag"],
                        carra_df[col] - carra_df[std_col],
                        carra_df[col] + carra_df[std_col],
                        alpha=0.2,
                    )

    # Plot settings for Carra (wind directions)
    plt.xlabel("Temperature Difference (°C)")
    plt.ylabel("Altitude Above Ground (m)")
    plt.title(f"XQ2-Carra - Wind Directions {plot_title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, f"carra_{plot_filename}_wind_directions.png")
    )
    plt.close()


def plot_xq2_reanalysis_correlation_matrix(
    input_directory, output_directory, variable_to_plot
):
    """
    Generates and saves a correlation matrix heatmap for xq2 vs reanalysis data.
    This function processes CSV files in the specified input directory, computes the mean correlation
    coefficient (r) and count (n) for different surfaces and wind direction combinations, and generates a heatmap
    plot with custom annotations. The heatmap is saved to the specified output directory.
    Parameters:
    input_directory (str): Path to the directory containing input CSV files.
    output_directory (str): Path to the directory where the output heatmap images will be saved.
    """

    # Define the surfaces and wind directions
    surfaces = ["tundra", "ice", "water", "lake", "all"]
    wind_directions = ["north", "east", "south", "west", "all"]

    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Initialize empty matrix for r values and counts
                matrix_r = pd.DataFrame(columns=wind_directions, index=surfaces)
                matrix_n = pd.DataFrame(columns=wind_directions, index=surfaces)

                # Fill the matrices with values from the dataframe
                for surface in surfaces:
                    for direction in wind_directions:
                        if surface == "all" and direction == "all":
                            # Mean and count for all surfaces and all wind directions
                            mean_r = df[variable_to_plot].mean()
                            count = df[variable_to_plot].count()
                        elif surface == "all":
                            # Mean and count for all surfaces for a specific wind direction
                            mean_r = df[df["wind_direction"] == direction][
                                variable_to_plot
                            ].mean()
                            count = df[df["wind_direction"] == direction][
                                variable_to_plot
                            ].count()
                        elif direction == "all":
                            # Mean and count for a specific surface across all wind directions
                            mean_r = df[df["surface"] == surface][
                                variable_to_plot
                            ].mean()
                            count = df[df["surface"] == surface][
                                variable_to_plot
                            ].count()
                        else:
                            # Mean and count for a specific surface and wind direction
                            filtered_df = df[
                                (df["surface"] == surface)
                                & (df["wind_direction"] == direction)
                            ]
                            mean_r = (
                                filtered_df[variable_to_plot].mean()
                                if not filtered_df.empty
                                else np.nan
                            )
                            count = filtered_df[variable_to_plot].count()

                        # Populate matrices
                        matrix_r.loc[surface, direction] = mean_r
                        matrix_n.loc[surface, direction] = count
                # Create custom annotations (n in the first row, r in the second row)
                annotations = (
                    "n = "
                    + matrix_n.astype(int).astype(str)
                    + "\n"
                    + "r = "
                    + matrix_r.astype("float").round(2).astype("string")
                )

                # Create a heatmap plot
                fig, ax = plt.subplots(figsize=(10, 8))

                # Define the heatmap, with a centered colormap and custom annotations
                sns.heatmap(
                    matrix_r.astype(float),
                    annot=annotations,
                    fmt="",
                    cmap="coolwarm",
                    linewidths=0.5,
                    cbar_kws={"label": variable_to_plot},
                    ax=ax,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    annot_kws={"color": "black"},
                )
                # Add title and labels
                title = "Correlation XQ2 " + (
                    "CARRA" if "carra" in file.lower() else "ERA5"
                )
                ax.set_title(title)
                ax.set_xlabel("Wind Direction")
                ax.set_ylabel("Surface")

                # Save the heatmap to the output directory
                if "all_profiles" in input_directory:
                    output_file = os.path.join(
                        output_directory,
                        f"matrix_all_profiles_{title.lower().replace(' ', '_')}.png",
                    )
                if "no_air_mass_change" in input_directory:
                    output_file = os.path.join(
                        output_directory,
                        f"matrix_no_air_mass_change_profiles_{title.lower().replace(' ', '_')}.png",
                    )
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()


def plot_xq2_reanalysis_differences_matrix(
    input_directory, output_directory, variable_to_plot
):
    """
    Generates and saves a differences matrix heatmap for xq2 vs reanalysis data.
    This function processes CSV files in the specified input directory arranges the data in a matrix format,
    and generates a heatmap plot with custom annotations. The heatmap is saved to the specified output directory.

    Parameters:
    input_directory (str): Path to the directory containing input CSV files.
    output_directory (str): Path to the directory where the output heatmap images will be saved.
    variable_to_plot (str): The variable to plot, e.g., "MAD" or "MD".
    """

    # Define the surfaces and wind directions
    surfaces = ["tundra", "ice", "water", "lake", "all"]
    wind_directions = ["north", "east", "south", "west", "all"]

    os.makedirs(output_directory, exist_ok=True)

    # Loop through all files in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Initialize empty matrices for MAD, MD, and n values
                matrix_mad = pd.DataFrame(columns=wind_directions, index=surfaces)
                matrix_md = pd.DataFrame(columns=wind_directions, index=surfaces)
                matrix_n = pd.DataFrame(columns=wind_directions, index=surfaces)

                # Fill the matrices with values from the dataframe
                for surface in surfaces:
                    for direction in wind_directions:
                        # Get MAD, MD, and n for a specific surface and wind direction
                        filtered_df = df[
                            (df["surface"] == surface)
                            & (df["wind_direction"] == direction)
                        ]
                        mad = (
                            filtered_df["MAD"].iloc[0]
                            if not filtered_df.empty
                            else np.nan
                        )
                        md = (
                            filtered_df["MD"].iloc[0]
                            if not filtered_df.empty
                            else np.nan
                        )
                        n = (
                            filtered_df["n"].iloc[0]
                            if not filtered_df.empty
                            else np.nan
                        )

                        # Populate matrices
                        matrix_mad.loc[surface, direction] = mad
                        matrix_md.loc[surface, direction] = md
                        matrix_n.loc[surface, direction] = n

                if variable_to_plot == "MAD":
                    matrix_to_plot = matrix_mad
                elif variable_to_plot == "MD":
                    matrix_to_plot = matrix_md
                # Create custom annotations (n in the first row, MAD/MD in the second row)
                annotations = (
                    "n = "
                    + matrix_n.astype(str)
                    + "\n"
                    + f"{variable_to_plot} = "
                    + matrix_to_plot.astype(float).round(2).astype(str)
                    + "\n"
                )
                # Create a heatmap plot
                fig, ax = plt.subplots(figsize=(10, 8))

                sns.heatmap(
                    matrix_to_plot.astype(float),
                    annot=annotations,
                    fmt="",
                    cmap="viridis",
                    linewidths=0.5,
                    cbar_kws={"label": variable_to_plot},
                    ax=ax,
                    vmin=0,  # Set minimum value of the color scale
                    annot_kws={"color": "black"},
                )

                # Add title and labels
                title = f"{variable_to_plot} Matrix " + (
                    "CARRA" if "carra" in file.lower() else "ERA5"
                )
                ax.set_title(title)
                ax.set_xlabel("Wind Direction")
                ax.set_ylabel("Surface")

                if "all_profiles" in input_directory:
                    output_file = os.path.join(
                        output_directory,
                        f"matrix_all_profiles_{title.lower().replace(' ', '_')}.png",
                    )
                if "no_air_mass_change" in input_directory:
                    output_file = os.path.join(
                        output_directory,
                        f"matrix_no_air_mass_change_profiles_{title.lower().replace(' ', '_')}.png",
                    )
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()


def plot_differences_array(file_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert the time columns to datetime
    df.columns = ["alt_ag"] + [
        pd.to_datetime(col, infer_datetime_format=True) for col in df.columns[1:]
    ]

    # Set the alt_ag as the index (Y-axis)
    df.set_index("alt_ag", inplace=True)

    # Convert the DataFrame to a NumPy array for imshow
    data = df.values

    # Find the maximum altitude where data is not NaN
    max_valid_alt_ag = df.index[df.notna().any(axis=1)].max()

    # Create a figure and axis for the heatmap
    plt.figure(figsize=(12, 8))

    # Plot the heatmap using imshow with flipped Y-axis and fixed color scale
    cax = plt.imshow(
        data,
        aspect="auto",
        cmap="RdYlBu",
        interpolation="nearest",
        origin="lower",
        vmin=-5,
        vmax=5,
    )

    # Set labels and title
    plt.title(os.path.basename(file_path))
    plt.xlabel("Time")
    plt.ylabel("Altitude Above Ground (m))")

    # Set X and Y ticks
    num_cols = len(df.columns) - 1
    num_rows = len(df.index)

    # Reduce the number of Y-ticks
    y_ticks = np.arange(0, num_rows, max(1, num_rows // 10))  # Show at most 10 labels
    plt.yticks(ticks=y_ticks, labels=df.index[y_ticks])

    # Reduce the number of X-ticks for better readability
    x_tick_indices = np.arange(
        0, num_cols, max(1, num_cols // 20)
    )  # Show at most 20 labels
    plt.xticks(
        ticks=x_tick_indices,
        labels=[df.columns[i + 1].strftime("%d.%m.%Y %H:%M") for i in x_tick_indices],
        rotation=45,
        ha="right",
    )

    # Set Y-axis limit to the maximum valid altitude
    plt.ylim(
        0, np.where(df.index <= max_valid_alt_ag)[0][-1] + 1
    )  # Setting limit based on index position

    # Add a colorbar with the fixed range
    plt.colorbar(cax, label="Temperature Difference (°C)")

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Save the plot to the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(
        output_dir, os.path.basename(file_path).replace(".csv", ".png")
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_differences_array_resampled(file_path, output_dir, upper_ylim=None):
    # Load the data
    df = pd.read_csv(file_path)

    # Convert the time columns to datetime
    df.set_index("alt_ag", inplace=True)
    df.columns = pd.to_datetime(df.columns)

    # Find the maximum altitude where data is not NaN
    max_valid_alt_ag = df.index[df.notna().any(axis=1)].max()

    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the heatmap using imshow with the Y-axis and fixed color scale
    if file_path.split("\\")[-1].startswith("abs"):
        cax = ax.imshow(
            df.values,
            aspect="auto",
            cmap="Reds",
            interpolation="spline16",
            origin="lower",
            vmin=0,
            vmax=6,
        )
        c_label = "Absolute Temperature Difference (°C)"
    else:
        cax = ax.imshow(
            df.values,
            aspect="auto",
            cmap="RdYlBu",
            interpolation="spline16",
            origin="lower",
            vmin=-5,
            vmax=5,
        )
        c_label = "Temperature Difference (°C)"
    # Add contour lines
    levels = np.linspace(-5, 5, 11)  # Define the levels for contour lines
    ax.contour(
        df.values,
        levels=levels,
        colors="black",
        linewidths=0.5,
        alpha=0.7,
        extent=[0, df.shape[1], 0, df.shape[0]],
    )

    # Set the labels and title
    ax.set_title(os.path.basename(file_path))
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude Above Ground (m)")

    # Configure x-ticks
    num_dates = len(df.columns)
    tick_interval = max(
        1, num_dates // 14
    )  # Adjust this divisor to control the number of ticks

    # Set x-ticks at appropriate intervals
    ax.set_xticks(np.arange(0, num_dates, tick_interval))

    # Set x-tick labels
    ax.set_xticklabels(
        [t.strftime("%Y-%m-%d") for t in df.columns[::tick_interval]],
        rotation=45,
        ha="right",
    )

    # Set Y-axis limit to the maximum valid altitude
    if upper_ylim is None:
        ax.set_ylim(0, np.where(df.index <= max_valid_alt_ag)[0][-1] + 1)
    if upper_ylim is not None:
        ax.set_ylim(0, upper_ylim)

    # Add a colorbar with a fixed range
    plt.colorbar(cax, ax=ax, label=c_label)

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    if upper_ylim is not None:
        output_dir = os.path.join(f"{output_dir}_{upper_ylim}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(
        output_dir, os.path.basename(file_path).replace(".csv", ".png")
    )
    print(os.path.basename(file_path))
    print(f"Saving plot to {output_file_path}")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()


def plot_profiles_array_resampled(
    file_path_resampled, file_path_not_resampled, output_dir, upper_ylim=None
):
    # Load the resampled data
    df_resampled = pd.read_csv(file_path_resampled)
    df_resampled.set_index("alt_ag", inplace=True)
    df_resampled.columns = pd.to_datetime(df_resampled.columns)

    # Load the non-resampled data to get the time events
    df_not_resampled = pd.read_csv(file_path_not_resampled)

    # Cleaning to just have time columns
    time_columns = df_not_resampled.drop("alt_ag", axis=1).columns

    # Convert valid time columns to datetime
    time_events = pd.to_datetime(time_columns, errors="coerce")

    # Drop any invalid datetime entries
    # time_events = time_events.dropna()

    # Find the maximum altitude where data is not NaN
    max_valid_alt_ag = df_resampled.index[df_resampled.notna().any(axis=1)].max()

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    if "gradient" in file_path_resampled:
        cax = ax.imshow(
            df_resampled.values,
            aspect="auto",
            cmap="plasma",
            interpolation="spline16",
            origin="lower",
            vmin=-0.04,
            vmax=0.04,
        )
        c_label = "Temperature Gradient per Meter (°C)"

        # Generate contour levels based on data range
        levels = np.linspace(df_resampled.min().min(), df_resampled.max().max(), 40)

        # Add contour lines
        ax.contour(
            df_resampled.values,
            levels=levels,
            colors="black",
            linewidths=0.5,
            alpha=0.7,
            extent=[0, df_resampled.shape[1], 0, df_resampled.shape[0]],
        )
    else:
        cax = ax.imshow(
            df_resampled.values,
            aspect="auto",
            cmap="RdYlBu_r",
            interpolation="spline16",
            origin="lower",
        )
        c_label = "Temperature (°C)"

        # Add contour lines
        ax.contour(
            df_resampled.values,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.7,
            extent=[0, df_resampled.shape[1], 0, df_resampled.shape[0]],
        )

    # Set the labels and title
    ax.set_title(os.path.basename(file_path_resampled))
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude Above Ground (m)")

    # Configure x-ticks
    num_dates = len(df_resampled.columns)
    tick_interval = max(
        1, num_dates // 14
    )  # Adjust this divisor to control the number of ticks
    ax.set_xticks(np.arange(0, num_dates, tick_interval))
    ax.set_xticklabels(
        [t.strftime("%Y-%m-%d") for t in df_resampled.columns[::tick_interval]],
        rotation=45,
        ha="right",
    )

    # Add vertical grey transparent bands for each valid event time
    time_start = df_resampled.columns[0]
    time_end = df_resampled.columns[-1]

    # Convert time to a numeric scale (e.g., total seconds from the start)
    total_seconds = (df_resampled.columns - time_start).total_seconds()

    for event_time in time_events:
        if time_start <= event_time <= time_end:
            # Convert the event time to total seconds
            event_seconds = (event_time - time_start).total_seconds()

            # Find the closest indices in the resampled time data
            closest_index = np.searchsorted(total_seconds, event_seconds)

            # Add a grey band at the corresponding position
            ax.axvspan(
                closest_index - 0.04, closest_index + 0.04, color="dimgrey", alpha=1
            )  # Slight transparency

    # Add a legend for the grey bands
    grey_band = Patch(color="dimgrey", alpha=1, label="Measured Profile")
    ax.legend(handles=[grey_band], loc="upper left")

    # Set Y-axis limit to the maximum valid altitude
    if upper_ylim is None:
        ax.set_ylim(0, np.where(df_resampled.index <= max_valid_alt_ag)[0][-1] + 1)
    else:
        ax.set_ylim(0, upper_ylim)

    # Add colorbar
    plt.colorbar(cax, ax=ax, label=c_label)

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Create the output directory if it doesn't exist
    if upper_ylim is not None:
        output_dir = os.path.join(f"{output_dir}_{upper_ylim}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot
    output_file_path = os.path.join(
        output_dir, os.path.basename(file_path_resampled).replace(".csv", ".png")
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
