import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np


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
        output_filename = f"{file_ending}_{date}.png"

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
    """ """
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


def plot_mean_differences(input_directory, output_directory):
    """
    Loads the mean differences or mean absolute differences from the specified directory,
    creates and stores plots for altitude vs temperature difference.

    Parameters:
    input_directory (str): Path to the directory containing the mean differences or mean absolute differences csv files.
    output_directory (str): Path to the output directory to save the plots.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Determine file names based on whether we're dealing with absolute differences or not
    if "absolute" in input_directory:
        carra_file = os.path.join(
            input_directory, "mean_absolute_differences_carra.csv"
        )
        era5_file = os.path.join(input_directory, "mean_absolute_differences_era5.csv")
        plot_title = "Mean Absolute Temperature Differences (°C)"
        plot_filename = "mean_absolute_differences_plot.png"
    else:
        carra_file = os.path.join(input_directory, "mean_differences_carra.csv")
        era5_file = os.path.join(input_directory, "mean_differences_era5.csv")
        plot_title = "Mean Temperature Differences (°C)"
        plot_filename = "mean_differences_plot.png"

    # Load the data
    carra_df = pd.read_csv(carra_file)
    era5_df = pd.read_csv(era5_file)

    # Reduce Data to start with 2m above ground to avoid lacking xq2 data near surface
    carra_df = carra_df[carra_df["alt_ag"] >= 2]
    era5_df = era5_df[era5_df["alt_ag"] >= 2]

    # Create the plot for Carra
    plt.figure(figsize=(6, 6))
    plt.plot(
        carra_df["mean_all_profiles"],
        carra_df["alt_ag"],
        label="All Profiles",
        color="blue",
    )
    plt.plot(carra_df["mean_tundra"], carra_df["alt_ag"], label="Tundra", color="green")
    plt.plot(carra_df["mean_water"], carra_df["alt_ag"], label="Water", color="red")
    plt.plot(carra_df["mean_ice"], carra_df["alt_ag"], label="Ice", color="purple")
    plt.plot(carra_df["mean_lake"], carra_df["alt_ag"], label="Lake", color="orange")

    # Plot settings for Carra
    plt.xlabel("Temperature Difference (°C)")
    plt.ylabel("Altitude Above Ground (m)")
    plt.title(f"XQ2-Carra - {plot_title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    # Save the Carra plot
    plt.savefig(os.path.join(output_directory, f"carra_{plot_filename}"))
    plt.close()

    # Create the plot for Era5
    plt.figure(figsize=(6, 6))
    plt.plot(
        era5_df["mean_all_profiles"],
        era5_df["alt_ag"],
        label="All Profiles",
        color="blue",
    )
    plt.plot(era5_df["mean_tundra"], era5_df["alt_ag"], label="Tundra", color="green")
    plt.plot(era5_df["mean_water"], era5_df["alt_ag"], label="Water", color="red")
    plt.plot(era5_df["mean_ice"], era5_df["alt_ag"], label="Ice", color="purple")
    plt.plot(era5_df["mean_lake"], era5_df["alt_ag"], label="Lake", color="orange")

    # Plot settings for Era5
    plt.xlabel("Temperature Difference (°C)")
    plt.ylabel("Altitude Above Ground (m)")
    plt.title(f"XQ2-Era5 - {plot_title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    # Save the Era5 plot
    plt.savefig(os.path.join(output_directory, f"era5_{plot_filename}"))
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
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
