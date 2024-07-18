import os
import pandas as pd
import matplotlib.pyplot as plt


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
    file_ending = (str): Ending of files to plot. When ".csv" all profiles of day will be plotted.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Count total files to determine the number of colors needed
    # Get a list of all files in the directory
    files = os.listdir(xq_2_path)

    # Count the number of .csv files
    file_count = sum(1 for file in files if file.endswith(".csv"))

    # Get a colormap with the same number of colors as the total number of files
    colors = plt.get_cmap("tab20b", file_count)
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
                    label=f"{time} xq2_{file}",
                    linestyle="dashdot",
                    linewidth=1.1,
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
                    label=f"{time} {file}",
                    linestyle="--",
                    linewidth=1.1,
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
                    label=f"{time} {file}",
                    linestyle="solid",
                    linewidth=1.1,
                    color=colors(file_counter_3),
                )
                file_counter_3 += 1

    ax.grid()
    ax.set_ylabel("Altitude above ground [m]", fontsize=12)
    ax.set_xlabel("T [°C]", fontsize=12)
    ax.set_title(date, fontsize=14)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
