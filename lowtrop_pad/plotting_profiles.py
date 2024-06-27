import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_raw_and_smoothed_profiles_of_day(
    date, directory_path_smoothed, directory_path_raw, varname="T"
):
    """
    Plot profiles from the specified directories for a given variable.

    Parameters:
    date (str): Date string used in the plot title.
    directory_path_smoothed (str): Path to the directory containing the averaged profiles.
    directory_path_raw (str): Path to the directory containing the original profiles.
    varname (str): Variable name to plot on X.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot averaged profiles
    for root, dirs, files in os.walk(directory_path_smoothed):
        for file in files:
            if file.endswith("3-1-2-tundra.csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                ax.plot(df[varname], df["alt"], label=f"Averaged {file}")

    # Plot original profiles
    for root, dirs, files in os.walk(directory_path_raw):
        for file in files:
            if file.endswith("3-1-2-tundra.csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                ax.plot(df[varname], df["alt"], label=f"Original {file}")

    ax.grid()
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel(varname)
    ax.set_title(date)
    ax.legend()
    plt.show()
