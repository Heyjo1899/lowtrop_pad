import os
import pandas as pd
import numpy as np
from datetime import datetime

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

def load_and_reduce_profile_top(directory_path, red_n=3):
    """
    Load and process profiles from CSV files in a specified directory.

    Parameters:
    directory_path (str): Path to the directory containing the CSV files.
    red_n (int): Number of values to drop at the highest elevation due to downdraft effect.

    Returns:
    dict: Dictionary containing the processed DataFrames.
    """
    # Dictionary to store the DataFrames
    xq2_all_raw = {}

    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Load the file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Add height above ground
                df["alt_ag"] = df["alt"] - df["alt"][0]

                # Add potential temperature
                df["T_pot"] = (df["T"] + 273.15) * ((1000 / df["Pressure"]) ** 0.286)

                # Drop the highest `red_n` values from df where unnatural warming effect occurs
                df_red = df.iloc[:-red_n]

                # Store the processed DataFrame in the dictionary
                xq2_all_raw[file] = df_red

    return xq2_all_raw


def average_profiles_by_vertical_bins(
    profiles,
    bin_size,
    output_directory="data/xq2/averaged_profiles_bin",
    custom_bins=False,
    bin1=3,
    bin2=5,
    bin3=10,
):
    """
    Average profiles by vertical bins and save the results to the specified output directory.

    Parameters:
    profiles (dict): Dictionary containing the DataFrames of profiles.
    bin_size (int): Size of the vertical bins for averaging.
    output_directory (str): Path to the directory where the averaged profiles will be saved.
    custom_bins (bool): If True, use custom bin sizes for the first three bins.
    bin1 (int): Size of the first bin (only used if custom_bins is True).
    bin2 (int): Size of the second bin (only used if custom_bins is True).
    bin3 (int): Size of the third bin (only used if custom_bins is True).
    """
    # Dictionary to store the averaged DataFrames
    xq2_avg = {}

    # Walk through all profiles
    for key in profiles:
        # Extract one profile
        df = profiles[key]

        if custom_bins:
            # Custom bins for the first three bins
            bins = [0, bin1, bin1 + bin2, bin1 + bin2 + bin3]  # First three bins
            remaining_bins = np.arange(
                bins[-1] + bin_size, df["alt_ag"].iloc[-1], bin_size
            )  # Remaining bins
            bins.extend(remaining_bins)
            alt_ag_bins = np.array(bins)
        else:
            # Default bins
            alt_ag_bins = np.arange(0, df["alt_ag"].iloc[-1], bin_size)

        # Initializing list to store avg values
        avg = []

        # Calculate average bins for every profile
        for i in range(1, len(alt_ag_bins), 1):
            # Parse every profile into bins
            df_bin = df[
                (df["alt_ag"] >= alt_ag_bins[i - 1]) & (df["alt_ag"] <= alt_ag_bins[i])
            ]

            # Check if df_bin is not empty, if empty skip the bin
            if not df_bin.empty:
                # Calculate mean values for every variable
                mean_P = df_bin["Pressure"].mean()
                mean_T = df_bin["T"].mean()
                mean_H = df_bin["Humidity"].mean()
                mean_HT = df_bin["Humidity Temp"].mean()
                mean_lon = df_bin["lon"].mean()
                mean_lat = df_bin["lat"].mean()
                mean_alt = df_bin["alt"].mean()
                mean_sat = df_bin["Sat"].mean()
                mean_alt_ag = df_bin["alt_ag"].mean()
                mean_T_pot = df_bin["T_pot"].mean()

                # Take the time of the first row of the bin
                time = df_bin["Datetime"].iloc[0]

                # Append mean values to list
                avg.append(
                    (
                        mean_P,
                        mean_T,
                        mean_T_pot,
                        mean_H,
                        mean_HT,
                        mean_lon,
                        mean_lat,
                        mean_alt,
                        alt_ag_bins[i],
                        mean_alt_ag,
                        mean_sat,
                        time,
                    )
                )
            else:
                # If bin is empty, continue to the next bin
                continue

            # When the last bin is taken, just averaging the values above
            if i + 1 == len(alt_ag_bins):
                df_bin = df[(df["alt_ag"] > alt_ag_bins[i])]

                mean_P = df_bin["Pressure"].mean()
                mean_T = df_bin["T"].mean()
                mean_H = df_bin["Humidity"].mean()
                mean_HT = df_bin["Humidity Temp"].mean()
                mean_lon = df_bin["lon"].mean()
                mean_lat = df_bin["lat"].mean()
                mean_alt = df_bin["alt"].mean()
                mean_sat = df_bin["Sat"].mean()
                mean_alt_ag = df_bin["alt_ag"].mean()
                mean_T_pot = df_bin["T_pot"].mean()

                # For the time taking the first value of the bin
                time = df_bin["Datetime"].iloc[0]

                avg.append(
                    (
                        mean_P,
                        mean_T,
                        mean_T_pot,
                        mean_H,
                        mean_HT,
                        mean_lon,
                        mean_lat,
                        mean_alt,
                        alt_ag_bins[i],
                        mean_alt_ag,
                        mean_sat,
                        time,
                    )
                )

        avg_df = pd.DataFrame(
            avg,
            columns=[
                "P",
                "T",
                "T_pot",
                "H",
                "HT",
                "lon",
                "lat",
                "alt",
                "alt_bin",
                "alt_ag",
                "sat",
                "time",
            ],
        )
        xq2_avg[f"avg_{key}"] = avg_df

    # Create main output directory with bin size
    if custom_bins:
        output_directory = f"{output_directory}_custom_{bin1}_{bin2}_{bin3}_{bin_size}"
    else:
        output_directory = f"{output_directory}_{bin_size}"
    os.makedirs(output_directory, exist_ok=True)

    # First read out the dates from labels in the directory and create directory to store data in
    dates = []
    for key in xq2_avg:
        dates.append(key[4:12])

    # Convert dates to DataFrame and remove duplicates
    df_dates = pd.DataFrame(dates, columns=["date"])
    df_dates = df_dates.drop_duplicates().reset_index(drop=True)

    # Generate directories to store the data
    for date in df_dates["date"]:
        os.makedirs(os.path.join(output_directory, date), exist_ok=True)

    # Save the data to the generated directories
    for key in xq2_avg:
        file_name = key
        df = xq2_avg[key]
        directory = key[4:12]
        df.to_csv(os.path.join(output_directory, directory, file_name), index=False)


def extract_profile_times_and_coords(folder_path):
    """
    Extracts the first value in the 'time', 'lat', and 'lon' columns from each profile in the given folder
    and returns them in a DataFrame.

    Parameters:
    folder_path (str): Path to the folder containing the profiles.

    Returns:
    pd.DataFrame: DataFrame containing the extracted times, latitudes, and longitudes of the profiles.
    """
    # List to store the extracted data
    data_list = []

    # Walk through all directories and files
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Load the profile data
                df_profile = pd.read_csv(file_path)

                # Extract the first value in the 'time' column
                first_time = df_profile["time"].iloc[0]

                # Convert the first time to datetime format
                first_time_dt = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S")

                # Extract the first value in the 'lat' and 'lon' columns
                first_lat = df_profile["lat"].iloc[0]
                first_lon = df_profile["lon"].iloc[0]

                # Append to the list
                data_list.append(
                    {
                        "time": first_time_dt,
                        "latitude": first_lat,
                        "longitude": first_lon,
                        "file_name": file_name,
                    }
                )

    # Create a DataFrame from the extracted data
    df_times_profiles = pd.DataFrame(
        data_list, columns=["time", "latitude", "longitude", "file_name"]
    )

    return df_times_profiles

def save_coordinates_from_profiles(profile_path1, profile_path2, profile_path3, output_path):
    """
    Read out the coordinates from the profile directories and store them as a csv.
    profile_path (str): Path to the folder of profiles. Can take 3 different paths (XQ2, CARRA, ERA5).
    output_path (str): Path to the output folder.
    """

    # List to store the extracted data
    data_list = []

    # Walk through all directories and files of profile path 1
    for root, dirs, files in os.walk(profile_path1):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Load the profile data
                df_profile = pd.read_csv(file_path)

                # Extract the first value in the 'time' column
                first_time = df_profile["time"].iloc[0]

                # Convert the first time to datetime format
                first_time_dt = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S")

                # Extract the first value in the 'lat' and 'lon' columns
                first_lat = df_profile["lat"].iloc[0]
                first_lon = df_profile["lon"].iloc[0]

                # taking just the data type of the file name
                short_file_name = split_and_concatenate(file_name)
                first_space = short_file_name.find(' ')
                data_type = short_file_name[:first_space]

                # Append to the list
                data_list.append(
                    {
                        "time": first_time_dt,
                        "latitude": first_lat,
                        "longitude": first_lon,
                        "data_type": data_type,
                        "file_name": file_name,
                    }
                )
    # Walk through all directories and files of profile path 2
    for root, dirs, files in os.walk(profile_path2):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Load the profile data
                df_profile = pd.read_csv(file_path)

                # Extract the first value in the 'time' column
                first_time = df_profile["time"].iloc[0]

                # Convert the first time to datetime format
                first_time_dt = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S")

                # Extract the first value in the 'lat' and 'lon' columns
                first_lat = df_profile["lat"].iloc[0]
                first_lon = df_profile["lon"].iloc[0]

                # taking just the data type of the file name
                short_file_name = split_and_concatenate(file_name)
                first_space = short_file_name.find(' ')
                data_type = short_file_name[:first_space]

                # Append to the list
                data_list.append(
                    {
                        "time": first_time_dt,
                        "latitude": first_lat,
                        "longitude": first_lon,
                        "data_type": data_type,
                        "file_name": file_name,
                    }
                )
    # Walk through all directories and files of profile path 3
    for root, dirs, files in os.walk(profile_path3):
        for file_name in files:
            if file_name.endswith(".csv"):
                file_path = os.path.join(root, file_name)

                # Load the profile data
                df_profile = pd.read_csv(file_path)

                # Extract the first value in the 'time' column
                first_time = df_profile["time"].iloc[0]

                # Convert the first time to datetime format
                first_time_dt = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S")

                # Extract the first value in the 'lat' and 'lon' columns
                first_lat = df_profile["lat"].iloc[0]
                first_lon = df_profile["lon"].iloc[0]

                # taking just the data type of the file name
                short_file_name = split_and_concatenate(file_name)
                first_space = short_file_name.find(' ')
                data_type = short_file_name[:first_space]

                # Append to the list
                data_list.append(
                    {
                        "time": first_time_dt,
                        "latitude": first_lat,
                        "longitude": first_lon,
                        "data_type": data_type,
                        "file_name": file_name,
                    }
                )
    # Create a DataFrame from the extracted data
    df_coordinates = pd.DataFrame(
        data_list, columns=["time", "latitude", "longitude", "data_type", "file_name"]
    )
    # taking the date from current file name for storing folder
    date = file_name[file_name.find('-') - 8 : file_name.find('-')]

    os.makedirs(output_path, exist_ok=True)
    df_coordinates.to_csv(f'{output_path}//coor-{date}.csv')
        