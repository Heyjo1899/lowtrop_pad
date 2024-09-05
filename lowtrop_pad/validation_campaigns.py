import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error




def read_xq2_data(xq2_file):
    df = pd.read_csv(xq2_file, parse_dates=True)

    # Extract only the XQ data (the rest is zero)
    df = df[
        [
            "XQ-iMet-XQ Pressure",
            "XQ-iMet-XQ Air Temperature",
            "XQ-iMet-XQ Humidity",
            "XQ-iMet-XQ Humidity Temp",
            "XQ-iMet-XQ Date",
            "XQ-iMet-XQ Time",
            "XQ-iMet-XQ Longitude",
            "XQ-iMet-XQ Latitude",
            "XQ-iMet-XQ Altitude",
            "XQ-iMet-XQ Sat Count",
        ]
    ]

    # Rename the columns (strip 'XQ-iMet-XQ ')
    df = df.rename(columns={k: k[11:] for k in df.keys()})

    # Combine Date and Time into a single Datetime column
    df["Datetime"] = pd.to_datetime(df["Date"] + "-" + df["Time"])

    # Drop the original Date and Time columns
    df = df.drop(columns=["Date", "Time"])

    # Rename columns
    df.rename(
        columns={
            "Longitude": "lon",
            "Latitude": "lat",
            "Altitude": "alt",
            "Air Temperature": "t",
            "Sat Count": "Sat",
            "Datetime": "time",
            "Humidity": "h",
        },
        inplace=True,
    )

    return df


def load_hl_data(directory):
    """
    Load all humilog files in the subdirectories of the given directory and return them in a dictionary.
    """
    data = {}
    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Humilog files:
            if file.endswith(".ASC"):
                # Get the folder name containing the file
                file_name = os.path.basename(file)

                # Extract the date part from the path
                date_part = root.split("\\")[-1]

                # Extract the desired substring '0817'
                hl_date = date_part[4:8]
                # Extract the last 5 numbers from the folder name
                hl_number = file_name[9:14]

                # Generate the individual name
                individual_name = f"hl_{hl_date}_{hl_number}"

                # Load the file
                file_path = os.path.join(root, file)
                # print(file_path)
                # Load the file
                hl_df = pd.read_table(file_path, encoding="latin-1", skiprows=4)

                # Strip the white spaces in the headers
                hl_df.columns = hl_df.columns.str.strip()
                # Rename to simpler headers
                hl_df.rename(
                    columns={
                        "Datum": "date",
                        "Zeit": "time_raw",
                        "1.Temperatur    [°C]": "t",
                        "2.rel.Feuchte    [%]": "h",
                    },
                    inplace=True,
                )

                hl_df = hl_df.iloc[:, :-1]

                hl_df["h"] = hl_df["h"].str.replace(",", ".").astype(float)
                hl_df["t"] = hl_df["t"].str.replace(",", ".").astype(float)

                # Convert 'time_raw' to datetime and reformat to 'YY-MM-DD HH:MM:SS'
                hl_df["time_raw"] = pd.to_datetime(
                    hl_df["date"] + " " + hl_df["time_raw"],
                    format="%d.%m.%y %H:%M:%S",
                    dayfirst=True,
                )

                # Drop the original 'date' column
                hl_df.drop(["date"], axis=1, inplace=True)

                # Adjust time by subtracting 2 hours
                hl_df["time"] = pd.to_datetime(hl_df["time_raw"]) - datetime.timedelta(
                    hours=2
                )

                # Store the processed DataFrame in the dictionary
                data[individual_name] = hl_df
    return data


def load_save_mast_xq2_ascents(
    directory, output_folder, thresh=2, start_buffer=4, end_buffer=2
):
    """
    Loops over .csv files in directory, separates them into individual ascents, resamples by 0.5m,
    interpolates linearly and saves each ascent as a CSV file.
    Start and end times of each experiment are extracted and returned as a DataFrame.

    Parameters:
        directory (str): Directory containing the .csv files.
        output_folder (str): Folder where the separated ascent CSV files will be saved.
        thresh (float): Altitude difference threshold for detecting ascents.
        start_buffer (int): Number of points before ascent detection to include.
        end_buffer (int): Number of points after ascent detection to include.
    """

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    times = pd.DataFrame(columns=["number", "start", "end"])

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                # Extract the relevant digits from the file name for individual identification
                digits = file[15:21]
                individual_name = f"{digits}"

                # Load the CSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Rename columns for consistency
                df.rename(
                    columns={
                        "T": "t",
                        "Sat Count": "Sat",
                        "Datetime": "time",
                        "Humidity": "h",
                    },
                    inplace=True,
                )

                # Extract start and end time
                start_time = df["time"].iloc[0]
                end_time = df["time"].iloc[-1]

                # Create a DataFrame for the current file's start and end times
                current_time_df = pd.DataFrame(
                    {
                        "number": [individual_name],
                        "start": [start_time],
                        "end": [end_time],
                    }
                )

                # Concatenate the current time data with the main DataFrame
                times = pd.concat([times, current_time_df], ignore_index=True)

                # Ensure no duplicate timestamps
                df = df.groupby("time").head(1).reset_index(drop=True)

                # Add potential temperature
                df["T_pot"] = (df["t"] + 273.15) * ((1000 / df["Pressure"]) ** 0.286)

                # Detect ascents
                idx = []
                for i in range(1, len(df["alt"])):
                    if df["alt"][i] - df["alt"][i - 1] >= thresh:
                        idx.append(i)

                # Identify boundaries of each ascent
                lower = [0]
                upper = []
                for i in range(1, len(idx) - 1):
                    if idx[i + 1] - idx[i] == 1 and idx[i] - idx[i - 1] != 1:
                        lower.append(idx[i] - start_buffer)
                    if idx[i] - idx[i - 1] == 1 and idx[i + 1] - idx[i] != 1:
                        upper.append(idx[i] + end_buffer)
                upper.append(idx[-1])

                # Extract and store individual ascents
                for i in range(len(lower)):
                    df_single = df.iloc[lower[i] : upper[i]].reset_index(drop=True)
                    # print('single', df_single)
                    # df_single['alt_ag'] = df_single['alt'] - df_single['alt'][0]
                    # Ensure the DataFrame is sorted by 'alt_ag'
                    df_single = df_single.sort_values(by="alt")
                    df_single["alt_ag"] = df_single["alt"] - df_single["alt"][0]

                    # Remove duplicates in 'alt_ag' before reindexing
                    df_single = df_single.drop_duplicates(subset=["alt_ag"])

                    # Convert 'alt_ag' to numeric
                    df_single["alt_ag"] = pd.to_numeric(df_single["alt_ag"])

                    # Calculate start and stop value
                    start = math.floor(df_single["alt_ag"].min())
                    stop = math.ceil(df_single["alt_ag"].max()) + 1

                    # Set index to 'alt_ag'
                    df_single.set_index("alt_ag", inplace=True)

                    # Convert object dtype columns to suitable numeric types before interpolation
                    df_single = df_single.infer_objects()

                    # Generate resampled altitudes
                    alt_resampled = np.arange(start, stop, 0.5)

                    # Interpolate data
                    df_resampled = (
                        df_single.reindex(df_single.index.union(alt_resampled))
                        .sort_index()
                        .interpolate(method="linear")
                        .loc[alt_resampled]
                    )
                    df_resampled.reset_index(inplace=True)
                    # Add the time of the first row to the resampled DataFrame
                    df_resampled["time"] = df_single["time"].iloc[0]

                    # Construct the filename for this ascent
                    ascent_filename = f"{individual_name}_ascent_{i+1}.csv"
                    ascent_path = os.path.join(output_folder, ascent_filename)

                    # Save the individual ascent to a CSV file
                    df_resampled.to_csv(ascent_path, index=False)
    times.to_csv("data//mast_experiment//single_experiments//times_mast_campaigns.csv", index=False)


def load_save_mast_xq2_descents(
    directory, output_folder, thresh=2, start_buffer=0, end_buffer=30
):
    """
    Loops over .csv files in directory, separates them into individual descents, resamples by 0.5m,
    interpolates linearly and saves each descent as a CSV file.
    Start and end times of each experiment are extracted and returned as a DataFrame.

    Parameters:
        directory (str): Directory containing the .csv files.
        output_folder (str): Folder where the separated descent CSV files will be saved.
        thresh (float): Altitude difference threshold for detecting descents.
        start_buffer (int): Number of points before descent detection to include.
        end_buffer (int): Number of points after descent detection to include.
    """

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    times = pd.DataFrame(columns=["number", "start", "end"])

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                # Extract the relevant digits from the file name for individual identification
                digits = file[15:21]
                individual_name = f"{digits}"

                # Load the CSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Rename columns for consistency
                df.rename(
                    columns={
                        "T": "t",
                        "Sat Count": "Sat",
                        "Datetime": "time",
                        "Humidity": "h",
                    },
                    inplace=True,
                )

                # Extract start and end time
                start_time = df["time"].iloc[0]
                end_time = df["time"].iloc[-1]

                # Create a DataFrame for the current file's start and end times
                current_time_df = pd.DataFrame(
                    {
                        "number": [individual_name],
                        "start": [start_time],
                        "end": [end_time],
                    }
                )

                # Concatenate the current time data with the main DataFrame
                times = pd.concat([times, current_time_df], ignore_index=True)

                # Ensure no duplicate timestamps
                df = df.groupby("time").head(1).reset_index(drop=True)

                # Add potential temperature
                df["T_pot"] = (df["t"] + 273.15) * ((1000 / df["Pressure"]) ** 0.286)

                # Detect descents
                idx = []
                for i in range(1, len(df["alt"])):
                    if df["alt"][i - 1] - df["alt"][i] >= thresh:
                        idx.append(i)

                # Identify boundaries of each descent
                lower = [idx[0]]
                upper = []
                for i in range(1,len(idx)-1,1):
                
                    # lower boundary
                    if idx[i+1] - idx[i] == 1 and idx[i] - idx[i-1] != 1:
                        lower.append(idx[i]-start_buffer)
                            
                    # upper boundary
                    if idx[i] - idx[i-1] == 1 and idx[i+1] - idx[i] != 1:
                        upper.append(idx[i]+end_buffer)
                upper.append(idx[-1]+end_buffer)    

                # Extract and store individual descents
                for i in range(len(lower)):
                    df_single = df.iloc[lower[i] : upper[i]].reset_index(drop=True)
                    # Ensure the DataFrame is sorted by 'alt_ag'
                    df_single = df_single.sort_values(by="alt", ascending=False)

                    df_single["alt_ag"] = df_single["alt"] - df_single["alt"].min()
                    # Remove duplicates in 'alt_ag' before reindexing
                    df_single = df_single.drop_duplicates(subset=["alt_ag"])

                    # Convert 'alt_ag' to numeric
                    df_single["alt_ag"] = pd.to_numeric(df_single["alt_ag"])

                    # Calculate start and stop value
                    start = math.floor(df_single["alt_ag"].min())
                    stop = math.ceil(df_single["alt_ag"].max()) + 1

                    # Set index to 'alt_ag'
                    df_single.set_index("alt_ag", inplace=True)

                    # Convert object dtype columns to suitable numeric types before interpolation
                    df_single = df_single.infer_objects()

                    # Generate resampled altitudes
                    alt_resampled = np.arange(start, stop, 0.5)

                    # Interpolate data
                    df_resampled = (
                        df_single.reindex(df_single.index.union(alt_resampled))
                        .sort_index()
                        .interpolate(method="linear")
                        .loc[alt_resampled]
                    )
                    df_resampled.reset_index(inplace=True)
                    # Add the time of the first row to the resampled DataFrame
                    df_resampled["time"] = df_single["time"].iloc[0]

                    # Construct the filename for this descent
                    descent_filename = f"{individual_name}_descent_{i+1}.csv"
                    descent_path = os.path.join(output_folder, descent_filename)

                    # Save the individual descent to a CSV file
                    df_resampled.to_csv(descent_path, index=False)
    times.to_csv("data//mast_experiment//single_experiments//times_mast_campaigns.csv", index=False)

def load_mast_data(directory):
    mast_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".MTO"):
                file_path = os.path.join(root, file)
                parts = file.split("_")

                # Extract the first part and the last part that contains the version number
                prefix = parts[0]  # 'ICOS'
                version = parts[-2]  # '2.0' (second to last part)

                # Combine them to form 'ICOS_2.0'
                individual_name = f"{prefix}_{version}"

                df = pd.read_csv(file_path, skiprows=35, delimiter=";")

                # Combine time columns into a single 'time' column
                df["time"] = pd.to_datetime(
                    df[["Year", "Month", "Day", "Hour", "Minute"]]
                )

                # Drop the old time-related columns
                df.drop(
                    columns=["Year", "Month", "Day", "Hour", "Minute"], inplace=True
                )

                # Rename columns
                df.rename(columns={"AT": "t", "RH": "h", "WS": "ws"}, inplace=True)
                df_reduced = df[
                    (df["time"] >= "2023-07-27 00:00:00")
                    & (df["time"] <= "2023-08-22 00:00:00")
                ]
                # Store the processed dataframe in the dictionary
                mast_data[individual_name] = df_reduced

    return mast_data


def resample_save_hl_mast_data(
    hl_data, mast_data, times_path, hl_output_folder, mast_output_folder
):
    '''
    Resample and interpolate Humilog and Mast data to match the time periods and time frequency of the xq2 data.
    Save the resampled data to CSV files in the specified output directories.

    Parameters:
        hl_data (dict): A dictionary containing Humilog data DataFrames.
        mast_data (dict): A dictionary containing Mast data DataFrames.
        times_path (str): The path to the CSV file containing the time periods.
        hl_output_folder (str): The folder to save the resampled Humilog data.
        mast_output_folder (str): The folder to save the resampled Mast data.
    '''
    os.makedirs(hl_output_folder, exist_ok=True)
    os.makedirs(mast_output_folder, exist_ok=True)
    times = pd.read_csv(times_path)
    times["start"] = pd.to_datetime(times["start"])
    times["end"] = pd.to_datetime(times["end"])

    # Iterate through each time period in the `times` dataframe
    for _, row in times.iterrows():
        start_time = row["start"]
        end_time = row["end"]
        time_id = row["number"]

        # Subset and save HL data
        for key, df in hl_data.items():
            hl_subset = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
            if not hl_subset.empty:
                if key == "hl_0817_22176" and start_time != times["start"][4]:
                    hl_subset["alt_ag"] = 8
                    hl_filename = f"{time_id}_{key}.csv"
                    hl_subset.to_csv(
                        os.path.join(hl_output_folder, hl_filename), index=False
                    )
                if key == "hl_0821_22176" and start_time == times["start"][4]:
                    hl_subset["alt_ag"] = 4.5
                    hl_filename = f"{time_id}_{key}.csv"
                    hl_subset.to_csv(
                        os.path.join(hl_output_folder, hl_filename), index=False
                    )
                if key == "hl_0817_22731" and start_time != times["start"][4]:
                    hl_subset["alt_ag"] = 1.85
                    hl_filename = f"{time_id}_{key}.csv"
                    hl_subset.to_csv(
                        os.path.join(hl_output_folder, hl_filename), index=False
                    )
                if key == "hl_0821_22731" and start_time == times["start"][4]:
                    hl_subset["alt_ag"] = 0.1
                    hl_filename = f"{time_id}_{key}.csv"
                    hl_subset.to_csv(
                        os.path.join(hl_output_folder, hl_filename), index=False
                    )

            else:
                print(f"No data for {key} in time range {start_time} to {end_time}")

        # Subset and resample Mast data
        for key, df in mast_data.items():
            if key != "ICOS_2.0":
                print("timezs", start_time, end_time)
                alt = float(key.split("_")[-1])
                mast_resampled = (
                    df.set_index("time")
                    .resample("1T")
                    .interpolate(method="time")
                    .reset_index()
                )
                mast_subset = mast_resampled[
                    (mast_resampled["time"] >= start_time)
                    & (mast_resampled["time"] <= end_time)
                ]
                mast_subset["alt_ag"] = alt
                if not mast_subset.empty:
                    mast_filename = f"{time_id}_{key}.csv"
                    mast_subset.to_csv(
                        os.path.join(mast_output_folder, mast_filename), index=False
                    )
                else:
                    print(f"No data for {key} in time range {start_time} to {end_time}")


def load_data_mast_plotting(mode):
    """
    Load CSV data from specified directories and organize it by mast number.

    Parameters:
        base_directory (str): The base working directory.

    Returns:
        dict: A dictionary with mast numbers as keys and lists of (DataFrame, plot_style, label) tuples as values.
    """
    mast_directory = os.path.join(
        "data",
        "mast_experiment",
        "single_experiments",
        "mast_data_single_exp",
    )
    xq2_directory = os.path.join(
        "data",
        "mast_experiment",
        "single_experiments",
        f"single_{mode}",
    )
    hl_directory = os.path.join(
        "data",
        "mast_experiment",
        "single_experiments",
        "humilog_single_exp",
    )

    # List of directories to loop through
    directories = [mast_directory, xq2_directory, hl_directory]

    # Prepare a dictionary to store data for each mast number
    data_by_mast = {str(i): [] for i in range(1, 6)}

    # Loop through the directories and load the CSV files
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and filename.split("_")[0] in data_by_mast:
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                print(f"Loaded: {filename}")
                # Extract mast number (first part of filename)
                mast_number = filename.split("_")[0]

                # Determine plot style and label based on the third part of the filename
                if "ascent" in filename.split("_")[2]:
                    plot_style = "-"
                    suffix = filename.split("_")[3].split(".")[0]
                    label = f"XQ {suffix}"
                elif "descent" in filename.split("_")[2]:
                    plot_style = "-"
                    suffix = filename.split("_")[3].split(".")[0]
                    label = f"XQ {suffix}"
                elif "ICOS" in filename.split("_")[2]:
                    plot_style = "*"
                    prefix = filename.split("_")[2]
                    suffix = filename.split("_")[3][0:2]
                    label = f"{prefix} {suffix}m"
                elif "hl" in filename.split("_")[2]:
                    plot_style = "o"
                    suffix = df["alt_ag"][0]
                    label = f"Humilog {suffix}m"
                else:
                    plot_style = "."  # Default plot style if not matched
                    label = "Other"

                # Store the data, plot style, and label
                data_by_mast[mast_number].append((df, plot_style, label))

    return data_by_mast


def plot_mast_data(mast_number, data_list, variable, plot_filename):
    """
    Plot a specific variable against altitude for a given mast number and save the plot.

    Parameters:
        mast_number (str): The mast number to plot data for.
        data_list (list): A list of tuples containing DataFrame, plot style, and label.
        variable (str): The variable to plot (e.g., 't' for temperature, 'h' for humidity).
        plot_filename (str): The filename for saving the plot.
        base_directory (str): The base directory for saving the plots.
    """
    plt.figure(figsize=(10, 8))

    for df, plot_style, label in data_list:
        plt.plot(df[variable], df["alt_ag"], plot_style, label=label)

    if variable == "t":
        plt.title(f"Temperature (°C) Campaign {mast_number}")
        plot_directory = os.path.join("plots", "mast_experiment", "t_mast")
    elif variable == "h":
        plt.title(f"Humidity (%) Campaign {mast_number}")
        plot_directory = os.path.join("plots", "mast_experiment", "h_mast")

    plt.xlabel(variable)
    plt.ylabel("Altitude above ground (m)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(os.path.join(plot_directory, plot_filename))
    plt.close()

def mast_data_df(mode):
    '''
    Loading ICOS, Humilog and xq2 data and building a Dataframe for all altitudes to compare with xq2 data.
    
    Parameters:
        None
    '''
    # Define directories
    mast_directory = os.path.join("data", "mast_experiment", "single_experiments", "mast_data_single_exp")
    xq2_directory = os.path.join("data", "mast_experiment", "single_experiments", f"single_{mode}")
    hl_directory = os.path.join("data", "mast_experiment", "single_experiments", "humilog_single_exp")
    
    # List of starting numbers for mast files
    starting_numbers = [1, 2, 3, 4, 5]
    
    # Initialize a dictionary to store results for each starting number
    dfs = {}
    
    # Process each starting number
    for number in starting_numbers:
        # Initialize the DataFrame to store the results for the current batch
        result_df = pd.DataFrame(columns=['alt_ag', 'time', 'xq2_t', 'xq2_h', 'station_t', 'station_h', 'station_id', 'Ascent', 'wind_speed_20m'])
        
        # Process XQ2 (single ascent) files
        for filename in os.listdir(xq2_directory):
            if filename.endswith(".csv") and filename.startswith(f"{number}_mast"):
                file_path = os.path.join(xq2_directory, filename)
                df = pd.read_csv(file_path)
                
                # Convert the 'time' column to datetime format
                df['time'] = pd.to_datetime(df['time'])
                
                # Get the time from the first row
                time = df["time"].iloc[0]
                
                # Adjust the altitude extraction if the file is a "5_mast" file
                if filename.startswith("5_mast"):
                    altitudes = [0, 4.5, 20, 85]  # Extract altitudes 0 and 4.5 for 5_mast
                else:
                    altitudes = [2, 8, 20, 85]
                
                # Filter rows by the specific altitudes and store in the result DataFrame
                for alt_ag in altitudes:
                    row = df[df['alt_ag'] == alt_ag]
                    if not row.empty:
                        new_row = {
                            'alt_ag': alt_ag,
                            'time': time,
                            'xq2_t': row['t'].values[0],  # Add 't' value to the xq2_t column
                            'xq2_h': row['h'].values[0],  # Add 'h' value to the xq2_h column
                            'station_t': np.nan,  # Placeholder for station t data
                            'station_h': np.nan,  # Placeholder for station h data
                            'station_id': np.nan,  # Placeholder for station ID
                            'Ascent': filename
                        }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Process mast data and Humilog files for the current batch
        for filename in os.listdir(mast_directory):
            if filename.endswith(".csv") and filename.startswith(f"{number}_mast"):
                file_path = os.path.join(mast_directory, filename)
                df = pd.read_csv(file_path)
                print(f'Processing: {filename} for number {number}')
                
                # Convert the 'time' column to datetime format
                df['time'] = pd.to_datetime(df['time'])
                
                if "20" in filename:
                    alt_ag = 20
                    # add the windspeed of the files containing '20' to all rows with the same closest time
                    for index, row in result_df.iterrows():
                        closest_time_idx = (df['time'] - row['time']).abs().idxmin()
                        closest_time_row = df.loc[closest_time_idx]
                        result_df.at[index, 'wind_speed_20m'] = closest_time_row['ws']
                    
                elif "85" in filename:
                    alt_ag = 85
                else:
                    continue
                
                # Find the closest time for each row in result_df
                for index, row in result_df[result_df['alt_ag'] == alt_ag].iterrows():
                    closest_time_idx = (df['time'] - row['time']).abs().idxmin()
                    closest_time_row = df.loc[closest_time_idx]
                    result_df.at[index, 'station_t'] = closest_time_row['t']  # Add 't' value to the station_t column
                    result_df.at[index, 'station_h'] = closest_time_row['h']  # Add 'h' value to the station_h column
                    result_df.at[index, 'station_id'] = 'ICOS'
                    
        for filename in os.listdir(hl_directory):
            if filename.endswith(".csv") and filename.startswith(f"{number}_mast"):
                file_path = os.path.join(hl_directory, filename)
                df = pd.read_csv(file_path)
                print(f'Processing: {filename} for number {number}')
                
                # Convert the 'time' column to datetime format
                df['time'] = pd.to_datetime(df['time'])
                
                if "22176" in filename:
                    if filename.startswith("5_mast"):
                        alt_ag = 4.5 
                    else: 
                        alt_ag = 8
                elif "22731" in filename:
                    if filename.startswith("5_mast"):
                        alt_ag = 0
                    else:
                        alt_ag = 2
                else:
                    continue
                
                # Find the closest time for each row in result_df
                for index, row in result_df[result_df['alt_ag'] == alt_ag].iterrows():
                    closest_time_idx = (df['time'] - row['time']).abs().idxmin()
                    closest_time_row = df.loc[closest_time_idx]
                    result_df.at[index, 'station_t'] = closest_time_row['t']  # Add 't' value to the station_t column
                    result_df.at[index, 'station_h'] = closest_time_row['h']  # Add 'h' value to the station_h column
                    result_df.at[index, 'station_id'] = 'Humilog'
        
        # Store the DataFrame for the current starting number
        dfs[number] = result_df
    
    # Merge all DataFrames into one
    final_df = pd.concat(dfs.values(), ignore_index=True)
    # calculate the differences
    final_df['diff_t'] = final_df['xq2_t'] - final_df['station_t']
    final_df['abs_diff_t'] = final_df['diff_t'].abs()
    final_df['diff_h'] = final_df['xq2_h'] - final_df['station_h']
    final_df['abs_diff_h'] = final_df['diff_h'].abs()
    final_df.to_csv(f"data//mast_experiment//single_experiments//mast_data_df_{mode}.csv", index=False)

def calculate_differences(mast_data_df_path, mode, output_dir= "data//mast_experiment//results"):
    '''
    Calculate differences between XQ2 and station data for temperature and humidity.

    Parameters:
        mast_data_df_path (str): Path to the CSV file containing the final_df DataFrame.
    '''
    os.makedirs(output_dir, exist_ok=True)
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['alt_ag', 'mean_diff_t', 'mean_abs_diff_t', 'mean_diff_h', 'mean_abs_diff_h'])
    
    # Get unique altitude levels
    final_df = pd.read_csv(mast_data_df_path)
    final_df['time'] = pd.to_datetime(final_df['time'])

    alt_ag_levels = final_df['alt_ag'].unique()
    
    # Iterate over each altitude level
    for alt_ag in alt_ag_levels:
        # Filter the dataframe for the current altitude level
        df_filtered = final_df[final_df['alt_ag'] == alt_ag]
        
        # Calculate differences for 't' and 'h'
        diff_t = df_filtered['xq2_t'] - df_filtered['station_t']
        diff_h = df_filtered['xq2_h'] - df_filtered['station_h']
        
        # Calculate mean differences and mean absolute differences
        mean_diff_t = diff_t.mean()
        mean_abs_diff_t = diff_t.abs().mean()
        mean_diff_h = diff_h.mean()
        mean_abs_diff_h = diff_h.abs().mean()
        
        # Store the results in the results_df DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'alt_ag': [alt_ag],
            'mean_diff_t': [mean_diff_t],
            'mean_abs_diff_t': [mean_abs_diff_t],
            'mean_diff_h': [mean_diff_h],
            'mean_abs_diff_h': [mean_abs_diff_h]
        })], ignore_index=True)
    
    # Calculate overall means without regard to alt_ag
    overall_diff_t = final_df['xq2_t'] - final_df['station_t']
    overall_diff_h = final_df['xq2_h'] - final_df['station_h']
    
    overall_mean_diff_t = overall_diff_t.mean()
    overall_mean_abs_diff_t = overall_diff_t.abs().mean()
    overall_mean_diff_h = overall_diff_h.mean()
    overall_mean_abs_diff_h = overall_diff_h.abs().mean()
    
    # Append the overall means as an additional row
    overall_row = pd.DataFrame({
        'alt_ag': ['Overall'],
        'mean_diff_t': [overall_mean_diff_t],
        'mean_abs_diff_t': [overall_mean_abs_diff_t],
        'mean_diff_h': [overall_mean_diff_h],
        'mean_abs_diff_h': [overall_mean_abs_diff_h]
    })
    
    results_df = pd.concat([results_df, overall_row], ignore_index=True)
    results_df.to_csv(os.path.join(output_dir, f"differences_{mode}.csv"), index=False)
    results_df.to_html(os.path.join(output_dir, f"differences_{mode}.html"), index=False)

# Assuming 'final_df' is the dataframe containing the data
def calculate_metrics(mast_data_df_path, mode, output_dir= "data//mast_experiment//results"):
    '''
    Takes DF and calculates Pearson correlation and RMSE for temperature and humidity data.

    Parameters:
        final_df (pd.DataFrame): DataFrame containing 'xq2_t', 'station_t', 'xq2_h', 'station_h', and 'alt_ag' columns.
    '''
    os.makedirs(output_dir, exist_ok=True)
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['alt_ag', 'pearson_corr_t', 'rmse_t', 'pearson_corr_h', 'rmse_h'])
    
    # Get unique altitude levels
    final_df = pd.read_csv(mast_data_df_path)
    final_df['time'] = pd.to_datetime(final_df['time'])
    alt_ag_levels = final_df['alt_ag'].unique()
    
    # Iterate over each altitude level
    for alt_ag in alt_ag_levels:
        # Filter the dataframe for the current altitude level
        df_filtered = final_df[final_df['alt_ag'] == alt_ag]
        
        # Calculate Pearson correlation for 't' and 'h'
        pearson_corr_t, _ = pearsonr(df_filtered['xq2_t'], df_filtered['station_t'])
        pearson_corr_h, _ = pearsonr(df_filtered['xq2_h'], df_filtered['station_h'])
        
        # Calculate RMSE for 't' and 'h'
        rmse_t = np.sqrt(mean_squared_error(df_filtered['xq2_t'], df_filtered['station_t']))
        rmse_h = np.sqrt(mean_squared_error(df_filtered['xq2_h'], df_filtered['station_h']))
        
        # Store the results in the results_df DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'alt_ag': [alt_ag],
            'pearson_corr_t': [pearson_corr_t],
            'rmse_t': [rmse_t],
            'pearson_corr_h': [pearson_corr_h],
            'rmse_h': [rmse_h]
        })], ignore_index=True)
    
    # Calculate overall Pearson correlation and RMSE without regard to alt_ag
    overall_pearson_corr_t, _ = pearsonr(final_df['xq2_t'], final_df['station_t'])
    overall_pearson_corr_h, _ = pearsonr(final_df['xq2_h'], final_df['station_h'])
    
    overall_rmse_t = np.sqrt(mean_squared_error(final_df['xq2_t'], final_df['station_t']))
    overall_rmse_h = np.sqrt(mean_squared_error(final_df['xq2_h'], final_df['station_h']))
    
    # Append the overall statistics as an additional row
    overall_row = pd.DataFrame({
        'alt_ag': ['Overall'],
        'pearson_corr_t': [overall_pearson_corr_t],
        'rmse_t': [overall_rmse_t],
        'pearson_corr_h': [overall_pearson_corr_h],
        'rmse_h': [overall_rmse_h]
    })
    
    results_df = pd.concat([results_df, overall_row], ignore_index=True)
    results_df.to_csv(os.path.join(output_dir, f"stat_metrics_{mode}.csv"), index=False)
    results_df.to_html(os.path.join(output_dir, f"stat_metrics_{mode}.html"), index=False)

def plot_scatterplots(mast_data_df_path, mode, output_dir='plots//mast_experiment'):
    """
    Plots scatterplots of xq2_t vs station_t and xq2_h vs station_h colored by alt_ag,
    and saves the figures to the specified directory with distinct colors for each alt_ag value.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'xq2_t', 'station_t', 'xq2_h', 'station_h', and 'alt_ag' columns.
        output_dir (str): Directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define a hardcoded list of colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Example colors for 6 categories
    df = pd.read_csv(mast_data_df_path)
    df['time'] = pd.to_datetime(df['time'])
    df['alt_ag'] = pd.to_numeric(df['alt_ag'])

    unique_alt_ag = df['alt_ag'].unique()

    # Scatterplot of xq2_t vs station_t colored by alt_ag
    plt.figure(figsize=(8, 8))
    for idx, alt_ag in enumerate(unique_alt_ag):
        subset = df[df['alt_ag'] == alt_ag]
        plt.scatter(subset['xq2_t'], subset['station_t'], 
                    color=colors[idx % len(colors)],  # Use predefined colors
                    label=f'{alt_ag}m',marker = 'x', s=10)  # Smaller dots
    plt.title('xq2 Temperature vs Reference Temperature')
    plt.xlabel('xq2 Temperature (°C)')
    plt.ylabel('Reference Temperature (°C)')
    plt.legend(title='Altitude above ground', loc='best')
    plt.ylim(-1, 12)
    plt.xlim(-1, 12)
    plt.grid(True)
    plt.tight_layout()
    
    # Add red diagonal line without legend entry
    plt.plot([0, 12], [0, 12], linestyle = '--', color = 'darkgrey')
    
    plt.savefig(os.path.join(output_dir, f'scatter_{mode}_xq2_t_vs_station_t.png'))
    plt.close()

    # Scatterplot of xq2_h vs station_h colored by alt_ag
    plt.figure(figsize=(8, 8))
    for idx, alt_ag in enumerate(unique_alt_ag):
        subset = df[df['alt_ag'] == alt_ag]
        plt.scatter(subset['xq2_h'], subset['station_h'], 
                    color=colors[idx % len(colors)],  # Use predefined colors
                    label=f'{alt_ag}m', marker = 'x', s=10)  # Smaller dots
    plt.title('xq2 Humidity vs Reference Humidity')
    plt.xlabel('xq2 Humidity (°C)')
    plt.ylabel('Reference Humidity (°C)')
    plt.legend(title='Altitude above ground', loc='best')
    plt.ylim(50, 105)
    plt.xlim(50, 105)
    plt.grid(True)
    plt.tight_layout()
    
    # Add red diagonal line without legend entry
    plt.plot([50, 105], [50, 105], linestyle = '--', color = 'darkgrey')
    
    plt.savefig(os.path.join(output_dir, f'scatter_{mode}_xq2_h_vs_station_h.png'))
    plt.close()



def plot_wind_speed_impact(descents_mast_data_df_path, ascents_mast_data_df_path, output_dir):
    '''
    Load the mast data DataFrame, calculate the differnce of xq2_t and station_t and of xq2_h and station_h, 
    then plot the differences against the wind speed at 20m.
    Parameters:
        descents_mast_data_df_path (str): The path to the CSV file containing the descents mast data DataFrame.
        ascents_mast_data_df_path (str): The path to the CSV file containing the ascents mast data DataFrame.
        output_dir (str): The directory to save the plots.
    '''
    os.makedirs(output_dir, exist_ok=True)

    # initialize an empty DataFrame to store the differences of ascents and descents
    final_df = pd.DataFrame(columns=['alt_ag', 'time', 'diff_t', 'diff_h', 'wind_speed_20m'])
    
    # load both datframes
    final_df_descents = pd.read_csv(descents_mast_data_df_path)
    final_df_descents['time'] = pd.to_datetime(final_df_descents['time'])
    final_df_ascents = pd.read_csv(ascents_mast_data_df_path)
    final_df_ascents['time'] = pd.to_datetime(final_df_ascents['time'])

    final_df["alt_ag"] = final_df_descents["alt_ag"]
    final_df["time"] = final_df_descents["time"]
    final_df["diff_t"] = final_df_ascents["xq2_t"] - final_df_descents["xq2_t"]
    final_df["diff_h"] = final_df_ascents["xq2_h"] - final_df_descents["xq2_h"]
    final_df["wind_speed_20m"] = final_df_ascents["wind_speed_20m"]

    # plot the differences as scatter plot with unique color for alt_ag levels
    plt.figure(figsize=(8, 8))
    for alt_ag in final_df['alt_ag'].unique():
        subset = final_df[final_df['alt_ag'] == alt_ag]
        plt.scatter(subset['wind_speed_20m'], subset['diff_t'], label=f'{alt_ag}m', s=10)
    plt.title('Wind Speed vs Temperature Difference Ascents - Descents')    
    plt.xlabel('Wind Speed at 20m (m/s)')
    plt.ylabel('Temperature Difference Ascent-Descent (°C)')
    plt.legend(title='Altitude above ground', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wind_speed_vs_temp_diff_modes.png'))
    plt.close()

    plt.figure(figsize=(8, 8))
    for alt_ag in final_df['alt_ag'].unique():
        subset = final_df[final_df['alt_ag'] == alt_ag]
        plt.scatter(subset['wind_speed_20m'], subset['diff_h'], label=f'{alt_ag}m', s=10)
    plt.title('Wind Speed vs Humidity Difference Ascents - Descents')
    plt.xlabel('Wind Speed at 20m (m/s)')
    plt.ylabel('Humidity Difference Ascent-Descent (%)')
    plt.legend(title='Altitude above ground', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wind_speed_vs_hum_diff_modes.png'))
    plt.close()

def uncertainty_thresholds(file_path_ascents, file_path_descents, thresholds_t, thresholds_h, output_dir):

    """
    Load two CSV files, compute the percentage of abs_diff_t and abs_diff_h values below or equal 
    to certain thresholds, store the results in DataFrames, and plot the results.
    
    Parameters:
        file_path_ascents (str): The path to the CSV file containing the ascent data.
        file_path_descents (str): The path to the CSV file containing the descent data.
        thresholds_t (list): A list of temperature thresholds to calculate the percentage of abs_diff_t values below or equal to.
        thresholds_h (list): A list of humidity thresholds to calculate the percentage of abs_diff_h values below or equal to.
        output_dir (str): The directory to save the plots.
    """
    # initialize output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df_ascents = pd.read_csv(file_path_ascents)
    df_descents = pd.read_csv(file_path_descents)
    
    # Convert time column to datetime
    df_ascents['time'] = pd.to_datetime(df_ascents['time'])
    df_descents['time'] = pd.to_datetime(df_descents['time'])

    # Initialize lists to store results
    results_t = []
    results_h = []

    # Calculate percentages for abs_diff_t
    total_points_df_ascents_t = len(df_ascents)
    total_points_df_descents_t = len(df_descents)
    for threshold in thresholds_t:
        percent_diff_t_ascents = (df_ascents['abs_diff_t'] <= threshold).sum() / total_points_df_ascents_t * 100
        percent_diff_t_descents = (df_descents['abs_diff_t'] <= threshold).sum() / total_points_df_descents_t * 100
        results_t.append({
            "Threshold": threshold, 
            "% Diff Ascents <= Threshold": percent_diff_t_ascents, 
            "% Diff Descents <= Threshold": percent_diff_t_descents
        })

    # Calculate percentages for abs_diff_h
    total_points_df_ascents_h = len(df_ascents)
    total_points_df_descents_h = len(df_descents)
    for threshold in thresholds_h:
        percent_diff_h_ascents = (df_ascents['abs_diff_h'] <= threshold).sum() / total_points_df_ascents_h * 100
        percent_diff_h_descents = (df_descents['abs_diff_h'] <= threshold).sum() / total_points_df_descents_h * 100
        results_h.append({
            "Threshold": threshold, 
            "% Diff Ascents <= Threshold": percent_diff_h_ascents, 
            "% Diff Descents <= Threshold": percent_diff_h_descents
        })

    # Convert lists to DataFrames
    results_df_t = pd.DataFrame(results_t)
    results_df_h = pd.DataFrame(results_h)

    # Plot for abs_diff_t
    plt.figure(figsize=(10, 6))
    plt.plot(results_df_t["Threshold"], results_df_t["% Diff Ascents <= Threshold"], label="T Agreement Ascents", marker='o')
    plt.plot(results_df_t["Threshold"], results_df_t["% Diff Descents <= Threshold"], label="T Agreement Descents", marker='s')
    plt.xlabel("Threshold (°C)")
    plt.ylabel("% <= Threshold")
    plt.title("Percentage of Temperature Measurements within uncertainty thesholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thresholds_plot_abs_diff_t.png"))
    plt.close()

    # Plot for abs_diff_h
    plt.figure(figsize=(10, 6))
    plt.plot(results_df_h["Threshold"], results_df_h["% Diff Ascents <= Threshold"], label="H Agreement Ascents", marker='o')
    plt.plot(results_df_h["Threshold"], results_df_h["% Diff Descents <= Threshold"], label="H Agreement Descents", marker='s')
    plt.xlabel("Threshold (%)")
    plt.ylabel("% <= Threshold")
    plt.title("Percentage of Humidity Measurements within uncertainty thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thresholds_plot_abs_diff_h.png"))
    plt.close()

    # Save the results as HTML tables
    results_df_t.to_html("data//mast_experiment//results//agreement_abs_diff_t.html", index=False)
    results_df_h.to_html("data//mast_experiment//results//agreement_abs_diff_h.html", index=False)
