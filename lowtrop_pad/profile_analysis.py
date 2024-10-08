import pandas as pd
import os


def calculate_and_save_differences(input_directory, output_directory):
    """
    Substracts carra and era5 T from xq2 T for all profiles in the directory and stores the differences in a csv file.

    Parameters:
    input_directory (str): Path to the directory containing the merged and interpolated profile csv files.
    output_directory (str): Path to the output directory to save the differences csv files.
    """
    # Initialize empty DataFrames to store results for carra and era5
    differences_carra_df = pd.DataFrame()
    differences_era5_df = pd.DataFrame()

    # Initialize empty DataFrames for each category
    categories = ["tundra", "water", "ice", "lake"]
    differences_carra_category_dfs = {
        category: pd.DataFrame() for category in categories
    }
    differences_era5_category_dfs = {
        category: pd.DataFrame() for category in categories
    }

    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Extract the profile name from the filename
                profile_name = file.split("_", 1)[1].rsplit(".", 1)[0]

                # Calculate differences for carra and era5
                df[f"{profile_name}_xq2-carra"] = df["xq2_T"] - df["carra_T"]
                df[f"{profile_name}_xq2-era5"] = df["xq2_T"] - df["era5_T"]

                # Select relevant columns for carra
                carra_columns = ["alt_ag", f"{profile_name}_xq2-carra"]
                carra_df = df[carra_columns]

                # Select relevant columns for era5
                era5_columns = ["alt_ag", f"{profile_name}_xq2-era5"]
                era5_df = df[era5_columns]

                # Merge with the main DataFrame for carra
                if differences_carra_df.empty:
                    differences_carra_df = carra_df
                else:
                    differences_carra_df = pd.merge(
                        differences_carra_df, carra_df, on="alt_ag", how="outer"
                    )

                # Merge with the main DataFrame for era5
                if differences_era5_df.empty:
                    differences_era5_df = era5_df
                else:
                    differences_era5_df = pd.merge(
                        differences_era5_df, era5_df, on="alt_ag", how="outer"
                    )

                # Check if the profile name belongs to any category and store accordingly
                for category in categories:
                    if category in profile_name:
                        # Merge into the appropriate category DataFrame for carra
                        if differences_carra_category_dfs[category].empty:
                            differences_carra_category_dfs[category] = carra_df
                        else:
                            differences_carra_category_dfs[category] = pd.merge(
                                differences_carra_category_dfs[category],
                                carra_df,
                                on="alt_ag",
                                how="outer",
                            )

                        # Merge into the appropriate category DataFrame for era5
                        if differences_era5_category_dfs[category].empty:
                            differences_era5_category_dfs[category] = era5_df
                        else:
                            differences_era5_category_dfs[category] = pd.merge(
                                differences_era5_category_dfs[category],
                                era5_df,
                                on="alt_ag",
                                how="outer",
                            )

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    # Save the overall DataFrames
    differences_carra_df.to_csv(
        os.path.join(output_directory, "differences_xq2-carra_all_profiles.csv"),
        index=False,
    )
    differences_era5_df.to_csv(
        os.path.join(output_directory, "differences_xq2-era5_all_profiles.csv"),
        index=False,
    )

    # Save the category-specific DataFrames
    for category in categories:
        differences_carra_category_dfs[category].to_csv(
            os.path.join(output_directory, f"differences_xq2-carra_{category}.csv"),
            index=False,
        )
        differences_era5_category_dfs[category].to_csv(
            os.path.join(output_directory, f"differences_xq2-era5_{category}.csv"),
            index=False,
        )


def calculate_and_save_absolute_differences(input_directory, output_directory):
    """
    Calculates the absolute differences between carra/era5 T and xq2 T for all profiles in the directory
    and stores the absolute differences in a csv file.

    Parameters:
    input_directory (str): Path to the directory containing the merged and interpolated profile csv files.
    output_directory (str): Path to the output directory to save the absolute differences csv files.
    """
    # Initialize empty DataFrames to store results for carra and era5
    abs_differences_carra_df = pd.DataFrame()
    abs_differences_era5_df = pd.DataFrame()

    # Initialize empty DataFrames for each category
    categories = ["tundra", "water", "ice", "lake"]
    abs_differences_carra_category_dfs = {
        category: pd.DataFrame() for category in categories
    }
    abs_differences_era5_category_dfs = {
        category: pd.DataFrame() for category in categories
    }

    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Extract the profile name from the filename
                profile_name = file.split("_", 1)[1].rsplit(".", 1)[0]

                # Calculate absolute differences for carra and era5
                df[f"{profile_name}_xq2-carra-abs-diff"] = abs(
                    df["xq2_T"] - df["carra_T"]
                )
                df[f"{profile_name}_xq2-era5-abs-diff"] = abs(
                    df["xq2_T"] - df["era5_T"]
                )

                # Select relevant columns for carra
                carra_columns = ["alt_ag", f"{profile_name}_xq2-carra-abs-diff"]
                carra_df = df[carra_columns]

                # Select relevant columns for era5
                era5_columns = ["alt_ag", f"{profile_name}_xq2-era5-abs-diff"]
                era5_df = df[era5_columns]

                # Merge with the main DataFrame for carra
                if abs_differences_carra_df.empty:
                    abs_differences_carra_df = carra_df
                else:
                    abs_differences_carra_df = pd.merge(
                        abs_differences_carra_df, carra_df, on="alt_ag", how="outer"
                    )

                # Merge with the main DataFrame for era5
                if abs_differences_era5_df.empty:
                    abs_differences_era5_df = era5_df
                else:
                    abs_differences_era5_df = pd.merge(
                        abs_differences_era5_df, era5_df, on="alt_ag", how="outer"
                    )

                # Check if the profile name belongs to any category and store accordingly
                for category in categories:
                    if category in profile_name:
                        # Merge into the appropriate category DataFrame for carra
                        if abs_differences_carra_category_dfs[category].empty:
                            abs_differences_carra_category_dfs[category] = carra_df
                        else:
                            abs_differences_carra_category_dfs[category] = pd.merge(
                                abs_differences_carra_category_dfs[category],
                                carra_df,
                                on="alt_ag",
                                how="outer",
                            )

                        # Merge into the appropriate category DataFrame for era5
                        if abs_differences_era5_category_dfs[category].empty:
                            abs_differences_era5_category_dfs[category] = era5_df
                        else:
                            abs_differences_era5_category_dfs[category] = pd.merge(
                                abs_differences_era5_category_dfs[category],
                                era5_df,
                                on="alt_ag",
                                how="outer",
                            )

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the overall DataFrames
    abs_differences_carra_df.to_csv(
        os.path.join(output_directory, "abs_differences_xq2-carra_all_profiles.csv"),
        index=False,
    )
    abs_differences_era5_df.to_csv(
        os.path.join(output_directory, "abs_differences_xq2-era5_all_profiles.csv"),
        index=False,
    )

    # Save the category-specific DataFrames
    for category in categories:
        abs_differences_carra_category_dfs[category].to_csv(
            os.path.join(output_directory, f"abs_differences_xq2-carra_{category}.csv"),
            index=False,
        )
        abs_differences_era5_category_dfs[category].to_csv(
            os.path.join(output_directory, f"abs_differences_xq2-era5_{category}.csv"),
            index=False,
        )


def calculate_mean_differences(input_directory, output_directory):
    """
    Calculates the mean differences or mean absolute differences and their standard deviations from 
    the files in the input directory and returns two DataFrames: one for carra and one for era5.

    Parameters:
    input_directory (str): Path to the directory containing the difference or absolute difference csv files.
    output_directory (str): Path to the directory where output CSVs will be saved.

    Returns:
    mean_differences_carra_df (pd.DataFrame): DataFrame with mean and standard deviation for carra.
    mean_differences_era5_df (pd.DataFrame): DataFrame with mean and standard deviation for era5.
    """

    # Initialize DataFrames to store means and standard deviations
    mean_differences_carra_df = pd.DataFrame()
    mean_differences_era5_df = pd.DataFrame()

    # Read all files in the input directory
    for file in os.listdir(input_directory):
        if file.endswith(".csv"):
            file_path = os.path.join(input_directory, file)
            df = pd.read_csv(file_path)

            # Determine the category and whether the file is for carra or era5
            if "carra" in file:
                if "all_profiles" in file:
                    mean_differences_carra_df["alt_ag"] = df["alt_ag"]
                    mean_differences_carra_df["mean_all_profiles"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_carra_df["std_all_profiles"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "tundra" in file:
                    mean_differences_carra_df["mean_tundra"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_carra_df["std_tundra"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "water" in file:
                    mean_differences_carra_df["mean_water"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_carra_df["std_water"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "ice" in file:
                    mean_differences_carra_df["mean_ice"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_carra_df["std_ice"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "lake" in file:
                    mean_differences_carra_df["mean_lake"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_carra_df["std_lake"] = df.iloc[:, 1:].std(axis=1, skipna=True)

            elif "era5" in file:
                if "all_profiles" in file:
                    mean_differences_era5_df["alt_ag"] = df["alt_ag"]
                    mean_differences_era5_df["mean_all_profiles"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_era5_df["std_all_profiles"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "tundra" in file:
                    mean_differences_era5_df["mean_tundra"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_era5_df["std_tundra"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "water" in file:
                    mean_differences_era5_df["mean_water"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_era5_df["std_water"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "ice" in file:
                    mean_differences_era5_df["mean_ice"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_era5_df["std_ice"] = df.iloc[:, 1:].std(axis=1, skipna=True)
                elif "lake" in file:
                    mean_differences_era5_df["mean_lake"] = df.iloc[:, 1:].mean(axis=1, skipna=True)
                    mean_differences_era5_df["std_lake"] = df.iloc[:, 1:].std(axis=1, skipna=True)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the mean differences and standard deviations DataFrames
    if "absolute" in input_directory:
        mean_differences_carra_df.to_csv(
            os.path.join(output_directory, "mean_absolute_differences_carra.csv"),
            index=False,
        )
        mean_differences_era5_df.to_csv(
            os.path.join(output_directory, "mean_absolute_differences_era5.csv"),
            index=False,
        )
    else:
        mean_differences_carra_df.to_csv(
            os.path.join(output_directory, "mean_differences_carra.csv"), index=False
        )
        mean_differences_era5_df.to_csv(
            os.path.join(output_directory, "mean_differences_era5.csv"), index=False
        )



def extract_times_from_merged_profiles(profile_directory):
    """
    Extracts time information from files in the merged_interpol_profiles directory
    and returns a dictionary mapping profile names to times.

    Parameters:
    profile_directory (str): Path to the directory containing the profile CSV files.

    Returns:
    dict: A dictionary mapping profile names to times.
    """
    time_map = {}

    for root, dirs, files in os.walk(profile_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Extract profile name and time
                if 'gradient' in file:
                    profile_name = file.split("_")[2].rsplit(".", 1)[0].split("_")[0]
                else:
                    profile_name = file.split("_", 1)[1].rsplit(".", 1)[0].split("_")[0]
                

                if "xq2_time" in df.columns:
                    # Take the first non-NaN time value
                    time_value = df["xq2_time"].dropna().iloc[0]
                    time_map[profile_name] = time_value

    return time_map


def time_array_of_profile_difference(
    profile_time_map, input_directory, output_directory
):
    """
    Replaces column names in each CSV file in the input directory with corresponding times
    using the provided profile time map.

    Parameters:
    input_directory (str): Path to the directory containing the CSV files with the column names to be replaced.
    profile_time_map (dict): Dictionary mapping profile names to times.
    """
    # Ensure the output directory is created
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)
            df = pd.read_csv(file_path)

            # Replace column names
            new_columns = []
            for col in df.columns:
                
                # Split by the last underscore and take the first part
                profile_name = col.rsplit("_", 1)[0].rsplit("_")[0]
                # Use the first part to look up in the profile_time_map
                if profile_name in profile_time_map:
                    new_columns.append(profile_time_map[profile_name])
                else:
                    new_columns.append(
                        col
                    )  
                    # keep the original name if no match is found

            df.columns = new_columns
            
            # Save the updated DataFrame
            output_file_path = os.path.join(output_directory, file_name)
            df.to_csv(output_file_path, index=False)


def resample_and_interpolate_time_arrays(
    input_directory,
    output_directory,
    resample_interval="1h",
    interpolation_method="linear",
):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all CSV files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_directory, file_name)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df_transposed = df.T

            # Step 2: Set the first row as header and convert the index to datetime
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed.drop(df_transposed.index[0])
            df_transposed.index = pd.to_datetime(
                df_transposed.index, infer_datetime_format=True, format="mixed"
            )

            # Step 3: Resample with a 1-hour interval
            df_resampled = df_transposed.resample(resample_interval).mean()

            # Step 4: Interpolate missing values (linear interpolation)
            df_interpolated = df_resampled.interpolate(interpolation_method)

            # Step 5: Transpose the DataFrame back to original structure
            df_final = df_interpolated.T
            df_final.columns.name = None  # Remove index name

            # Create the output file path
            output_file_path = os.path.join(output_directory, f'resampled_{file_name}')

            # Save the resulting DataFrame to a CSV file
            df_final.to_csv(output_file_path)

def single_profiles_to_array(profile_time_map, input_directory, variable, output_directory):
    """
    Read all profiles from directory and combine their selected variable into a single DataFrame. 
    The column headers are the file names. The index is the altitude above ground.
    
    Parameters:
    input_directory (str): Path to the directory containing the profiles.
    variable (str): Variable to extract from the profiles.
    output_directory (str): Path to the directory where the combined profiles will be saved.
    """

    # Dictionary to store the profiles
    all_profiles = pd.DataFrame()

    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                # Load the file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                # Add 'alt_ag' column only once as index
                if 'alt_ag' not in all_profiles.columns:
                    all_profiles['alt_ag'] = df['alt_ag']
                
                # ALL PROFILES:
                # Store the selected variable with the file name as the column header
                if 'gradient' in file:
                    all_profiles[file.rsplit(sep='_')[2].rsplit(sep = '.')[0]] = df[variable]
                else:
                    all_profiles[file.rsplit(sep='_')[1].rsplit(sep = '.')[0]] = df[variable]


    # Set the 'alt_ag' as the index of the DataFrame
    all_profiles.set_index('alt_ag', inplace=True)
    
    
    new_columns = []
    print('map',profile_time_map)
    for col in all_profiles.columns:
        print(col)
        if col in profile_time_map:
            new_columns.append(profile_time_map[col])
        else:
            new_columns.append(
                col
            )
    all_profiles.columns = new_columns
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the combined all_profiles DataFrame
    output_path = os.path.join(output_directory, f'array_{variable}_all_profiles.csv')
    all_profiles.to_csv(output_path)

    categories = ["tundra", "water", "ice", "lake"]

    for category in categories: 

        profiles = pd.DataFrame() 

        # Loop through the directory and its subdirectories
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".csv"):  # Only process CSV files
                    if category in file:
                        # Load the file
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        # Add 'alt_ag' column only once as index
                        if 'alt_ag' not in profiles.columns:
                            profiles['alt_ag'] = df['alt_ag']
                        
                        # Store the selected variable with the file name as the column header
                        if 'gradient' in file:
                            print('mark', file)
                            profiles[file.rsplit(sep='_')[2].rsplit(sep = '.')[0]] = df[variable]
                        else:
                            profiles[file.rsplit(sep='_')[1].rsplit(sep = '.')[0]] = df[variable]


        # Set the 'alt_ag' as the index of the DataFrame
        profiles.set_index('alt_ag', inplace=True)
        
        new_columns = []
        for col in profiles.columns:

            if col in profile_time_map:
                new_columns.append(profile_time_map[col])
            else:
                new_columns.append(
                    col
                )
        profiles.columns = new_columns
        
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        # Save the combined profiles DataFrame
        output_path = os.path.join(output_directory, f'array_{variable}_{category}_profiles.csv')
        profiles.to_csv(output_path)

def identify_air_mass_changes(input_directory, met_station_directory, time_delta, temp_diff, output_directory): 
    """
    Identifies air mass changes defined as the temperature change exceeding the threshold within the time delta.
    
    Parameters:
    input_directory (str): Path to the directory containing the profile CSV files.
    met_station_directory (str): Path to the directory containing the meteorological station data.
    time_delta (str): Time span to look for air mass change (e.g., '1D' for 1 day, '6H' for 6 hours).
    temp_diff (float): Temperature difference threshold for identifying air mass changes.
    output_directory (str): Path to the output directory to save the identified changes.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create a list to store the file names with no air mass change
    no_change_files = []
    change_files = []

    # Load the meteorological station data using the correct delimiter (semicolon in this case)
    met_station = pd.read_csv(met_station_directory, sep=';')

    # Now 'DateTime' should be a separate column
    met_station['DateTime'] = pd.to_datetime(met_station['DateTime'], errors='coerce')
    met_station = met_station.dropna(subset=['DateTime', 'Temp(oC 9m)'])


    # Loop through the input directory and its subdirectories, processing CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                # Load the profile file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Get the first valid 'carra_time' and convert it to a datetime
                carra_time = pd.to_datetime(df['carra_time'].dropna().iloc[0], errors='coerce')

                # Define the start and end time for the time window
                start_time = carra_time - pd.Timedelta(time_delta)
                end_time = carra_time + pd.Timedelta(time_delta)

                # Extract the meteorological station data within the time window
                met_data = met_station[(met_station['DateTime'] >= start_time) & 
                                       (met_station['DateTime'] <= end_time)]

                # If there is no data in the time window, skip this file
                if met_data.empty:
                    print(f"Skipping file {file}: no meteorological data in the time window.")
                    continue

                # Calculate the maximal temperature difference within the time window
                temp_diff_max = met_data['Temp(oC 9m)'].max() - met_data['Temp(oC 9m)'].min()

                # Check if the temperature difference exceeds the threshold
                if temp_diff_max <= temp_diff:
                    # Store the file name in the list of no air mass changes
                    no_change_files.append(file)
                else:
                    # Store the file name in the list of air mass changes
                    change_files.append(file)

    # Save the list of files with no and air mass changes
    no_change_file_path = os.path.join(output_directory, f'no_air_mass_change_{temp_diff}_files.txt')
    change_file_path = os.path.join(output_directory, f'air_mass_change_{temp_diff}_files.txt')
    
    with open(no_change_file_path, 'w') as f:
        for file in no_change_files:
            f.write(f"{file}\n")

    with open(change_file_path, 'w') as f:
        for file in change_files:
            f.write(f"{file}\n")
    
def categorize_by_wind_direction(input_directory, met_station_directory, time_delta, output_directory):
    """
    Categorizes profile files based on the average wind direction into four cardinal directions: 
    North, East, South, and West, using the mean wind direction in the time window.

    Parameters:
    input_directory (str): Path to the directory containing the profile CSV files.
    met_station_directory (str): Path to the directory containing the meteorological station data.
    time_delta (str): Time span to look for wind direction (e.g., '1D' for 1 day, '1H' for 1 hour).
    output_directory (str): Path to the output directory to save the categorized files.
    """

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create lists to store file names based on wind direction categories
    north_files = []
    east_files = []
    south_files = []
    west_files = []

    # Load the meteorological station data using the correct delimiter (semicolon)
    met_station = pd.read_csv(met_station_directory, sep=';')

    # Convert 'DateTime' to a datetime object
    met_station['DateTime'] = pd.to_datetime(met_station['DateTime'], errors='coerce')
    met_station = met_station.dropna(subset=['DateTime', 'VD(degrees 9m)'])

    # Loop through the input directory and its subdirectories, processing CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                # Load the profile file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Get the first valid 'carra_time' and convert it to a datetime
                carra_time = pd.to_datetime(df['carra_time'].dropna().iloc[0], errors='coerce')

                # Define the start and end time for the time window
                start_time = carra_time - pd.Timedelta(time_delta)
                end_time = carra_time + pd.Timedelta(time_delta)

                # Extract the meteorological station data within the time window
                met_data = met_station[(met_station['DateTime'] >= start_time) & 
                                       (met_station['DateTime'] <= end_time)]

                # If there is no data in the time window, skip this file
                if met_data.empty:
                    print(f"Skipping file {file}: no meteorological data in the time window.")
                    continue

                # Calculate the mean wind direction within the time window
                mean_wind_direction = met_data['VD(degrees 9m)'].mean()

                # Categorize the mean wind direction into one of the four cardinal directions
                if (315 <= mean_wind_direction <= 360) or (0 <= mean_wind_direction <= 45):
                    north_files.append(file)
                elif 45 < mean_wind_direction <= 135:
                    east_files.append(file)
                elif 135 < mean_wind_direction <= 225:
                    south_files.append(file)
                elif 225 < mean_wind_direction <= 315:
                    west_files.append(file)

    # Save the lists of files based on wind direction
    north_file_path = os.path.join(output_directory, 'north_wind_files.txt')
    east_file_path = os.path.join(output_directory, 'east_wind_files.txt')
    south_file_path = os.path.join(output_directory, 'south_wind_files.txt')
    west_file_path = os.path.join(output_directory, 'west_wind_files.txt')

    with open(north_file_path, 'w') as f:
        for file in north_files:
            f.write(f"{file}\n")
    
    with open(east_file_path, 'w') as f:
        for file in east_files:
            f.write(f"{file}\n")
    
    with open(south_file_path, 'w') as f:
        for file in south_files:
            f.write(f"{file}\n")
    
    with open(west_file_path, 'w') as f:
        for file in west_files:
            f.write(f"{file}\n")