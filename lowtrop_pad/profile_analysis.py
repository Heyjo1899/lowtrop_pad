import pandas as pd
import os
import numpy as np
from scipy.stats import kendalltau
import itertools


def calculate_and_save_differences(
    input_directory, output_directory, air_mass_change=True, wind_direction=None
):
    """
    Subtracts carra and era5 T from xq2 T for profiles and stores the differences in csv files based on air mass change and wind direction.

    Parameters:
    input_directory (str): Path to the directory containing the merged and interpolated profile csv files.
    output_directory (str): Path to the output directory to save the differences csv files.
    air_mass_change (bool): Whether to filter profiles with no air mass change (default is True). File is set in function.
    wind_direction (str): Wind direction to filter profiles by ('north', 'west', 'east', 'south', or None for all directions).
    """

    # Initialize DataFrames for all directions and surfaces, including 'all' surface types
    categories = ["tundra", "water", "ice", "lake", "all"]
    directions = ["north", "east", "south", "west", "all"]
    differences_carra_dfs = {
        direction: {category: pd.DataFrame() for category in categories}
        for direction in directions
    }
    differences_era5_dfs = {
        direction: {category: pd.DataFrame() for category in categories}
        for direction in directions
    }

    # Load the air mass change and wind direction profile lists
    air_mass_file = os.path.join(
        "results", "conditional_profiles", "no_air_mass_change_3_files.txt"
    )
    if air_mass_change:
        with open(air_mass_file, "r") as f:
            air_mass_profiles = [line.strip() for line in f.readlines()]
    else:
        air_mass_profiles = None  # No filter if air_mass_change is False

    # Load the wind direction files if specified
    wind_files = {}
    for direction in directions[:-1]:  # Skip 'all' since it includes all profiles
        with open(
            os.path.join(
                "results", "conditional_profiles", f"{direction}_wind_files.txt"
            ),
            "r",
        ) as f:
            wind_files[direction] = [line.strip() for line in f.readlines()]
    # Loop through the input directory and process CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Filter by air mass change
                if air_mass_profiles and file not in air_mass_profiles:
                    continue

                df = pd.read_csv(file_path)
                profile_name = file.split("_", 1)[1].rsplit(".", 1)[0]

                # Calculate differences for carra and era5
                df[f"{profile_name}_xq2-carra"] = df["xq2_T"] - df["carra_T"]
                df[f"{profile_name}_xq2-era5"] = df["xq2_T"] - df["era5_T"]

                carra_columns = ["alt_ag", f"{profile_name}_xq2-carra"]
                era5_columns = ["alt_ag", f"{profile_name}_xq2-era5"]

                carra_df = df[carra_columns]
                era5_df = df[era5_columns]

                # Determine wind direction for this profile (if wind_direction is set)
                for direction in directions[
                    :-1
                ]:  # Iterate over north, east, south, west
                    if wind_direction is None or wind_direction == direction:
                        if (
                            file not in wind_files[direction]
                            and wind_direction != "all"
                        ):
                            continue

                        # Loop through surface types
                    surface_matched = False  # Flag to track if the profile matches a known surface type
                    for category in categories[:-1]:  # Exclude 'all' category
                        if category in profile_name:
                            surface_matched = True
                            # Merge carra data for the specific surface category
                            if differences_carra_dfs[direction][category].empty:
                                differences_carra_dfs[direction][category] = carra_df
                            else:
                                differences_carra_dfs[direction][category] = pd.merge(
                                    differences_carra_dfs[direction][category],
                                    carra_df,
                                    on="alt_ag",
                                    how="outer",
                                )

                            # Merge era5 data for the specific surface category
                            if differences_era5_dfs[direction][category].empty:
                                differences_era5_dfs[direction][category] = era5_df
                            else:
                                differences_era5_dfs[direction][category] = pd.merge(
                                    differences_era5_dfs[direction][category],
                                    era5_df,
                                    on="alt_ag",
                                    how="outer",
                                )

                    # Add the profile to the 'all' surface category for each specific wind direction
                    if differences_carra_dfs[direction]["all"].empty:
                        differences_carra_dfs[direction]["all"] = carra_df
                    else:
                        differences_carra_dfs[direction]["all"] = pd.merge(
                            differences_carra_dfs[direction]["all"],
                            carra_df,
                            on="alt_ag",
                            how="outer",
                        )

                    if differences_era5_dfs[direction]["all"].empty:
                        differences_era5_dfs[direction]["all"] = era5_df
                    else:
                        differences_era5_dfs[direction]["all"] = pd.merge(
                            differences_era5_dfs[direction]["all"],
                            era5_df,
                            on="alt_ag",
                            how="outer",
                        )

                # Add profiles to 'all' wind direction category (regardless of surface type)
                surface_matched = False
                for category in categories[:-1]:
                    if category in profile_name:
                        surface_matched = True
                        # Merge carra data
                        if differences_carra_dfs["all"][category].empty:
                            differences_carra_dfs["all"][category] = carra_df
                        else:
                            differences_carra_dfs["all"][category] = pd.merge(
                                differences_carra_dfs["all"][category],
                                carra_df,
                                on="alt_ag",
                                how="outer",
                            )

                        # Merge era5 data
                        if differences_era5_dfs["all"][category].empty:
                            differences_era5_dfs["all"][category] = era5_df
                        else:
                            differences_era5_dfs["all"][category] = pd.merge(
                                differences_era5_dfs["all"][category],
                                era5_df,
                                on="alt_ag",
                                how="outer",
                            )

                # If no specific surface category matched, add to 'all' surface for 'all' wind direction
                if not surface_matched:
                    # Merge carra data into the 'all' surface for 'all' wind direction
                    if differences_carra_dfs["all"]["all"].empty:
                        differences_carra_dfs["all"]["all"] = carra_df
                    else:
                        differences_carra_dfs["all"]["all"] = pd.merge(
                            differences_carra_dfs["all"]["all"],
                            carra_df,
                            on="alt_ag",
                            how="outer",
                        )

                    # Merge era5 data into the 'all' surface for 'all' wind direction
                    if differences_era5_dfs["all"]["all"].empty:
                        differences_era5_dfs["all"]["all"] = era5_df
                    else:
                        differences_era5_dfs["all"]["all"] = pd.merge(
                            differences_era5_dfs["all"]["all"],
                            era5_df,
                            on="alt_ag",
                            how="outer",
                        )

                # Always merge into 'all_all' regardless of surface or direction
                if differences_carra_dfs["all"]["all"].empty:
                    differences_carra_dfs["all"]["all"] = carra_df
                else:
                    differences_carra_dfs["all"]["all"] = pd.merge(
                        differences_carra_dfs["all"]["all"],
                        carra_df,
                        on="alt_ag",
                        how="outer",
                    )

                if differences_era5_dfs["all"]["all"].empty:
                    differences_era5_dfs["all"]["all"] = era5_df
                else:
                    differences_era5_dfs["all"]["all"] = pd.merge(
                        differences_era5_dfs["all"]["all"],
                        era5_df,
                        on="alt_ag",
                        how="outer",
                    )

    # Create output directories
    if air_mass_change:
        output_directory = os.path.join(output_directory, "no_air_mass_change_profiles")

    else:
        output_directory = os.path.join(output_directory, "all_profiles")

    os.makedirs(output_directory, exist_ok=True)

    # Add 'mean' column and save the results for each wind direction and surface
    for direction in directions:
        for category in categories:
            if not differences_carra_dfs[direction][category].empty:
                # Add the 'mean' column to carra differences DataFrame
                differences_carra_dfs[direction][category]["mean"] = (
                    differences_carra_dfs[direction][category]
                    .drop(
                        columns=["alt_ag"]
                    )  # Drop 'alt_ag' and any other specified columns
                    .mean(axis=1)
                )
            else:
                print(
                    f"Skipping mean calculation for empty carra DataFrame in direction: {direction}, category: {category}"
                )
            # Add the 'mean' column to era5 differences DataFrame
            if not differences_era5_dfs[direction][category].empty:
                # Add the 'mean' column to carra differences DataFrame
                differences_era5_dfs[direction][category]["mean"] = (
                    differences_era5_dfs[direction][category]
                    .drop(
                        columns=["alt_ag"]
                    )  # Drop 'alt_ag' and any other specified columns
                    .mean(axis=1)
                )
            else:
                print(
                    f"Skipping mean calculation for empty era5 DataFrame in direction: {direction}, category: {category}"
                )

    # Save the results for each wind direction and surface
    for direction in directions:
        for category in categories:
            # Save carra differences
            differences_carra_dfs[direction][category].to_csv(
                os.path.join(
                    output_directory,
                    f"differences_xq2-carra_{category}_{direction}.csv",
                ),
                index=False,
            )
            # Save era5 differences
            differences_era5_dfs[direction][category].to_csv(
                os.path.join(
                    output_directory, f"differences_xq2-era5_{category}_{direction}.csv"
                ),
                index=False,
            )


def calculate_and_save_absolute_differences(
    input_directory, output_directory, air_mass_change=True, wind_direction=None
):
    """
    Calculates the absolute differences between carra/era5 T and xq2 T for profiles in the directory
    and stores the absolute differences in csv files based on air mass change and wind direction.

    Parameters:
    input_directory (str): Path to the directory containing the merged and interpolated profile csv files.
    output_directory (str): Path to the output directory to save the absolute differences csv files.
    air_mass_change (bool): Whether to filter profiles with no air mass change (default is True).
    wind_direction (str): Wind direction to filter profiles by ('north', 'west', 'east', 'south', or None for all directions).
    """
    # Initialize DataFrames for all directions and surfaces
    categories = ["tundra", "water", "ice", "lake", "all"]
    directions = ["north", "east", "south", "west", "all"]
    abs_differences_carra_dfs = {
        direction: {category: pd.DataFrame() for category in categories}
        for direction in directions
    }
    abs_differences_era5_dfs = {
        direction: {category: pd.DataFrame() for category in categories}
        for direction in directions
    }

    # Load the air mass change and wind direction profile lists
    air_mass_file = os.path.join(
        "results", "conditional_profiles", "no_air_mass_change_3_files.txt"
    )
    if air_mass_change:
        with open(air_mass_file, "r") as f:
            air_mass_profiles = [line.strip() for line in f.readlines()]
    else:
        air_mass_profiles = None  # No filter if air_mass_change is False

    # Load the wind direction files if specified
    wind_files = {}
    for direction in directions[:-1]:  # Skip 'all' since it includes all profiles
        with open(
            os.path.join(
                "results", "conditional_profiles", f"{direction}_wind_files.txt"
            ),
            "r",
        ) as f:
            wind_files[direction] = [line.strip() for line in f.readlines()]

    # Loop through the input directory and process CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Filter by air mass change
                if air_mass_profiles and file not in air_mass_profiles:
                    continue

                df = pd.read_csv(file_path)
                profile_name = file.split("_", 1)[1].rsplit(".", 1)[0]

                # Calculate absolute differences for carra and era5
                df[f"{profile_name}_xq2-carra-abs-diff"] = abs(
                    df["xq2_T"] - df["carra_T"]
                )
                df[f"{profile_name}_xq2-era5-abs-diff"] = abs(
                    df["xq2_T"] - df["era5_T"]
                )

                carra_columns = ["alt_ag", f"{profile_name}_xq2-carra-abs-diff"]
                era5_columns = ["alt_ag", f"{profile_name}_xq2-era5-abs-diff"]

                carra_df = df[carra_columns]
                era5_df = df[era5_columns]

                # Determine wind direction for this profile (if wind_direction is set)
                for direction in directions[
                    :-1
                ]:  # Iterate over north, east, south, west
                    if wind_direction is None or wind_direction == direction:
                        if (
                            file not in wind_files[direction]
                            and wind_direction != "all"
                        ):
                            continue

                        # Loop through surface types
                        surface_matched = False  # Flag to track if the profile matches a known surface type
                        for category in categories[:-1]:  # Exclude 'all' category
                            if category in profile_name:
                                surface_matched = True
                                # Merge carra data for the specific surface category
                                if abs_differences_carra_dfs[direction][category].empty:
                                    abs_differences_carra_dfs[direction][category] = (
                                        carra_df
                                    )
                                else:
                                    abs_differences_carra_dfs[direction][category] = (
                                        pd.merge(
                                            abs_differences_carra_dfs[direction][
                                                category
                                            ],
                                            carra_df,
                                            on="alt_ag",
                                            how="outer",
                                        )
                                    )

                                # Merge era5 data for the specific surface category
                                if abs_differences_era5_dfs[direction][category].empty:
                                    abs_differences_era5_dfs[direction][category] = (
                                        era5_df
                                    )
                                else:
                                    abs_differences_era5_dfs[direction][category] = (
                                        pd.merge(
                                            abs_differences_era5_dfs[direction][
                                                category
                                            ],
                                            era5_df,
                                            on="alt_ag",
                                            how="outer",
                                        )
                                    )

                        # Add the profile to the 'all' surface category for each specific wind direction
                        if abs_differences_carra_dfs[direction]["all"].empty:
                            abs_differences_carra_dfs[direction]["all"] = carra_df
                        else:
                            abs_differences_carra_dfs[direction]["all"] = pd.merge(
                                abs_differences_carra_dfs[direction]["all"],
                                carra_df,
                                on="alt_ag",
                                how="outer",
                            )

                        if abs_differences_era5_dfs[direction]["all"].empty:
                            abs_differences_era5_dfs[direction]["all"] = era5_df
                        else:
                            abs_differences_era5_dfs[direction]["all"] = pd.merge(
                                abs_differences_era5_dfs[direction]["all"],
                                era5_df,
                                on="alt_ag",
                                how="outer",
                            )

                # Add profiles to 'all' wind direction category (regardless of surface type)
                surface_matched = False
                for category in categories[:-1]:
                    if category in profile_name:
                        surface_matched = True
                        # Merge carra data
                        if abs_differences_carra_dfs["all"][category].empty:
                            abs_differences_carra_dfs["all"][category] = carra_df
                        else:
                            abs_differences_carra_dfs["all"][category] = pd.merge(
                                abs_differences_carra_dfs["all"][category],
                                carra_df,
                                on="alt_ag",
                                how="outer",
                            )

                        # Merge era5 data
                        if abs_differences_era5_dfs["all"][category].empty:
                            abs_differences_era5_dfs["all"][category] = era5_df
                        else:
                            abs_differences_era5_dfs["all"][category] = pd.merge(
                                abs_differences_era5_dfs["all"][category],
                                era5_df,
                                on="alt_ag",
                                how="outer",
                            )

                # If no specific surface category matched, add to 'all' surface for 'all' wind direction
                if not surface_matched:
                    # Merge carra data into the 'all' surface for 'all' wind direction
                    if abs_differences_carra_dfs["all"]["all"].empty:
                        abs_differences_carra_dfs["all"]["all"] = carra_df
                    else:
                        abs_differences_carra_dfs["all"]["all"] = pd.merge(
                            abs_differences_carra_dfs["all"]["all"],
                            carra_df,
                            on="alt_ag",
                            how="outer",
                        )

                    # Merge era5 data into the 'all' surface for 'all' wind direction
                    if abs_differences_era5_dfs["all"]["all"].empty:
                        abs_differences_era5_dfs["all"]["all"] = era5_df
                    else:
                        abs_differences_era5_dfs["all"]["all"] = pd.merge(
                            abs_differences_era5_dfs["all"]["all"],
                            era5_df,
                            on="alt_ag",
                            how="outer",
                        )

                # Always merge into 'all_all' regardless of surface or direction
                if abs_differences_carra_dfs["all"]["all"].empty:
                    abs_differences_carra_dfs["all"]["all"] = carra_df
                else:
                    abs_differences_carra_dfs["all"]["all"] = pd.merge(
                        abs_differences_carra_dfs["all"]["all"],
                        carra_df,
                        on="alt_ag",
                        how="outer",
                    )

                if abs_differences_era5_dfs["all"]["all"].empty:
                    abs_differences_era5_dfs["all"]["all"] = era5_df
                else:
                    abs_differences_era5_dfs["all"]["all"] = pd.merge(
                        abs_differences_era5_dfs["all"]["all"],
                        era5_df,
                        on="alt_ag",
                        how="outer",
                    )

    # Create output directories
    if air_mass_change:
        output_directory = os.path.join(output_directory, "no_air_mass_change_profiles")
    else:
        output_directory = os.path.join(output_directory, "all_profiles")

    os.makedirs(output_directory, exist_ok=True)

    # Add 'mean' column and save the results for each wind direction and surface
    for direction in directions:
        for category in categories:
            if not abs_differences_carra_dfs[direction][category].empty:
                # Add the 'mean' column to carra differences DataFrame
                abs_differences_carra_dfs[direction][category]["mean"] = (
                    abs_differences_carra_dfs[direction][category]
                    .drop(
                        columns=["alt_ag"]
                    )  # Drop 'alt_ag' and any other specified columns
                    .mean(axis=1)
                )
            else:
                print(
                    f"Skipping abs mean calculation for empty carra DataFrame in direction: {direction}, category: {category}"
                )
            # Add the 'mean' column to era5 differences DataFrame
            if not abs_differences_era5_dfs[direction][category].empty:
                # Add the 'mean' column to carra differences DataFrame
                abs_differences_era5_dfs[direction][category]["mean"] = (
                    abs_differences_era5_dfs[direction][category]
                    .drop(
                        columns=["alt_ag"]
                    )  # Drop 'alt_ag' and any other specified columns
                    .mean(axis=1)
                )
            else:
                print(
                    f"Skipping abs mean calculation for empty era5 DataFrame in direction: {direction}, category: {category}"
                )

    # Save the results for each wind direction and surface
    for direction in directions:
        for category in categories:
            # Save carra differences
            abs_differences_carra_dfs[direction][category].to_csv(
                os.path.join(
                    output_directory,
                    f"abs_differences_xq2-carra_{category}_{direction}.csv",
                ),
                index=False,
            )
            # Save era5 differences
            abs_differences_era5_dfs[direction][category].to_csv(
                os.path.join(
                    output_directory,
                    f"abs_differences_xq2-era5_{category}_{direction}.csv",
                ),
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

            # Initialize surface_type and wind_direction
            surface_type = None
            wind_direction = None

            try:
                # Attempt to read the CSV file
                df = pd.read_csv(file_path)

                # Check if DataFrame is empty after reading
                if df.empty or df.shape[1] == 0:
                    raise pd.errors.EmptyDataError("No columns to parse from file.")

                # Split filename to extract surface type and wind direction
                parts = file.split("_")
                if "absolute" in input_directory:
                    surface_type = parts[3]
                    wind_direction = parts[4]
                else:
                    surface_type = parts[2]
                    wind_direction = parts[3]

                # Ensure surface_type and wind_direction are correctly assigned
                if surface_type is None or wind_direction is None:
                    raise ValueError(f"Invalid filename format: {file}")

            except (pd.errors.EmptyDataError, ValueError):
                # Fill the DataFrame with NaN for this category regardless of surface type and wind direction
                # Split filename to extract surface type and wind direction
                parts = file.split("_")
                if "absolute" in input_directory:
                    surface_type = parts[3]
                    wind_direction = parts[4]
                else:
                    surface_type = parts[2]
                    wind_direction = parts[3]
                mean_col_name = f"mean_{surface_type}_{wind_direction}"
                std_col_name = f"std_{surface_type}_{wind_direction}"
                mean_differences_carra_df[mean_col_name] = np.nan
                mean_differences_carra_df[std_col_name] = np.nan
                continue

            # Determine if the file is for carra or era5
            if "carra" in file:
                # Prepare column names for this specific category
                mean_col_name = f"mean_{surface_type}_{wind_direction}"
                std_col_name = f"std_{surface_type}_{wind_direction}"

                # Initialize columns if they don't exist
                if "alt_ag" not in mean_differences_carra_df.columns:
                    mean_differences_carra_df["alt_ag"] = df["alt_ag"]

                mean_differences_carra_df[mean_col_name] = df.drop(
                    columns=["alt_ag", "mean"], errors="ignore"
                ).mean(axis=1, skipna=True)
                mean_differences_carra_df[std_col_name] = df.drop(
                    columns=["alt_ag", "mean"], errors="ignore"
                ).std(axis=1, skipna=True)

            elif "era5" in file:
                # Prepare column names for this specific category
                mean_col_name = f"mean_{surface_type}_{wind_direction}"
                std_col_name = f"std_{surface_type}_{wind_direction}"

                # Initialize columns if they don't exist
                if "alt_ag" not in mean_differences_era5_df.columns:
                    mean_differences_era5_df["alt_ag"] = df["alt_ag"]

                mean_differences_era5_df[mean_col_name] = df.drop(
                    columns=["alt_ag", "mean"], errors="ignore"
                ).mean(axis=1, skipna=True)
                mean_differences_era5_df[std_col_name] = df.drop(
                    columns=["alt_ag", "mean"], errors="ignore"
                ).std(axis=1, skipna=True)
    # Ensure output directory exists
    if "no_air_mass_change_profiles" in input_directory:
        output_directory = os.path.join(output_directory, "no_air_mass_change_profiles")
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = os.path.join(output_directory, "all_profiles")
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


def calculate_and_save_correlations(
    input_directory, output_directory, air_mass_change=True, wind_direction=None
):
    """
    Calculates Mann-Kendall correlations for xq2, carra, and era5 profiles and stores the mean correlations and F-values.

    Parameters:
    input_directory (str): Directory containing the merged/interpolated profile csv files.
    output_directory (str): Directory where the results will be saved.
    air_mass_change (bool): Whether to filter profiles by air mass change (default is True).
    wind_direction (str): Wind direction filter ('north', 'west', 'east', 'south', or None for all directions).
    """
    # Initialize DataFrames for storing results
    categories = ["tundra", "water", "ice", "lake", "all"]
    directions = ["north", "east", "south", "west", "all"]

    # Prepare to store mean correlation and p-value results
    results_carra = []
    results_era5 = []

    # Load the air mass change and wind direction profile lists (same as before)
    air_mass_file = os.path.join(
        "results", "conditional_profiles", "no_air_mass_change_3_files.txt"
    )
    if air_mass_change:
        with open(air_mass_file, "r") as f:
            air_mass_profiles = [line.strip() for line in f.readlines()]
    else:
        air_mass_profiles = None  # No filter if air_mass_change is False

    wind_files = {}
    for direction in directions[:-1]:  # Skip 'all'
        with open(
            os.path.join(
                "results", "conditional_profiles", f"{direction}_wind_files.txt"
            ),
            "r",
        ) as f:
            wind_files[direction] = [line.strip() for line in f.readlines()]

    # Loop through files in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Filter by air mass change
                if air_mass_profiles and file not in air_mass_profiles:
                    # print('file', file)
                    # print('air masses', air_mass_profiles)

                    continue
                df = pd.read_csv(file_path)
                profile_name = file.split("_", 1)[1].rsplit(".", 1)[0]
                # print(profile_name)
                # Calculate Mann-Kendall correlation
                # Calculate Mann-Kendall correlation
                file_in_direction = (
                    False  # Flag to check if the file is in any wind direction category
                )
                for direction in directions[
                    :-1
                ]:  # Iterate over north, east, south, west
                    if file in wind_files[direction]:
                        file_in_direction = True
                        if wind_direction is None or wind_direction == direction:
                            # Proceed with the normal calculation
                            for category in categories[:-1]:  # Exclude 'all' category
                                if category in profile_name:
                                    df_cleaned = (
                                        df[["xq2_T", "carra_T", "era5_T"]].dropna()
                                    )  # Drop rows with NaNs in relevant columns

                                    if not df_cleaned.empty:  # Ensure there are enough data points after removing NaNs
                                        corr_carra, p_value_carra = kendalltau(
                                            df_cleaned["xq2_T"], df_cleaned["carra_T"]
                                        )
                                        corr_era5, p_value_era5 = kendalltau(
                                            df_cleaned["xq2_T"], df_cleaned["era5_T"]
                                        )
                                    else:
                                        corr_carra, p_value_carra = (
                                            float("nan"),
                                            float("nan"),
                                        )
                                        corr_era5, p_value_era5 = (
                                            float("nan"),
                                            float("nan"),
                                        )

                                    # Store results
                                    results_carra.append(
                                        {
                                            "profile": profile_name,
                                            "surface": category,
                                            "wind_direction": direction,
                                            "r": corr_carra,
                                            "p-value": p_value_carra,
                                        }
                                    )
                                    results_era5.append(
                                        {
                                            "profile": profile_name,
                                            "surface": category,
                                            "wind_direction": direction,
                                            "r": corr_era5,
                                            "p-value": p_value_era5,
                                        }
                                    )

                # If the file is not found in any direction, set direction to 'unclear'
                if not file_in_direction:
                    for category in categories[:-1]:  # Exclude 'all' category
                        if category in profile_name:
                            df_cleaned = df[
                                ["xq2_T", "carra_T", "era5_T"]
                            ].dropna()  # Drop rows with NaNs in relevant columns

                            if not df_cleaned.empty:
                                corr_carra, p_value_carra = kendalltau(
                                    df_cleaned["xq2_T"], df_cleaned["carra_T"]
                                )
                                corr_era5, p_value_era5 = kendalltau(
                                    df_cleaned["xq2_T"], df_cleaned["era5_T"]
                                )
                            else:
                                corr_carra, p_value_carra = float("nan"), float("nan")
                                corr_era5, p_value_era5 = float("nan"), float("nan")

                            # Store results with 'unclear' direction
                            results_carra.append(
                                {
                                    "profile": profile_name,
                                    "surface": category,
                                    "wind_direction": "unclear",
                                    "r": corr_carra,
                                    "p-value": p_value_carra,
                                }
                            )
                            results_era5.append(
                                {
                                    "profile": profile_name,
                                    "surface": category,
                                    "wind_direction": "unclear",
                                    "r": corr_era5,
                                    "p-value": p_value_era5,
                                }
                            )

    # Save results to a CSV file
    result_carra = pd.DataFrame(results_carra)
    result_era5 = pd.DataFrame(results_era5)
    # prepare and make output directory
    if air_mass_change:
        output_directory = os.path.join(output_directory, "no_air_mass_change_profiles")
    else:
        output_directory = os.path.join(output_directory, "all_profiles")
    os.makedirs(output_directory, exist_ok=True)
    result_carra.to_csv(
        os.path.join(output_directory, "mann_kendall_correlations_carra.csv"),
        index=False,
    )
    result_era5.to_csv(
        os.path.join(output_directory, "mann_kendall_correlations_era5.csv"),
        index=False,
    )


def calculate_mean_correlation_combinations(input_directory, output_directory):
    """
    Takes a dataframe with correlations of individual profiles and groups them into categories based on
    surface and wind direction and calculates the mean correlation for each category. Saves as csv.
    input_directory = directory to load the data from
    output directory = Directory to store the resulting csv fieles.
    """
    # Define unique surfaces and wind directions
    surfaces = ["ice", "water", "lake", "tundra", "all"]
    wind_directions = ["north", "west", "south", "east", "all"]

    # Create all combinations of surfaces and wind directions
    combinations = list(itertools.product(surfaces, wind_directions))

    os.makedirs(output_directory, exist_ok=True)

    # Loop through all CSV files in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Replace 'unclear' with 'all' in wind_direction
                df["wind_direction"] = df["wind_direction"].replace("unclear", "all")

                # Initialize lists to store mean r values and counts for this file
                mean_r_list = []
                count_list = []

                # Calculate mean r and count for each combination
                for surface, wind_direction in combinations:
                    if surface == "all" and wind_direction == "all":
                        mean_r = df["r"].mean()
                        count = df["r"].count()
                    elif surface == "all":
                        mean_r = df[df["wind_direction"] == wind_direction]["r"].mean()
                        count = df[df["wind_direction"] == wind_direction]["r"].count()
                    elif wind_direction == "all":
                        mean_r = df[df["surface"] == surface]["r"].mean()
                        count = df[df["surface"] == surface]["r"].count()
                    else:
                        mean_r = df[
                            (df["surface"] == surface)
                            & (df["wind_direction"] == wind_direction)
                        ]["r"].mean()
                        count = df[
                            (df["surface"] == surface)
                            & (df["wind_direction"] == wind_direction)
                        ]["r"].count()

                    # Append mean r and count to the lists
                    mean_r_list.append(mean_r)
                    count_list.append(count)

                # Create a DataFrame for the current file
                results = pd.DataFrame(
                    combinations, columns=["surface", "wind_direction"]
                )
                results["file"] = file
                results["mean_r"] = mean_r_list
                results["n"] = count_list

                # Write the results to the output directory
                if "carra" in file:
                    results.to_csv(
                        os.path.join(
                            output_directory,
                            "mean_mann_kendall_corr_combinations_carra.csv",
                        ),
                        index=False,
                    )
                elif "era5" in file:
                    results.to_csv(
                        os.path.join(
                            output_directory,
                            "mean_mann_kendall_corr_combinations_era5.csv",
                        ),
                        index=False,
                    )


def extract_times_from_merged_profiles(profile_directory, output_csv_dir=None):
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
                if "gradient" in file:
                    profile_name = file.split("_")[2].rsplit(".", 1)[0].split("_")[0]
                else:
                    profile_name = file.split("_", 1)[1].rsplit(".", 1)[0].split("_")[0]

                if "xq2_time" in df.columns:
                    # Take the first non-NaN time value
                    time_value = df["xq2_time"].dropna().iloc[0]
                    time_map[profile_name] = time_value
    # Optionally save the data as a CSV
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)

        # Convert the dictionary to a DataFrame
        time_df = pd.DataFrame(list(time_map.items()), columns=["Profile", "Time"])
        # Save as CSV
        time_df.to_csv(os.path.join(output_csv_dir, "profile_times.csv"), index=False)

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
                    new_columns.append(col)
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
            output_file_path = os.path.join(output_directory, f"resampled_{file_name}")

            # Save the resulting DataFrame to a CSV file
            df_final.to_csv(output_file_path)


def single_profiles_to_array(
    profile_time_map, input_directory, variable, output_directory
):
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
                if "alt_ag" not in all_profiles.columns:
                    all_profiles["alt_ag"] = df["alt_ag"]

                # ALL PROFILES:
                # Store the selected variable with the file name as the column header
                if "gradient" in file:
                    all_profiles[file.rsplit(sep="_")[2].rsplit(sep=".")[0]] = df[
                        variable
                    ]
                else:
                    all_profiles[file.rsplit(sep="_")[1].rsplit(sep=".")[0]] = df[
                        variable
                    ]

    # Set the 'alt_ag' as the index of the DataFrame
    all_profiles.set_index("alt_ag", inplace=True)

    new_columns = []
    print("map", profile_time_map)
    for col in all_profiles.columns:
        print(col)
        if col in profile_time_map:
            new_columns.append(profile_time_map[col])
        else:
            new_columns.append(col)
    all_profiles.columns = new_columns
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save the combined all_profiles DataFrame
    output_path = os.path.join(output_directory, f"array_{variable}_all_profiles.csv")
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
                        if "alt_ag" not in profiles.columns:
                            profiles["alt_ag"] = df["alt_ag"]

                        # Store the selected variable with the file name as the column header
                        if "gradient" in file:
                            print("mark", file)
                            profiles[file.rsplit(sep="_")[2].rsplit(sep=".")[0]] = df[
                                variable
                            ]
                        else:
                            profiles[file.rsplit(sep="_")[1].rsplit(sep=".")[0]] = df[
                                variable
                            ]

        # Set the 'alt_ag' as the index of the DataFrame
        profiles.set_index("alt_ag", inplace=True)

        new_columns = []
        for col in profiles.columns:
            if col in profile_time_map:
                new_columns.append(profile_time_map[col])
            else:
                new_columns.append(col)
        profiles.columns = new_columns

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        # Save the combined profiles DataFrame
        output_path = os.path.join(
            output_directory, f"array_{variable}_{category}_profiles.csv"
        )
        profiles.to_csv(output_path)


def identify_air_mass_changes(
    input_directory, met_station_directory, time_delta, temp_diff, output_directory
):
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
    met_station = pd.read_csv(met_station_directory, sep=";")

    # Now 'DateTime' should be a separate column
    met_station["DateTime"] = pd.to_datetime(met_station["DateTime"], errors="coerce")
    met_station = met_station.dropna(subset=["DateTime", "Temp(oC 9m)"])

    # Loop through the input directory and its subdirectories, processing CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                # Load the profile file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Get the first valid 'carra_time' and convert it to a datetime
                carra_time = pd.to_datetime(
                    df["carra_time"].dropna().iloc[0], errors="coerce"
                )

                # Define the start and end time for the time window
                start_time = carra_time - pd.Timedelta(time_delta)
                end_time = carra_time + pd.Timedelta(time_delta)

                # Extract the meteorological station data within the time window
                met_data = met_station[
                    (met_station["DateTime"] >= start_time)
                    & (met_station["DateTime"] <= end_time)
                ]

                # If there is no data in the time window, skip this file
                if met_data.empty:
                    print(
                        f"Skipping file {file}: no meteorological data in the time window."
                    )
                    continue

                # Calculate the maximal temperature difference within the time window
                temp_diff_max = (
                    met_data["Temp(oC 9m)"].max() - met_data["Temp(oC 9m)"].min()
                )

                # Check if the temperature difference exceeds the threshold
                if temp_diff_max <= temp_diff:
                    # Store the file name in the list of no air mass changes
                    no_change_files.append(file)
                else:
                    # Store the file name in the list of air mass changes
                    change_files.append(file)

    # Save the list of files with no and air mass changes
    no_change_file_path = os.path.join(
        output_directory, f"no_air_mass_change_{temp_diff}_files.txt"
    )
    change_file_path = os.path.join(
        output_directory, f"air_mass_change_{temp_diff}_files.txt"
    )

    with open(no_change_file_path, "w") as f:
        for file in no_change_files:
            f.write(f"{file}\n")

    with open(change_file_path, "w") as f:
        for file in change_files:
            f.write(f"{file}\n")


def categorize_by_wind_direction(
    input_directory, met_station_directory, time_delta, output_directory
):
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
    met_station = pd.read_csv(met_station_directory, sep=";")

    # Convert 'DateTime' to a datetime object
    met_station["DateTime"] = pd.to_datetime(met_station["DateTime"], errors="coerce")
    met_station = met_station.dropna(subset=["DateTime", "VD(degrees 9m)"])

    # Loop through the input directory and its subdirectories, processing CSV files
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):  # Only process CSV files
                # Load the profile file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # Get the first valid 'carra_time' and convert it to a datetime
                carra_time = pd.to_datetime(
                    df["carra_time"].dropna().iloc[0], errors="coerce"
                )

                # Define the start and end time for the time window
                start_time = carra_time - pd.Timedelta(time_delta)
                end_time = carra_time + pd.Timedelta(time_delta)

                # Extract the meteorological station data within the time window
                met_data = met_station[
                    (met_station["DateTime"] >= start_time)
                    & (met_station["DateTime"] <= end_time)
                ]

                # If there is no data in the time window, skip this file
                if met_data.empty:
                    print(
                        f"Skipping file {file}: no meteorological data in the time window."
                    )
                    continue

                # Calculate the mean wind direction within the time window
                mean_wind_direction = met_data["VD(degrees 9m)"].mean()

                # Categorize the mean wind direction into one of the four cardinal directions
                if (315 <= mean_wind_direction <= 360) or (
                    0 <= mean_wind_direction <= 45
                ):
                    north_files.append(file)
                elif 45 < mean_wind_direction <= 135:
                    east_files.append(file)
                elif 135 < mean_wind_direction <= 225:
                    south_files.append(file)
                elif 225 < mean_wind_direction <= 315:
                    west_files.append(file)

    # Save the lists of files based on wind direction
    north_file_path = os.path.join(output_directory, "north_wind_files.txt")
    east_file_path = os.path.join(output_directory, "east_wind_files.txt")
    south_file_path = os.path.join(output_directory, "south_wind_files.txt")
    west_file_path = os.path.join(output_directory, "west_wind_files.txt")

    with open(north_file_path, "w") as f:
        for file in north_files:
            f.write(f"{file}\n")

    with open(east_file_path, "w") as f:
        for file in east_files:
            f.write(f"{file}\n")

    with open(south_file_path, "w") as f:
        for file in south_files:
            f.write(f"{file}\n")

    with open(west_file_path, "w") as f:
        for file in west_files:
            f.write(f"{file}\n")
