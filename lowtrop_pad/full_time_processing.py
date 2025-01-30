import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import glob


def combine_CARRA_datasets(
    path_multilevel,  # Path to merged_carra_own_grid.nc file
    path_t_skin,  # Path to carra_full_skin2.nc file
    path_t_2m,  # Path to carra_full_2m.nc file
    output_path,  # Output file path for the combined dataset
):
    """
    Load datasets, expand dimensions for single heights, combine datasets,
    and save the combined dataset.

    Parameters:
    - path_multilevel (str): Path to multilevel Carra file.
    - path_t_skin (str): Path to tskin Carra file.
    - path_t_2m (str): Path to 2m Carra file.
    - output_path (str): Path to save the combined dataset.
    """
    # Load datasets
    ds_carra_multilevel = xr.open_dataset(path_multilevel)
    ds_carra_t_skin = xr.open_dataset(path_t_skin)
    ds_carra_t_2m = xr.open_dataset(path_t_2m)

    # Expand dimensions and rename variables
    ds_carra_t_skin_expanded = ds_carra_t_skin.expand_dims({"heightAboveGround": [0]})
    ds_carra_t_2m_expanded = ds_carra_t_2m.expand_dims({"heightAboveGround": [2]})

    # Rename variables to have a consistent name for temperature
    ds_carra_t_skin_expanded = ds_carra_t_skin_expanded.rename({"skt": "t"})
    ds_carra_t_2m_expanded = ds_carra_t_2m_expanded.rename({"t2m": "t"})

    # Combine datasets
    ds_combined = xr.concat(
        [ds_carra_t_skin_expanded, ds_carra_t_2m_expanded, ds_carra_multilevel],
        dim="heightAboveGround",
    )

    # Sort by heightAboveGround to maintain order
    ds_combined = ds_combined.sortby("heightAboveGround")

    # Print the combined dataset to verify
    print(ds_combined)

    # Save the combined dataset to the specified location
    ds_combined.to_netcdf(output_path)


def extract_temperature_data(input_file_path, output_directory, coords=False):
    """
    Extracts temperature profiles from CARRA netcdf for specific coordinates and all heights.
    Saves the extracted data for tundra, ice, and water surfaces to CSV files.
    Parameters:
        input_file_path (str): Path to the merged dataset file.
        output_directory (str): Directory to save the extracted data CSV files.
        coords (bool): If True, saves the selected coordinates to a CSV file.
    """
    # Define the target coordinates for each surface type
    target_coordinates = {
        "tundra": (
            81.56750000000001,
            -16.600000000000023,
        ),  # (81.59346447299638783, -16.57929481465396293),
        "ice": (
            81.56750000000001,
            -15.902499999999975,
        ),  # (81.55784926749311, -15.86558759324902),
        "water": (81.57936778503125, -16.29312124236424),
    }

    # Load the merged dataset
    dataset = xr.open_dataset(input_file_path)

    # Convert longitudes to negative values if necessary
    dataset["longitude"] = np.where(
        dataset["longitude"] > 180, dataset["longitude"] - 360, dataset["longitude"]
    )

    # Get available latitudes and longitudes
    available_latitudes = dataset["latitude"].values
    available_longitudes = dataset["longitude"].values

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Prepare a dictionary to hold the selected coordinates
    selected_coordinates = {}
    print(target_coordinates.items())
    # Find the nearest available coordinates
    for surface_type, (target_lat, target_lon) in target_coordinates.items():
        nearest_lat = available_latitudes[
            np.abs(available_latitudes - target_lat).argmin()
        ]
        nearest_lon = available_longitudes[
            np.abs(available_longitudes - target_lon).argmin()
        ]
        selected_coordinates[surface_type] = (nearest_lat, nearest_lon)

        # Extract temperature values for the specified coordinates
        t_surface = dataset["t"].sel(
            latitude=nearest_lat, longitude=nearest_lon, method="nearest"
        )

        # Convert temperature from Kelvin to Celsius
        t_surface = t_surface - 273.15  # Convert to Celsius

        # Get all available heights from the dataset and convert to integers
        heights = dataset["heightAboveGround"].values.astype(int)

        # Create a dictionary to hold temperature data for this surface type
        temperature_data = {}

        # Loop through the available heights to extract the temperature data
        for height in heights:
            temperature_data[height] = t_surface.sel(heightAboveGround=height).values

        # Create a DataFrame
        df_surface = pd.DataFrame(temperature_data)

        # Set the index to the valid_time and convert it to the desired date format
        df_surface.index = dataset["valid_time"].values
        df_surface.index = pd.to_datetime(df_surface.index)
        df_surface.index = df_surface.index.strftime("%d.%m.%Y")  # Format as DD.MM.YYYY

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(output_directory, f"{surface_type}_1991_2024.csv")
        df_surface.to_csv(csv_file_path, header=True)

        print(f"Data for {surface_type} saved to {csv_file_path}")

    # Export the selected coordinates to a CSV file
    if coords:
        coord_df = pd.DataFrame(selected_coordinates).T.reset_index()
        coord_df.columns = ["Surface Type", "Latitude", "Longitude"]
        coord_csv_path = os.path.join(
            r"C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\GIS",
            "surface_coordinates_carra_full.csv",
        )
        coord_df.to_csv(coord_csv_path, index=False)

        print(f"Coordinates saved to {coord_csv_path}")


def extract_snow_fraction(
    tundra_csv_file, snow_fraction_netcdf, tundra_lat, tundra_lon, output_directory
):
    """
    Extracts snow fraction data from a CARRA netCDF file and aligns it with the time index from an existing raw profile CSV file.
    Ensures that the resulting data only includes the dates existing in the raw CSV file.
    Saves the resulting data to a new CSV file.

    Parameters:
        tundra_csv_file (str): Path to one of the existing raw profile CSV files.
        snow_fraction_netcdf (str): Path to the snow fraction dataset file.
        tundra_lat (float): Latitude of the tundra coordinate.
        tundra_lon (float): Longitude of the tundra coordinate.
        output_directory (str): Directory to save the extracted snow fraction CSV file.
    """

    # Load the tundra CSV and format the index as datetime
    df_raw = pd.read_csv(tundra_csv_file, index_col=0, parse_dates=True)

    # Ensure the time format is consistent (DD.MM.YYYY) for tundra CSV index
    time_index = pd.to_datetime(df_raw.index, dayfirst=True)

    # Load the snow fraction dataset
    snow_dataset = xr.open_dataset(snow_fraction_netcdf)

    # Convert longitudes to negative values if necessary
    snow_dataset["longitude"] = np.where(
        snow_dataset["longitude"] > 180,
        snow_dataset["longitude"] - 360,
        snow_dataset["longitude"],
    )

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Find the nearest available coordinate for the tundra location
    nearest_lat = snow_dataset["latitude"].values[
        np.abs(snow_dataset["latitude"].values - tundra_lat).argmin()
    ]
    nearest_lon = snow_dataset["longitude"].values[
        np.abs(snow_dataset["longitude"].values - tundra_lon).argmin()
    ]

    # Extract snow fraction values for the nearest coordinate
    snow_fraction = snow_dataset["fscov"].sel(
        latitude=nearest_lat, longitude=nearest_lon, method="nearest"
    )

    # Convert the 'valid_time' in snow_fraction to datetime format
    valid_time_snow = pd.to_datetime(snow_fraction["valid_time"].values)

    # Remove the time part (HH:MM:SS) from valid_time_snow to align with the tundra CSV time index
    valid_time_snow = valid_time_snow.normalize()  # Remove the time part

    # Create a DataFrame with the snow fraction data
    df_snow = pd.DataFrame(
        {"time": valid_time_snow, "snow_fraction": snow_fraction.values}
    )

    # Set 'valid_time' as the index for alignment
    df_snow.set_index("time", inplace=True)

    # Ensure both datasets have the same date format
    # Filter the snow fraction DataFrame to only include the dates present in the tundra CSV time index
    df_snow_aligned = df_snow.loc[df_snow.index.isin(time_index)].sort_index()
    # Save the aligned DataFrame to a CSV file
    csv_file_path = os.path.join(output_directory, "snow_fraction_1991_2024.csv")
    df_snow_aligned.to_csv(csv_file_path, header=True)

    print(f"Snow fraction data aligned with tundra CSV saved to {csv_file_path}")


def interpolate_and_resample_data(
    input_file, output_dir, step=1, interpolation_method="linear"
):
    """
    Interpolate and resample profiles to common vertical bins.
    Parameters:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the resampled CSV file.
        step (int): Step size for the new altitude range.
        interpolation_method (str): Interpolation method to use
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV file
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # Define the new altitude range for interpolation (from 0 to 500 meters)
    alt_resampled = np.arange(0, 501, step)  # 501 to include 500 in 1m steps

    # Prepare a DataFrame to store the interpolated data
    df_resampled = pd.DataFrame(index=df.index, columns=alt_resampled)

    # Interpolate each row individually
    for date, row in df.iterrows():
        # Original altitude levels from column names (excluding the first, which is the date)
        original_altitudes = np.array([int(col) for col in df.columns], dtype=float)
        # Interpolate to the new altitude range
        interpolated_row = np.interp(alt_resampled, original_altitudes, row.values)
        # Store in resampled DataFrame
        df_resampled.loc[date] = interpolated_row

    # Construct output filename
    output_file = os.path.join(
        output_dir, os.path.basename(input_file).replace(".csv", "_resampled.csv")
    )

    # Save the interpolated DataFrame to CSV
    df_resampled.to_csv(output_file)
    print(f"Interpolated data saved to {output_file}")


def calculate_anomalies(
    input_file,
    output_dir,
    period_start="01.01.1991",
    period_end="31.12.2020",
    span=0.18,
):
    """
    Calculate temperature anomalies based on a 30-year climate period (daily means for each height),
    with LOESS smoothing applied to the daily means.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the anomalies CSV file.
        period_start (str): Start date of the climate period in 'dd.mm.yyyy' format.
        period_end (str): End date of the climate period in 'dd.mm.yyyy' format.
        span (float): The span parameter for LOESS smoothing (default is 0.18).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV file and parse dates
    df = pd.read_csv(input_file, index_col=0, parse_dates=True, dayfirst=True)

    # Temporarily switch index to datetime for filtering
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y")
    period_start = pd.to_datetime(period_start, format="%d.%m.%Y")
    period_end = pd.to_datetime(period_end, format="%d.%m.%Y")

    # Filter for the 30-year period
    period_data = df.loc[period_start:period_end]

    # Calculate daily mean temperatures for each day-of-year and height
    daily_mean = period_data.groupby(period_data.index.dayofyear).mean()

    # Apply LOESS smoothing to each height's daily mean
    smoothed_daily_mean = pd.DataFrame(
        index=daily_mean.index, columns=daily_mean.columns
    )

    for height in daily_mean.columns:
        smoothed_values = lowess(daily_mean[height], daily_mean.index, frac=span)
        smoothed_daily_mean[height] = smoothed_values[:, 1]

    # Calculate anomalies by subtracting smoothed daily mean values
    anomalies = df.groupby(df.index.dayofyear).apply(
        lambda group: group - smoothed_daily_mean.loc[group.name]
    )

    # Reset MultiIndex and format date index back to 'dd.mm.yyyy'
    anomalies = anomalies.reset_index(level=0, drop=True)  # Drop day-of-year level
    anomalies.index = pd.to_datetime(
        anomalies.index, format="%d.%m.%Y"
    )  # Convert to datetime
    anomalies.sort_index(inplace=True)  # Sort based on datetime index
    # sort anomalies based on index
    anomalies.index = anomalies.index.strftime(
        "%d.%m.%Y"
    )  # Convert back to string after sorting
    # Save the anomalies to a CSV file
    output_file = os.path.join(
        output_dir, os.path.basename(input_file).replace(".csv", "_anomalies.csv")
    )
    anomalies.to_csv(output_file, index=True)
    print(f"Anomalies saved to {output_file}")


def calculate_and_save_gradients(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # File pattern for CSV files in the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    files = glob.glob(file_pattern)

    # Process each file
    for file in files:
        # Extract the surface name from the filename
        surface = os.path.basename(file).split("_")[0]

        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)
        print(df)

        # Calculate the vertical temperature gradients (rowwise)
        df_gradients = df.diff(
            axis=1
        )  # Calculate the difference along each row (between consecutive columns)
        print(df_gradients)
        # Format the index as dd.mm.yyyy
        df_gradients.index = df_gradients.index.strftime("%d.%m.%Y")

        # Save the gradients to a new CSV file
        output_file = os.path.join(
            output_directory, f"{surface}_gradients_carra_1991_2024.csv"
        )
        df_gradients.to_csv(output_file)

        print(f"Saved temperature gradients for {surface} to: {output_file}")


def calculate_and_save_differences(
    input_directory, output_directory, air_mass_change=False
):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # File pattern for CSV files in the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    files = glob.glob(file_pattern)
    data = {}
    # Process each file
    for file in files:
        # I have 3 files, one for each surface (tundra, water ice) I want to load all 3 files and calculate the differences

        # Extract the surface name from the filename
        surface = os.path.basename(file).split("_")[0]

        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)

        # Store the data in a dictionary
        data[surface] = df

    # Calculate the differences between the surfaces
    diff_tundra_ice = data["tundra"] - data["ice"]
    diff_tundra_water = data["tundra"] - data["water"]

    # Format the index as dd.mm.yyyy
    diff_tundra_ice.index = diff_tundra_ice.index.strftime("%d.%m.%Y")
    diff_tundra_water.index = diff_tundra_water.index.strftime("%d.%m.%Y")

    # Save the differences to new CSV files
    output_file_tundra_ice = os.path.join(
        output_directory, "tundra-ice_differences.csv"
    )
    output_file_tundra_water = os.path.join(
        output_directory, "tundra-water_differences.csv"
    )

    diff_tundra_ice.to_csv(output_file_tundra_ice)
    diff_tundra_water.to_csv(output_file_tundra_water)

    print("Saved differences between tundra and ice/water")


def calculate_anomalies_era5(
    file_path,
    temp_level=850,
    geopotential_level=500,
    frac=0.18,
    out_dir=r"G:\LOWTROP_VRS\data\reanalysis",
):
    """
    Prepares the NetCDF data by calculating daily climatologies and anomalies for temperature and geopotential height,
    with smoothing applied to the climatologies.

    Parameters:
    - file_path (str): Path to the NetCDF file containing variables `t` and `z`.
    - temp_level (int): Pressure level for temperature (default is 850 hPa).
    - geopotential_level (int): Pressure level for geopotential height (default is 500 hPa).
    - frac (float): The fraction of points to use for LOESS smoothing.

    Returns:
    - anomalies (xarray.Dataset): Anomalies for temperature and geopotential height.
    - climatologies (xarray.Dataset): Smoothed climatologies for temperature and geopotential height.
    """
    g = 9.80665  # Gravitational acceleration in m/s²

    # Load the dataset
    ds = xr.open_dataset(file_path)

    # Extract temperature and geopotential height data
    temp_data = (
        ds["t"].sel(pressure_level=temp_level) - 273.15
    )  # Convert temperature to Celsius
    geopotential_data = (
        ds["z"].sel(pressure_level=geopotential_level) / g
    )  # Convert to geopotential height

    # Filter for the climate period (1991-2020)
    climate_period = slice("1991-06-01", "2020-08-31")
    temp_data_climate = temp_data.sel(valid_time=climate_period)
    geopotential_data_climate = geopotential_data.sel(valid_time=climate_period)

    # Group by day of the year to calculate climatologies (within the climate period)
    gb_temp_data = temp_data_climate.groupby("valid_time.dayofyear")
    gb_geopotential_data = geopotential_data_climate.groupby("valid_time.dayofyear")

    # Compute climatologies (mean over time for each grid cell within the climate period)
    temp_climatology = gb_temp_data.mean(dim="valid_time")
    geopotential_climatology = gb_geopotential_data.mean(dim="valid_time")

    # Apply LOESS smoothing to climatologies for each grid point
    def smooth_with_loess(data, frac):
        smoothed_data = np.full(data.shape, np.nan)
        for lat_idx in range(data.shape[1]):
            for lon_idx in range(data.shape[2]):
                series = data[:, lat_idx, lon_idx].values
                smoothed_series = lowess(series, np.arange(len(series)), frac=frac)[
                    :, 1
                ]
                smoothed_data[:, lat_idx, lon_idx] = smoothed_series
        return xr.DataArray(smoothed_data, coords=data.coords, dims=data.dims)

    smoothed_temp_climatology = smooth_with_loess(temp_climatology, frac)
    smoothed_geopotential_climatology = smooth_with_loess(
        geopotential_climatology, frac
    )

    # Compute anomalies (subtract smoothed climatology from daily data)
    temp_anomaly = temp_data.groupby("valid_time.dayofyear") - smoothed_temp_climatology
    geopotential_anomaly = (
        geopotential_data.groupby("valid_time.dayofyear")
        - smoothed_geopotential_climatology
    )

    # Remove the `dayofyear` coordinate from anomalies if it exists
    if "dayofyear" in temp_anomaly.coords:
        temp_anomaly = temp_anomaly.drop("dayofyear")
    if "dayofyear" in geopotential_anomaly.coords:
        geopotential_anomaly = geopotential_anomaly.drop("dayofyear")

    # Remove unnecessary coordinates from climatologies (retain only dayofyear)
    smoothed_temp_climatology = smoothed_temp_climatology.reset_coords(drop=True)
    smoothed_geopotential_climatology = smoothed_geopotential_climatology.reset_coords(
        drop=True
    )

    # Combine anomalies and climatologies into datasets
    anomalies = xr.Dataset(
        {"t_anomaly": temp_anomaly, "z_anomaly": geopotential_anomaly},
        coords=temp_data.coords,
    )
    climatologies = xr.Dataset(
        {
            "t_climatology": smoothed_temp_climatology,
            "z_climatology": smoothed_geopotential_climatology,
        },
        coords={
            "dayofyear": smoothed_temp_climatology["dayofyear"],
            "latitude": smoothed_temp_climatology["latitude"],
            "longitude": smoothed_temp_climatology["longitude"],
        },
    )

    # Define the output paths for anomalies and climatologies
    output_anomalies_path = os.path.join(out_dir, "era5_1991_2024_anomalies_smooth2.nc")
    output_climatologies_path = os.path.join(
        out_dir, "era5_1991_2024_climatologies_smooth2.nc"
    )

    # Save to NetCDF
    anomalies.to_netcdf(output_anomalies_path)
    climatologies.to_netcdf(output_climatologies_path)

    # Check results
    print(f"Anomalies saved to: {output_anomalies_path}")
    print(f"Climatologies saved to: {output_climatologies_path}")

    return anomalies, climatologies


def calculate_mass_change(
    flade_isblink_full_path,
    flade_isblink_north_path,
    mass_changes_file_path,
    output_directory,
):
    # Step 1: Load the CSV files
    flade_isblink_full_df = pd.read_csv(flade_isblink_full_path)
    flade_isblink_north_df = pd.read_csv(flade_isblink_north_path)
    mass_change_df = pd.read_csv(mass_changes_file_path)

    # Step 2: Clean and extract the relevant parts of the IDs
    # Clean `rgi_id` from Flade Isblink files
    flade_isblink_full_df["cleaned_id"] = flade_isblink_full_df["rgi_id"].apply(
        lambda x: x.split("-")[-2] + "." + x.split("-")[-1]
    )
    flade_isblink_north_df["cleaned_id"] = flade_isblink_north_df["rgi_id"].apply(
        lambda x: x.split("-")[-2] + "." + x.split("-")[-1]
    )

    # Clean `RGIId` from mass changes file
    mass_change_df["RGIId_cleaned"] = mass_change_df["RGIId"].apply(
        lambda x: x.split("-")[-1]
    )

    # Step 3: Filter rows for Full and North regions
    filtered_mass_change_full = mass_change_df[
        mass_change_df["RGIId_cleaned"].isin(flade_isblink_full_df["cleaned_id"])
    ]
    filtered_mass_change_north = mass_change_df[
        mass_change_df["RGIId_cleaned"].isin(flade_isblink_north_df["cleaned_id"])
    ]

    # Step 4: Extract year columns
    year_columns = [col for col in mass_change_df.columns if col.isdigit()]

    # Step 5: Merge area data with mass change data
    full_combined = filtered_mass_change_full.merge(
        flade_isblink_full_df[["cleaned_id", "area_km2"]],
        left_on="RGIId_cleaned",
        right_on="cleaned_id",
    )
    north_combined = filtered_mass_change_north.merge(
        flade_isblink_north_df[["cleaned_id", "area_km2"]],
        left_on="RGIId_cleaned",
        right_on="cleaned_id",
    )

    # For Full region
    # Multiply mass change (in original unit) by area to get absolute mass change per subregion
    for year in year_columns:
        full_combined[f"abs_mass_change_{year}"] = (
            full_combined[year] * full_combined["area_km2"] * 1e6 / 1e12
        )  # Convert to Gt

    # Sum absolute mass change over all subregions for each year
    abs_mass_change_full = full_combined[
        [f"abs_mass_change_{year}" for year in year_columns]
    ].sum()

    # Calculate area-weighted mean mass change for each year
    mean_mass_change_full = (
        full_combined[year_columns].multiply(full_combined["area_km2"], axis=0)
    ).sum() / full_combined["area_km2"].sum()

    # For North region
    for year in year_columns:
        north_combined[f"abs_mass_change_{year}"] = (
            north_combined[year] * north_combined["area_km2"] * 1e6 / 1e12
        )  # Convert to Gt

    abs_mass_change_north = north_combined[
        [f"abs_mass_change_{year}" for year in year_columns]
    ].sum()

    mean_mass_change_north = (
        north_combined[year_columns].multiply(north_combined["area_km2"], axis=0)
    ).sum() / north_combined["area_km2"].sum()

    # Step 8: Combine results into a single DataFrame
    combined_df = pd.DataFrame(
        {
            "Year": year_columns,
            "absolute_mass_change_FI_full": abs_mass_change_full.values,
            "mean_mass_change_FI_full": mean_mass_change_full.values,
            "absolute_mass_change_FI_north": abs_mass_change_north.values,
            "mean_mass_change_FI_north": mean_mass_change_north.values,
        }
    )

    # Step 9: Save the combined results to a CSV file
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, "Flade_Isblink_mass_change.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return combined_df
