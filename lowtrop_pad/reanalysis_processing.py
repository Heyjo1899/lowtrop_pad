import xarray as xr
import pandas as pd
import os
import numpy as np


### Function that MERGes CSV and NETCDF files, then loads Processed by flight data and takes the dates
def merge_era5_with_height_info(netcdf_path, csv_path):
    """
    Merges an ERA5 NetCDF dataset with a CSV file containing height information and adjusts the longitude values.

    Parameters:
    netcdf_path (str): Path to the ERA5 NetCDF file.
    csv_path (str): Path to the CSV file containing 'n' and 'Geometric Altitude [m]' columns.

    Returns:
    xr.Dataset: Merged xarray Dataset with adjusted longitude values.
    """
    # Open the NetCDF file as an xarray dataset
    ds_era5 = xr.open_dataset(netcdf_path)

    # Read the CSV file into a pandas DataFrame
    df_csv = pd.read_csv(csv_path, delimiter=";")

    # Extract the relevant columns: 'n' and 'Geometric Altitude [m]'
    df_csv_relevant = df_csv[["n", "Geometric Altitude [m]"]]

    # Convert the DataFrame to an xarray Dataset
    ds_csv = df_csv_relevant.set_index("n").to_xarray()

    # Assign the geometric altitude to the heightAboveGround variable in the ERA5 dataset
    ds_era5["heightAboveGround"] = ds_csv["Geometric Altitude [m]"].sel(
        n=ds_era5["level"].values
    )

    # Adjust the longitude values by subtracting 360
    ds_era5["longitude"] = ds_era5["longitude"] - 360
    return ds_era5


def extract_ERA5_profiles_to_csv(df_times_profiles, ds_era5, output_folder):
    """
    Extract vertical profiles of variables t and q from the ERA5 dataset for all profiles in df_times_profiles based on nearest time and coordinates,
    and save them as CSV files organized in subdirectories by date.

    Parameters:
    df_times_profiles (pd.DataFrame): DataFrame containing time, latitude, and longitude information.
    ds_era5 (xr.Dataset): xarray Dataset containing ERA5 data.
    output_folder (str): Path to the output folder for saving CSV files.
    """
    for index, row in df_times_profiles.iterrows():
        # Find the closest time in ds_era5
        closest_time = ds_era5.time.sel(time=row["time"], method="nearest").values

        # Find the closest latitude and longitude in ds_era5
        closest_lat = ds_era5.latitude.sel(
            latitude=row["latitude"], method="nearest"
        ).values
        closest_lon = ds_era5.longitude.sel(
            longitude=row["longitude"], method="nearest"
        ).values

        # Extract the profiles for t, q, and heightAboveGround
        profile_t = ds_era5.t.sel(
            time=closest_time, latitude=closest_lat, longitude=closest_lon
        ).to_dataframe()
        profile_q = ds_era5.q.sel(
            time=closest_time, latitude=closest_lat, longitude=closest_lon
        ).to_dataframe()
        profile_height = ds_era5.heightAboveGround.to_dataframe()

        # Combine profiles into one DataFrame
        profile_combined = profile_t.copy()
        profile_combined["q"] = profile_q["q"]
        profile_combined["alt_ag"] = profile_height["heightAboveGround"]

        # Convert temperature from Kelvin to Celsius
        profile_combined["T"] = profile_combined["t"] - 273.15

        # Add coordinates and time information
        profile_combined["lat"] = closest_lat
        profile_combined["lon"] = closest_lon
        profile_combined["time"] = pd.to_datetime(closest_time)

        # Prepare output directory based on the date
        date_str = pd.to_datetime(row["time"]).strftime("%Y%m%d")
        output_dir = os.path.join(output_folder, date_str)
        os.makedirs(output_dir, exist_ok=True)

        # Construct file names
        file_name = f"ERA5_{row['file_name']}"

        # Save the combined profile to CSV
        output_file_path = os.path.join(output_dir, file_name)
        profile_combined.to_csv(output_file_path, index=False)

        ds_era5.close()


def extract_CARRA_profiles_to_csv(df_times_profiles, file_path_carra, output_folder):
    """
    Extract vertical profiles of temperature from the CARRA dataset for all profiles
    in df_times_profiles based on nearest time and coordinates, and save them as CSV files
    organized in subdirectories by date.

    Parameters:
    - df_times_profiles (pd.DataFrame): DataFrame containing time, latitude, and longitude information.
    - ds_carra (xr.Dataset): xarray Dataset containing CARRA data.
    - output_folder (str): Path to the output folder for saving CSV files.
    """
    # Load the dataset
    ds_carra = xr.open_dataset(file_path_carra)

    # Adjust longitudes if necessary
    ds_carra["longitude"] = xr.where(
        ds_carra["longitude"] > 180, ds_carra["longitude"] - 360, ds_carra["longitude"]
    )

    for index, row in df_times_profiles.iterrows():
        # Find the closest time in ds_carra
        time_deltas = ds_carra.time - np.datetime64(row["time"])
        closest_time_index = np.abs(time_deltas).argmin()
        closest_time = ds_carra.time.isel(time=closest_time_index).values

        # Get latitude and longitude values as 2D arrays
        latitude_values = ds_carra.latitude.values
        longitude_values = ds_carra.longitude.values

        # Compute the absolute differences with the row's lat and lon
        lat_diff = np.abs(latitude_values - row["latitude"])
        lon_diff = np.abs(longitude_values - row["longitude"])

        # Combine lat and lon differences to get the nearest point in 2D space
        total_diff = np.sqrt(lat_diff**2 + lon_diff**2)
        lat_idx, lon_idx = np.unravel_index(total_diff.argmin(), total_diff.shape)

        # Extract the nearest lat/lon values from the 2D arrays
        closest_lat = latitude_values[lat_idx, lon_idx]
        closest_lon = longitude_values[lat_idx, lon_idx]

        # Extract the profile using the nearest indices and convert to a DataFrame
        profile_t = (
            ds_carra.t.sel(time=closest_time, method="nearest")
            .isel(y=lat_idx, x=lon_idx)
            .to_dataframe()
            .reset_index()
        )
        profile_t["T"] = (
            profile_t["t"] - 273.15
        )  # Convert temperature from Kelvin to Celsius

        # Add latitude, longitude, and time information
        profile_t["lat"] = closest_lat
        profile_t["lon"] = closest_lon
        profile_t["time"] = pd.to_datetime(closest_time)
        profile_t["alt_ag"] = profile_t["heightAboveGround"]

        # Prepare output directory based on the date
        date_str = pd.to_datetime(closest_time).strftime("%Y%m%d")
        output_dir = os.path.join(output_folder, date_str)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the CSV file name
        file_name = f"CARRA_{row['file_name']}"
        output_file_path = os.path.join(output_dir, file_name)

        # Save the temperature profile to CSV
        profile_t.to_csv(output_file_path, index=False)

        ds_carra.close()
