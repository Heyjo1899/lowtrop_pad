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


def process_and_combine_CARRA_datasets(
    path_multilevel,  # Path to the CARRA_t_profiles.nc file
    path_t_skin,  # Path to the CARRA_t_skin_.nc file
    path_t_2m,  # Path to the CARRA_t_2m.nc file
    lon_min,
    lon_max,
    lat_min,
    lat_max,  # Geographical bounds
    output_path,  # Output file path for the combined dataset
):
    """
    Load multiple datasets, subset by geographical region, align timesteps,
    expand dimensions, combine datasets, and save the combined dataset.

    Parameters:
    - path_multilevel (str): Path to the CARRA_t_profiles.nc file.
    - path_t_skin (str): Path to the CARRA_t_skin_.nc file.
    - path_t_2m (str): Path to the CARRA_t_2m.nc file.
    - lon_min, lon_max (float): Longitude bounds.
    - lat_min, lat_max (float): Latitude bounds.
    - output_path (str): Path to save the combined dataset.
    """
    # Load datasets
    ds_carra_multilevel = xr.open_dataset(path_multilevel)
    ds_carra_t_skin = xr.open_dataset(path_t_skin)
    ds_carra_t_2m = xr.open_dataset(path_t_2m)

    # Adjust longitudes if necessary
    ds_carra_multilevel["longitude"] = xr.where(
        ds_carra_multilevel["longitude"] > 180,
        ds_carra_multilevel["longitude"] - 360,
        ds_carra_multilevel["longitude"],
    )
    ds_carra_t_skin["longitude"] = xr.where(
        ds_carra_t_skin["longitude"] > 180,
        ds_carra_t_skin["longitude"] - 360,
        ds_carra_t_skin["longitude"],
    )
    ds_carra_t_2m["longitude"] = xr.where(
        ds_carra_t_2m["longitude"] > 180,
        ds_carra_t_2m["longitude"] - 360,
        ds_carra_t_2m["longitude"],
    )

    # Subset the data based on lat/lon bounds
    ds_carra_multilevel_subset = ds_carra_multilevel.where(
        (ds_carra_multilevel.latitude >= lat_min)
        & (ds_carra_multilevel.latitude <= lat_max)
        & (ds_carra_multilevel.longitude >= lon_min)
        & (ds_carra_multilevel.longitude <= lon_max),
        drop=True,
    )
    ds_carra_t_skin_subset = ds_carra_t_skin.where(
        (ds_carra_t_skin.latitude >= lat_min)
        & (ds_carra_t_skin.latitude <= lat_max)
        & (ds_carra_t_skin.longitude >= lon_min)
        & (ds_carra_t_skin.longitude <= lon_max),
        drop=True,
    )
    ds_carra_t_2m_subset = ds_carra_t_2m.where(
        (ds_carra_t_2m.latitude >= lat_min)
        & (ds_carra_t_2m.latitude <= lat_max)
        & (ds_carra_t_2m.longitude >= lon_min)
        & (ds_carra_t_2m.longitude <= lon_max),
        drop=True,
    )

    # Determine common timesteps across datasets
    common_times = np.intersect1d(
        ds_carra_multilevel_subset.time.values, ds_carra_t_skin_subset.time.values
    )
    common_times = np.intersect1d(common_times, ds_carra_t_2m_subset.time.values)

    # Subset datasets to include only common timesteps
    ds_carra_multilevel_subset = ds_carra_multilevel_subset.sel(time=common_times)
    ds_carra_t_skin_subset = ds_carra_t_skin_subset.sel(time=common_times)
    ds_carra_t_2m_subset = ds_carra_t_2m_subset.sel(time=common_times)

    # Expand dimensions and rename variables if necessary
    ds_carra_t_skin_expanded = ds_carra_t_skin_subset.expand_dims(
        {"heightAboveGround": [0]}
    )
    ds_carra_t_2m_expanded = ds_carra_t_2m_subset.expand_dims(
        {"heightAboveGround": [2]}
    )

    # Rename variables to have a consistent name for temperature
    ds_carra_t_skin_expanded = ds_carra_t_skin_expanded.rename({"skt": "t"})
    ds_carra_t_2m_expanded = ds_carra_t_2m_expanded.rename({"t2m": "t"})

    # Combine datasets
    ds_combined = xr.concat(
        [ds_carra_t_skin_expanded, ds_carra_t_2m_expanded, ds_carra_multilevel_subset],
        dim="heightAboveGround",
    )

    # Sort by heightAboveGround to maintain order
    ds_combined = ds_combined.sortby("heightAboveGround")

    # Print the combined dataset to verify
    print(ds_combined)

    # Save the combined dataset to the specified location
    ds_combined.to_netcdf(output_path)


def extract_CARRA_profiles_to_csv(
    df_times_profiles, file_path_carra, output_folder, fixed_coords=True
):
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

    if fixed_coords:
        # Define the coordinates for each keyword
        lat_tundra, lon_tundra = 81.59346447299638783, -16.57929481465396293
        lat_ice, lon_ice = 81.55784926749311, -15.86558759324902
        lat_water, lon_water = 81.57936778503125, -16.29312124236424
        lat_lake, lon_lake = 81.59346447299638783, -16.57929481465396293

        # Replace latitude and longitude values based on file_name
        df_times_profiles.loc[
            df_times_profiles["file_name"].str.contains("tundra"),
            ["latitude", "longitude"],
        ] = [lat_tundra, lon_tundra]
        df_times_profiles.loc[
            df_times_profiles["file_name"].str.contains("ice"),
            ["latitude", "longitude"],
        ] = [lat_ice, lon_ice]
        df_times_profiles.loc[
            df_times_profiles["file_name"].str.contains("water"),
            ["latitude", "longitude"],
        ] = [lat_water, lon_water]
        df_times_profiles.loc[
            df_times_profiles["file_name"].str.contains("lake"),
            ["latitude", "longitude"],
        ] = [lat_lake, lon_lake]

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

        # Combine lat and lon differences to get the nearest point by euclidian distance in 2D space
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
