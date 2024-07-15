import xarray as xr
import pandas as pd
import os
import numpy as np


 ### Function that MERGes CSV and NETCDF files, then loads Processed by flight data and takes the dates and
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
    df_csv = pd.read_csv(csv_path, delimiter=';')
    
    # Extract the relevant columns: 'n' and 'Geometric Altitude [m]'
    df_csv_relevant = df_csv[['n', 'Geometric Altitude [m]']]
    
    # Convert the DataFrame to an xarray Dataset
    ds_csv = df_csv_relevant.set_index('n').to_xarray()
    
    # Assign the geometric altitude to the heightAboveGround variable in the ERA5 dataset
    ds_era5['heightAboveGround'] = ds_csv['Geometric Altitude [m]'].sel(n=ds_era5['level'].values)

    # Adjust the longitude values by subtracting 360
    ds_era5['longitude'] = ds_era5['longitude'] - 360
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
        closest_time = ds_era5.time.sel(time=row['time'], method='nearest').values

        # Find the closest latitude and longitude in ds_era5
        closest_lat = ds_era5.latitude.sel(latitude=row['latitude'], method='nearest').values
        closest_lon = ds_era5.longitude.sel(longitude=row['longitude'], method='nearest').values

        # Extract the profiles for t, q, and heightAboveGround
        profile_t = ds_era5.t.sel(time=closest_time, latitude=closest_lat, longitude=closest_lon).to_dataframe()
        profile_q = ds_era5.q.sel(time=closest_time, latitude=closest_lat, longitude=closest_lon).to_dataframe()
        profile_height = ds_era5.heightAboveGround.to_dataframe()

        # Combine profiles into one DataFrame
        profile_combined = profile_t[['t']].copy()
        profile_combined['q'] = profile_q['q']
        profile_combined['heightAboveGround'] = profile_height['heightAboveGround']

        # Convert temperature from Kelvin to Celsius
        profile_combined['t'] = profile_combined['t'] - 273.15

        # Add coordinates and time information
        profile_combined['latitude'] = closest_lat
        profile_combined['longitude'] = closest_lon
        profile_combined['time'] = pd.to_datetime(closest_time)

        # Prepare output directory based on the date
        date_str = pd.to_datetime(row['time']).strftime('%Y%m%d')
        output_dir = os.path.join(output_folder, date_str)
        os.makedirs(output_dir, exist_ok=True)

        # Extract the relevant part of the file name
        file_name_parts = row['file_name'].rsplit('.', 1)[0]  # Remove the file extension
        new_file_name = f"ERA5_{file_name_parts}.csv"

        # Save the combined profile to CSV
        output_file_path = os.path.join(output_dir, new_file_name)
        profile_combined.to_csv(output_file_path, index=False)

        ds_era5.close()

def load_and_reduce_CARRA_data(file_path, lon_min=-18, lon_max = -15, lat_min = 81, lat_max = 82):
    """
    Process the dataset by converting longitudes and subsetting the data.

    Parameters:
    - file_path (str): Path to the CARRA file.
    - lon_min (float): Minimum longitude for the subset region.
    - lon_max (float): Maximum longitude for the subset region.
    - lat_min (float): Minimum latitude for the subset region.
    - lat_max (float): Maximum latitude for the subset region.

    Returns:
    - subset_CARRA (xarray.Dataset): The subsetted and processed dataset.
    """
    # Load the dataset
    ds_carra = xr.open_dataset(file_path)

    # Convert longitudes from 0-360 to -180 to 180 format if necessary
    ds_carra['longitude'] = ds_carra['longitude'] % 360
    ds_carra['longitude'] = xr.where(ds_carra['longitude'] > 180, 
                                     ds_carra['longitude'] - 360, 
                                     ds_carra['longitude'])

    # Use boolean indexing to create a mask for the desired region
    lat_mask = (ds_carra['latitude'] >= lat_min) & (ds_carra['latitude'] <= lat_max)
    lon_mask = (ds_carra['longitude'] >= lon_min) & (ds_carra['longitude'] <= lon_max)
    region_mask = lat_mask & lon_mask

    # Apply the mask to the dataset
    subset_CARRA = ds_carra.where(region_mask, drop=True)
    ds_carra.close()
    
    return subset_CARRA

### THIS FUNCTION WORKS BUT TEMPERATURE IS NOT STORED, why is there no temp. data?
def extract_CARRA_profiles_to_csv(df_times_profiles, ds_carra, output_folder):
    """
    Extract vertical profiles of temperature from the CARRA dataset for all profiles
    in df_times_profiles based on nearest time and coordinates, and save them as CSV files
    organized in subdirectories by date.

    Parameters:
    - df_times_profiles (pd.DataFrame): DataFrame containing time, latitude, and longitude information.
    - ds_carra (xr.Dataset): xarray Dataset containing CARRA data.
    - output_folder (str): Path to the output folder for saving CSV files.
    """
    # Convert the times in the DataFrame to pandas.Timestamp for comparisons
    df_times_profiles['time'] = pd.to_datetime(df_times_profiles['time'])

    for index, row in df_times_profiles.iterrows():
        # Find the closest time in ds_carra
        time_deltas = ds_carra.time - np.datetime64(row['time'])
        closest_time_index = np.abs(time_deltas).argmin()
        closest_time = ds_carra.time.isel(time=closest_time_index).values
        
        # Get latitude and longitude values as 2D arrays
        latitude_values = ds_carra.latitude.values
        longitude_values = ds_carra.longitude.values
        
        # Find index of nearest latitude
        lat_idx = np.unravel_index(np.abs(latitude_values - row['latitude']).argmin(), latitude_values.shape)
        lon_idx = np.unravel_index(np.abs(longitude_values - row['longitude']).argmin(), longitude_values.shape)
        
        # Extract the nearest lat/lon values
        closest_lat = latitude_values[lat_idx]
        closest_lon = longitude_values[lon_idx]

        # Extract the profile for temperature
        profile_t = ds_carra.t.sel(time=closest_time, method='nearest').isel(y=lat_idx[0], x=lat_idx[1]).to_dataframe().reset_index()
        print('profile_t:', profile_t)
        # Add latitude, longitude, and time information
        profile_t['latitude'] = closest_lat
        profile_t['longitude'] = closest_lon
        profile_t['time'] = pd.to_datetime(closest_time)

        # Prepare output directory based on the date
        date_str = pd.to_datetime(closest_time).strftime('%Y%m%d')
        output_dir = os.path.join(output_folder, date_str)
        os.makedirs(output_dir, exist_ok=True)
    
        # Construct the CSV file name
        file_name_parts = row['file_name'].rsplit('.', 1)[0]  # Remove the file extension
        file_name = f"CARRA_{file_name_parts}.csv"
        output_file_path = os.path.join(output_dir, file_name)

        # Save the temperature profile to CSV
        profile_t.to_csv(output_file_path, index=False)

        print(f"Saved CARRA: {output_file_path}")

        ds_carra.close()
