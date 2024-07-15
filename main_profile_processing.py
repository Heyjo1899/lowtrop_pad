import os
from lowtrop_pad.processing_profiles import (
    load_and_reduce_profile_top,
    average_profiles_by_vertical_bins,
    extract_profile_times_and_coords
)
from lowtrop_pad.reanalysis_processing import (
    merge_era5_with_height_info,
    extract_ERA5_profiles_to_csv,
    extract_CARRA_profiles_to_csv,
    load_and_reduce_CARRA_data,
)


by_flight_dir = os.path.join(os.getcwd(), "data", "xq2", "by_flight")

# Load all profiles in a dictionary and reduce the top values by n
all_profiles = load_and_reduce_profile_top(by_flight_dir, red_n=3)

# Smoothing the profiles by vertical bins, including possiblity of custom near-surface bins
average_profiles_by_vertical_bins(
    all_profiles,
    bin_size=20,
    output_directory="data/xq2/averaged_profiles",
    custom_bins=True,
    bin1=3,
    bin2=5,
    bin3=10,
)

# Loading all processed profiles and extracting the date and time for every ascent
folder_path = 'data\\xq2\\averaged_profiles_custom_3_5_10_20'
df_times_profiles = extract_profile_times_and_coords(folder_path)

# Loading Era 5 Data and merging with height above ground information over levels
ERA5_path = 'G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_profiles.nc'
height_info_path = 'data\\reanalysis\\ERA5_level_info.csv'
ds_era5 = merge_era5_with_height_info(ERA5_path, height_info_path)

# Compare all dates of xq2 profiles and extract closest ERA5 profiles timewise and spatially
extract_ERA5_profiles_to_csv(df_times_profiles, ds_era5, output_folder = 'data\\reanalysis\\ERA5_extracted_profiles')

# Loading CARRA Data and reducing it to the specified area
file_path_CARRA = 'G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_t_profiles.nc'
ds_CARRA = load_and_reduce_CARRA_data(file_path_CARRA, lon_min=-18, lon_max = -15, lat_min = 81, lat_max = 82)

# Extracting CARRA profiles for all xq2 profiles
extract_CARRA_profiles_to_csv(df_times_profiles, ds_CARRA, 'data\\reanalysis\\CARRA_extracted_profiles')