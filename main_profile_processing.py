import os
from lowtrop_pad.processing_profiles import (
    load_and_reduce_profile_top,
    average_profiles_by_vertical_bins,
    extract_profile_times_and_coords,
    save_coordinates_from_profiles,
    resample_interpolate_merge_profiles,
)
from lowtrop_pad.reanalysis_processing import (
    merge_era5_with_height_info,
    extract_ERA5_profiles_to_csv,
    extract_CARRA_profiles_to_csv,
    process_and_combine_CARRA_datasets,
)


if False:
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

if False:
    print("Starting to process ERA5 data")
    # Loading all processed profiles and extracting the date and time for every ascent
    xq2_path = "data\\xq2\\averaged_profiles_custom_3_5_10_20"
    df_times_profiles = extract_profile_times_and_coords(xq2_path)

    # Loading Era 5 Data and merging with height above ground information over levels
    ERA5_path = "G:\\LOWTROP_VRS\\data\\reanalysis\\ERA5_profiles.nc"
    height_info_path = "data\\reanalysis\\ERA5_level_info.csv"
    ds_era5 = merge_era5_with_height_info(ERA5_path, height_info_path)

    # Compare all dates of xq2 profiles and extract closest ERA5 profiles timewise and spatially
    extract_ERA5_profiles_to_csv(
        df_times_profiles,
        ds_era5,
        output_folder="data\\reanalysis\\ERA5_extracted_profiles",
    )

# Loading Carra multilevel and 2m and tskin, subsetting the region and  merging the data based on height above ground, save the nc file
if False:
    print("Starting to process CARRA data")
    lon_min = -18
    lon_max = -14
    lat_min = 80
    lat_max = 83

    # Define output path for the combined dataset
    output_file_path = "G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_subset_all_levels.nc"

    # Call the function with dataset paths
    process_and_combine_CARRA_datasets(
        path_multilevel="G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_t_profiles.nc",
        path_t_skin="G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_t_skin_.nc",
        path_t_2m="G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_t_2m.nc",
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        output_path=output_file_path,
    )
    print("Finished combining CARRA data")

# Extracting CARRA profiles for all xq2 profiles
if False:
    print("Starting to extract CARRA profiles")

    # Loading all processed profiles and extracting the date and time for every ascent
    folder_path_profiles = "data\\xq2\\averaged_profiles_custom_3_5_10_20"
    df_times_profiles = extract_profile_times_and_coords(folder_path_profiles)

    # Compare all dates of xq2 profiles and extract closest CARRA profiles timewise and spatially
    file_path_CARRA = "G:\\LOWTROP_VRS\\data\\reanalysis\\CARRA_subset_all_levels.nc"
    extract_CARRA_profiles_to_csv(
        df_times_profiles,
        file_path_carra=file_path_CARRA,
        output_folder="data\\reanalysis\\CARRA_extracted_profiles",
    )
    print("Finished extracting CARRA profiles")
# Save the coordinates of all profiles of a single day in a single file
if False:
    date = "20230810"
    xq2_path = f"data\\xq2\\averaged_profiles_custom_3_5_10_20\\{date}"
    carra_path = f"data\\reanalysis\\CARRA_extracted_profiles\\{date}"
    era5_path = f"data\\reanalysis\\ERA5_extracted_profiles\\{date}"

    save_coordinates_from_profiles(
        xq2_path, carra_path, era5_path, output_path="data\\coordinates_of_profiles"
    )

# Resample and interpolate all profiles to a common vertical grid and merge them into one file
if True:
    print("Starting to resample profiles")
    path1 = "data\\xq2\\averaged_profiles_custom_3_5_10_20"
    path2 = "data\\reanalysis\\CARRA_extracted_profiles"
    path3 = "data\\reanalysis\\ERA5_extracted_profiles"

    resample_interpolate_merge_profiles(
        path1,
        path2,
        path3,
        output_directory="data\\merged_interpol_profiles",
        interpolation_method="pchip",
        step=1,
    )
    print("Finished resample profiles")
