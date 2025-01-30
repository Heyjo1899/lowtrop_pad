import cdsapi
import os


if False:
    output_directory = "G:\\LOWTROP_VRS\\data\\reanalysis\\"

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-carra-height-levels",
        {
            "format": "netcdf",
            "domain": "west_domain",
            "variable": ["relative_humidity", "temperature"],
            "height_level": [
                "100_m",
                "150_m",
                "15_m",
                "200_m",
                "250_m",
                "300_m",
                "30_m",
                "400_m",
                "500_m",
                "50_m",
                "75_m",
            ],
            "product_type": "analysis",
            "time": [
                "00:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ],
            "year": "2023",
            "month": [
                "07",
                "08",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
        },
        output_directory + "CARRA_profiles2.nc",
    )


# LEARNING: when plotting correct area is shown,
# What is happening with the grid dimensions? Weird Longitudes though
# When having the copied grid parameters, I get a way finer grid than
# the original data, when using 1-1 as grid parameters, I get just 1 datapoint
# > Which grid parameter for original grid?

# Own Approach with Original grid downloading in chunks
if False:
    # Base path for saving the subsetted files
    output_dir = r"G:\LOWTROP_VRS\data\reanalysis\carra_full_time"
    output_dir_large = r"G:\LOWTROP_VRS\data\reanalysis\carra_large"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_large, exist_ok=True)

    # Define subset bounds
    lat_min, lat_max = 81.5, 81.6
    lon_min, lon_max = 343.4, 344.1

    # Define the months for summer (June, July, August)
    summer_months = [6, 7, 8]

    # Define the time of day for the data (12:00 UTC)
    target_hour = "12:00"

    # Function to create a dynamic filename for each year's subset
    def get_subset_filename(year):
        return os.path.join(output_dir, f"carra_subset_heights_{year}.nc")

    # Initialize the CDS API client
    client = cdsapi.Client()

    # Loop through each year
    years = range(1997, 2025)

    for year in years:
        # Define the request to download the dataset
        request = {
            "domain": "west_domain",  # Ensure this is properly defined
            "variable": ["temperature"],
            "height_level": [
                "15_m",
                "30_m",
                "50_m",
                "75_m",
                "100_m",
                "150_m",
                "200_m",
                "250_m",
                "300_m",
                "400_m",
                "500_m",
            ],
            "product_type": "analysis",
            "time": [target_hour],
            "year": [str(year)],
            "month": ["06", "07", "08"],
            "day": [
                str(day).zfill(2) for day in range(1, 32)
            ],  # Generates '01' to '31'
            "data_format": "netcdf",
        }

        # Download the dataset for the year
        filename_full = (
            f"G:\\LOWTROP_VRS\\data\\reanalysis\\carra_large\\carra_full_{year}.nc"
        )
        client.retrieve("reanalysis-carra-height-levels", request).download(
            filename_full
        )


if True:
    # Dataset and request details
    dataset = "reanalysis-carra-single-levels"
    request = {
        "domain": "west_domain",
        "level_type": "surface_or_atmosphere",
        "variable": ["fraction_of_snow_cover"],
        "product_type": "analysis",
        "time": ["12:00"],  # Only daily 12:00 data
        "year": [
            str(year)
            for year in range(1991, 2025)  # Generates the years dynamically
        ],
        "month": ["06", "07", "08"],  # Summer months
        "day": [f"{day:02d}" for day in range(1, 32)],  # Day range
        "grid": [0.0225, 0.0225],  # Approx. 2.5 km resolution
        "area": [81.6, -16.6, 81.5, -15.9],  # Verify coordinates [N, W, S, E]
        "data_format": "netcdf",
    }

    # CDS API client setup and retrieval
    client = cdsapi.Client()
    output_path = r"C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\Analysis\lowtrop_pad\data\reanalysis\carra_full_albedo_fr_snow.nc"

    try:
        client.retrieve(dataset, request).download(output_path)
        print(f"Data successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error during data retrieval: {e}")

# ALbedo and SNow fraction full domain
if False:
    dataset = "reanalysis-carra-single-levels"
    request = {
        "domain": "west_domain",
        "level_type": "surface_or_atmosphere",
        "variable": ["albedo", "fraction_of_snow_cover"],
        "product_type": "analysis",
        "time": ["12:00"],
        "year": ["2023"],
        "month": ["08"],
        "day": ["11"],  # Only 11th August
        "data_format": "netcdf",
    }

    client = cdsapi.Client()
    filename_full = (
        r"G:\LOWTROP_VRS\data\reanalysis\carra_albedo_snow_fraction_aug11.nc"
    )
    client.retrieve(dataset, request).download(filename_full)
