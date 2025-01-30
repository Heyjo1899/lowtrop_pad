import cdsapi

c = cdsapi.Client()
output_directory = "G:\\LOWTROP_VRS\\data\\reanalysis\\"

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "geopotential",
            "temperature",
        ],
        "pressure_level": [
            "500",
            "850",
            "1000",
        ],
        "year": [str(year) for year in range(1991, 2025)],
        "month": [
            "06",
            "07",
            "08",
        ],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [
            "12:00",
        ],
        "area": [
            85,
            -40,
            77,
            17,
        ],
    },
    output_directory + "ERA5_daily_1991-2024_JJA.nc",
)
