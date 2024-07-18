import os
from lowtrop_pad.plotting_profiles import (
    plot_raw_and_smoothed_profiles_of_day,
    plot_xq2_vs_reanalysis_profiles_of_day,
)

if False:
    date = "20230814"
    directory_path_smoothed = os.path.join("data", "xq2", "averaged_profiles_20", date)
    directory_path_raw = os.path.join("data", "xq2", "by_flight", date)
    directory_path_smoothed_custom = os.path.join(
        "data", "xq2", "averaged_profiles_custom_3_5_10_20", date
    )

    plot_raw_and_smoothed_profiles_of_day(
        date,
        directory_path_smoothed_custom,
        directory_path_raw,
        varname="T",
        file_ending=".csv",
    )

if True:
    date = "20230804"
    xq_2_path = os.path.join("data/xq2/averaged_profiles_custom_3_5_10_20", date)
    carra_path = os.path.join("data/reanalysis/CARRA_extracted_profiles", date)
    era5_path = os.path.join("data/reanalysis/ERA5_extracted_profiles", date)
    plot_xq2_vs_reanalysis_profiles_of_day(
        date,
        xq_2_path,
        carra_path,
        era5_path,
        x_varname="T",
        y_varname="alt_ag",
        file_ending=".csv",
    )
