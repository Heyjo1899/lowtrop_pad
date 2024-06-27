import os
from lowtrop_pad.plotting_profiles import plot_raw_and_smoothed_profiles_of_day

date = "20230810"
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
