import os
from lowtrop_pad.processing_profiles import (
    load_and_reduce_profile_top,
    average_profiles_by_vertical_bins,
)

by_flight_dir = os.path.join(os.getcwd(), "data", "xq2", "by_flight")
all_profiles = load_and_reduce_profile_top(by_flight_dir, red_n=3)
average_profiles_by_vertical_bins(
    all_profiles,
    bin_size=20,
    output_directory="data/xq2/averaged_profiles",
    custom_bins=True,
    bin1=3,
    bin2=5,
    bin3=10,
)
