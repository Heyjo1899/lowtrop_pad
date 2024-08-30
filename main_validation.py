import os
from lowtrop_pad.validation_campaigns import(
    load_save_mast_xq2,
    load_hl_data
)

# Load and save mast xq2 data in seprate ascents
if True:
    directory_path = "data//mast_experiment//xq2"
    output_folder = os.path.join("data//mast_experiment", "single_ascents")
    load_save_mast_xq2(directory_path, output_folder, thresh = 2, start_buffer=7, end_buffer=2)

# load hunmilog data
if True:
    directory_path = "data//mast_experiment"
    hl_data = load_hl_data(directory_path)
    
for key in hl_data.keys():
    print(key)
    print(hl_data[key].head())
    print("\n")