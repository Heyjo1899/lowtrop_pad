import numpy as np
from lowtrop_pad.validation_campaigns import (
    load_save_mast_xq2_ascents,
    load_save_mast_xq2_descents,
    load_hl_data,
    load_mast_data,
    resample_save_hl_mast_data,
    load_data_mast_plotting,
    plot_mast_data,
    mast_data_df,
    calculate_differences,
    calculate_metrics,
    plot_scatterplots,
    plot_wind_speed_impact,
    uncertainty_thresholds,
    diff_ascent_descent,
    all_diffs_vs_wind_scatter,
)

# The code needs to run 2 times to study ascent and descents, maybe adapt that
# if it hasnt ran 2 times, the function for wind speed will fail

# Setting the mode if ascents or descents should be processed
mode = "ascents"
# mode = 'descents'
# Load and save mast xq2 data in seperate ascents
if False and mode == "ascents":
    directory_path = "data//mast_experiment//xq2"
    output_folder = "data//mast_experiment//single_experiments//single_ascents"

    load_save_mast_xq2_ascents(
        directory_path, output_folder, thresh=2, start_buffer=4, end_buffer=2
    )
if False and mode == "descents":
    directory_path = "data//mast_experiment//xq2"
    output_folder = "data//mast_experiment//single_experiments//single_descents"

    load_save_mast_xq2_descents(
        directory_path, output_folder, thresh=3.5, start_buffer=1, end_buffer=30
    )

# load humilog data
if False:
    directory_path = "data//mast_experiment"
    hl_data = load_hl_data(directory_path)


# load Mast Data
if False:
    directory = "data//met_stations//ICOS_mast"
    mast_data = load_mast_data(directory)

    hl_output_dir = "data//mast_experiment//single_experiments//humilog_single_exp"
    mast_output_dir = "data//mast_experiment//single_experiments//mast_data_single_exp"
    times_path = "data//mast_experiment//single_experiments//times_mast_campaigns.csv"
    resample_save_hl_mast_data(
        hl_data, mast_data, times_path, hl_output_dir, mast_output_dir
    )


# Loading data conveniently for plotting and plot the mast campaigns against altitude
if False:
    # Load the data
    data_by_mast = load_data_mast_plotting(mode)

    # Loop over each mast number and plot both temperature and humidity
    for mast_number, data_list in data_by_mast.items():
        # Plot temperature (t) vs altitude (alt_ag)
        plot_mast_data(
            mast_number, data_list, "t", f"{mast_number}_mast_t_vs_alt_ag_{mode}.png"
        )

        # Plot relative humidity (h) vs altitude (alt_ag)
        plot_mast_data(
            mast_number, data_list, "h", f"{mast_number}_mast_h_vs_alt_ag_{mode}.png"
        )

# Generate df on alt_ag station levels to compare with xq2
if False:
    mast_data_df(mode)

# Calculate mean differences and mean absolute differences
if False:
    mast_data_df_path = (
        f"data//mast_experiment//single_experiments//mast_data_df_{mode}.csv"
    )
    calculate_differences(mast_data_df_path, mode)

# Calculate r and RMSE
if False:
    mast_data_df_path = (
        f"data//mast_experiment//single_experiments//mast_data_df_{mode}.csv"
    )
    metrics_df = calculate_metrics(mast_data_df_path, mode)

# Scatterplot of xq2 vs reference data
if False:
    mast_data_df_path = (
        f"data//mast_experiment//single_experiments//mast_data_df_{mode}.csv"
    )
    plot_scatterplots(mast_data_df_path, mode, output_dir="plots/mast_experiment")

# Impact of Windspeed on Differences and returns df of differences Ascents - Descents
if True:
    descents_mast_data_df_path = (
        "data//mast_experiment//single_experiments//mast_data_df_descents.csv"
    )
    ascents_mast_data_df_path = (
        "data//mast_experiment//single_experiments//mast_data_df_ascents.csv"
    )
    plot_wind_speed_impact(
        descents_mast_data_df_path,
        ascents_mast_data_df_path,
        output_dir="plots//mast_experiment//wind_speed_impact",
    )

# Agreement with different thresholds
if True:
    thresholds_t = np.arange(0.1, 0.9, 0.1)  # From 0.3 to 1.2, step 0.1
    thresholds_h = np.arange(1, 15, 1)
    descents_mast_data_df_path = (
        "data//mast_experiment//single_experiments//mast_data_df_descents.csv"
    )
    ascents_mast_data_df_path = (
        "data//mast_experiment//single_experiments//mast_data_df_ascents.csv"
    )
    uncertainty_thresholds(
        ascents_mast_data_df_path,
        descents_mast_data_df_path,
        thresholds_t,
        thresholds_h,
        output_dir="plots//mast_experiment//uncertainty_thresholds",
    )


if True:
    # Change working directory and call the function
    diff_ascent_descent(
        ascents_dir=r"data\mast_experiment\single_experiments\single_ascents",
        descents_dir=r"data\mast_experiment\single_experiments\single_descents",
        output_dir=r"plots\mast_experiment\ascents_vs_descents",
    )

if True:
    wind_file_path = (
        "data//mast_experiment//single_experiments//mast_data_df_ascents.csv"
    )

    all_diffs_vs_wind_scatter(
        ascents_dir=r"data\mast_experiment\single_experiments\single_ascents",
        descents_dir=r"data\mast_experiment\single_experiments\single_descents",
        wind_file_path=wind_file_path,
        output_dir=r"plots\mast_experiment\ascents_vs_descents",
    )
print(f"Done for mode {mode}")
