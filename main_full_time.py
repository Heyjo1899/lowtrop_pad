import os
import numpy as np

from lowtrop_pad.full_time_processing import (
    extract_temperature_data,
    combine_CARRA_datasets,
    interpolate_and_resample_data,
    calculate_anomalies,
    calculate_and_save_gradients,
    calculate_and_save_differences,
    extract_snow_fraction,
    calculate_anomalies_era5,
    calculate_mass_change,
)
from lowtrop_pad.full_time_plotting import (
    process_and_plot_trends,
    plot_monthly,
    plot_monthly_differences,
    plot_surface_signature_threshold,
    plot_climatology_temperature_difference,
    plot_slope_over_doy,
    plot_snow_trends,
    plot_snow_fraction_bins,
    plot_snow_free_days,
    plot_snow_anomaly_trend,
    plot_snow_profiles,
    plot_average_snow_anomalies,
    km_clusters,
    evaluate_kmeans,
    plot_mean_anomalies_by_cluster,
    plot_cluster_occurrences,
    plot_mass_change,
    plot_snow_fraction_vs_in_situ,
    correlation_heatmap,
    plot_cluster_difference_anomaly,
    plot_smb_cluster,
    plot_smb_mar,
    cluster_contribution,
    plot_smb_and_t2m_anomalies,
    daily_means_per_cluster,
    plot_yearly_smb_with_t2m_anomalies,
    plot_smb_vs_elevation,
    plot_smb_doy_cluster,
    plot_combined_climatology_and_correlation,
    plot_combined_synoptic_and_occurrences,
    plot_smb_cluster_elevation,
    plot_smb_vs_ela,
    plot_smb_time_and_hyps,
    plot_smb_t2m_anomalies_and_scatter,
)

from lowtrop_pad.mar_processing import (
    subset_and_extract_files,
    merge_files_by_time,
    convert_smb,
    merge_mar_cluster,
    mar_to_elev_bins,
    summarize_ela_AAR,
)

# Combine CARRA datasets to include t2m and tskin (13 levels total)
if False:
    output_file_path = (
        "G:\\LOWTROP_VRS\\data\\reanalysis\\carra_own_grid\\CARRA_full_all_levels.nc"
    )
    combine_CARRA_datasets(
        path_multilevel="G:\\LOWTROP_VRS\\data\\reanalysis\\carra_own_grid\\merged_carra_own_grid.nc",
        path_t_skin="G:\\LOWTROP_VRS\\data\\reanalysis\\carra_own_grid\\carra_full_skin2.nc",
        path_t_2m="G:\\LOWTROP_VRS\\data\\reanalysis\\carra_own_grid\\carra_full_2m.nc",
        output_path=output_file_path,
    )
    print("Finished combining CARRA data")

# Extract temperature profiles for tundra, ice, and water surfaces
if False:
    merged_file_path = (
        r"G:\LOWTROP_VRS\data\reanalysis\carra_own_grid\CARRA_full_all_levels.nc"
    )
    output_directory = r"data\reanalysis\carra_1991_2024\raw"

    extract_temperature_data(merged_file_path, output_directory, coords=False)
    print("Finished extracting CARRA profiles")

# Extract Snow fraction to csv for tundra surface
# Make sure to take the same coords for tundra like in the extract temp profile function!
if False:
    tundra_lat = 81.56750000000001
    tundra_lon = -16.600000000000023
    tundra_csv_file = r"data\reanalysis\carra_1991_2024\raw\tundra_1991_2024.csv"
    snow_fraction_netcdf = (
        r"G:\LOWTROP_VRS\data\reanalysis\carra_own_grid\carra_full_albedo_fr_snow.nc"
    )
    output_directory = r"data\reanalysis\carra_1991_2024\raw"
    extract_snow_fraction(
        tundra_csv_file, snow_fraction_netcdf, tundra_lat, tundra_lon, output_directory
    )

# Resample and interpolate the extracted data to common vertical steps
if False:
    input_path = r"data\reanalysis\carra_1991_2024\raw"
    output_path = r"data\reanalysis\carra_1991_2024\resampled"

    for file in ["ice_1991_2024.csv", "tundra_1991_2024.csv", "water_1991_2024.csv"]:
        interpolate_and_resample_data(
            os.path.join(input_path, file),
            output_path,
            step=1,
            interpolation_method="pchip",
        )

# Calculate Daily Anomalies
if False:
    # Example usage
    input_path = r"data\reanalysis\carra_1991_2024\resampled"
    output_path = r"data\reanalysis\carra_1991_2024\resampled_anomalies"

    # Calculate anomalies for each file
    for file in [
        "ice_1991_2024_resampled.csv",
        "tundra_1991_2024_resampled.csv",
        "water_1991_2024_resampled.csv",
    ]:
        calculate_anomalies(
            os.path.join(input_path, file),
            output_path,
            period_start="01.01.1991",
            period_end="31.12.2020",
            span=0.18,
        )

# Calculate Daily Anomalies for differences between ice tundra and water tundra
if False:
    # Example usage
    input_path = r"data\reanalysis\carra_1991_2024\differences"
    output_path = r"data\reanalysis\carra_1991_2024\differences_anomalies"

    # Calculate anomalies for each file
    for file in [
        "tundra-ice_differences.csv",
        "tundra-water_differences.csv",
    ]:
        anom = calculate_anomalies(
            os.path.join(input_path, file),
            output_path,
            period_start="01.01.1991",
            period_end="31.12.2020",
            span=0.18,
        )

# Calculate Gradients for the resampled data
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled"
    output_directory = r"data\reanalysis\carra_1991_2024\resampled_gradients"
    calculate_and_save_gradients(input_directory, output_directory)

# Plot Mean Monthly and Yearly Anomalies
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled_anomalies"
    output_path = r"plots\carra_1991_2024\monthly_anomalies"

    # Call the function
    plot_monthly(input_directory, output_path, vmin=-2.5, vmax=2.5, ylim=500)

# Plot Mean Monthly and Yearly profiles
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled"
    output_path = r"plots\carra_1991_2024\monthly_profiles"

    # Call the function
    plot_monthly(input_directory, output_path, vmin=-3, vmax=6, ylim=500)

# Plot Linear Trend Coefficients in each height for anomalies
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled_anomalies"
    output_directory = r"plots\carra_1991_2024\trends"
    process_and_plot_trends(input_directory, output_directory, method="linear")

# Plot Linear Trend Coefficients/slope ove day of year
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    output_directory = r"plots\carra_1991_2024\trends"
    plot_slope_over_doy(
        input_directory,
        output_directory,
        height_level=400,
        method="linear",
        smooth_window=5,
    )  # Use smoothing with a 7-day window


# Calculate differences between Tundra - Ice and Tundra - Water
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled"
    output_directory = r"data\reanalysis\carra_1991_2024\differences"
    calculate_and_save_differences(input_directory, output_directory)

# Plot monthly differences between Tundra - Ice and Tundra - Water
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    output_path = r"plots\carra_1991_2024\differences"
    plot_monthly(input_directory, output_path, vmin=-2, vmax=2)

# Plot trends of monthly differences in different heights
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    output_directory = r"plots\carra_1991_2024\trends"
    process_and_plot_trends(input_directory, output_directory, method="linear")

# Plot mean monthly differences in different heights
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    output_directory = r"plots\carra_1991_2024\monthly_differences"
    plot_monthly_differences(input_directory, output_directory, ylim=500)

# Plot climatology temperature difference with snow fraction
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\differences"
    plot_climatology_temperature_difference(
        input_directory=input_directory,
        snow_fraction_file=snow_fraction_file,
        vmin=-2,
        vmax=2,
        output_path=output_path,
    )

# Plot Heatmap of Trends of differences and Snow Fraction
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\differences"

    plot_snow_trends(input_directory, snow_fraction_file, output_path, method="linear")

# Plot Heatmap of Trends of Temp. Anomalies and Snow Fraction
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\resampled_anomalies"
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\resampled_anomalies_trends_snow"

    plot_snow_trends(
        input_directory,
        snow_fraction_file,
        output_path,
        method="linear",
        vmin=-1.5,
        vmax=1.5,
    )

# Plot surface signature thresholds
if False:
    input_directory = r"data\reanalysis\carra_1991_2024\differences"
    output_directory = r"plots\carra_1991_2024\surfaces_signature_thresholds"
    plot_surface_signature_threshold(
        input_directory, output_directory, thresholds=np.arange(0.8, 3.4, 0.2), ylim=500
    )

# Plot snow fraction over time in different seasonal bins
if False:
    doy_list = np.arange(152, 245, 1).tolist()

    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\snow\snow_fraction_over_time_bins"

    plot_snow_fraction_bins(snow_fraction_file, output_path, doy_list, num_bins=6)

# Plot trends of snow free days for different snow fraction thresholds
if False:
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\snow\trend_snow_free_days"
    plot_snow_free_days(snow_fraction_file, output_path)

# Plot Anomalies  with and without snow  heatmap
if False:
    plot_snow_profiles(
        snow_fraction_file="data/reanalysis/carra_1991_2024/raw/snow_fraction_1991_2024.csv",
        tundra_file="data/reanalysis/carra_1991_2024/resampled_anomalies/tundra_1991_2024_resampled_anomalies.csv",
        output_dir="plots/carra_1991_2024/snow_no_snow_t_profiles",
        ylim_max=100,
        smooth_sigma=None,  # Set to None for no smoothing
    )

# Plot average t anomalies with and without snow
if False:
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    tundra_file = r"data\reanalysis\carra_1991_2024\resampled_anomalies\tundra_1991_2024_resampled_anomalies.csv"
    output_dir = r"plots\carra_1991_2024\snow_no_snow_t_profiles"

    plot_average_snow_anomalies(
        snow_fraction_file, tundra_file, output_dir, ylim_max=100
    )

# Plot snow fraction anomaly trends
if False:
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    output_path = r"plots\carra_1991_2024\snow\anomaly_trends"
    plot_snow_anomaly_trend(snow_fraction_file, output_path)

# Calculate Anomalies for ERA5 Data
if False:
    file_path = r"G:\LOWTROP_VRS\data\reanalysis\ERA5_daily_1991-2024_JJA.nc"
    anomalies, climatologies = calculate_anomalies_era5(
        file_path,
        temp_level=850,
        geopotential_level=500,
        frac=0.18,
        out_dir=r"G:\LOWTROP_VRS\data\reanalysis",
    )


# Evaluate KMeans Clustering to find optimal number of cluster
if False:
    data_file = r"data\reanalysis\carra_1991_2024\resampled_gradients\tundra_gradients_carra_1991_2024.csv"
    output_dir = r"plots\carra_1991_2024\KM_Clusters\optiomal_K"

    evaluate_kmeans(
        data_file=data_file,
        output_dir=output_dir,
        exclude_lowest=100,
        cluster_range=(2, 10),
        random_state=42,
        max_iter=500,
    )

# Cluster the data based on gradients and plot mean profiles per cluster and occurences
if False:
    gradient_file = r"data\reanalysis\carra_1991_2024\resampled_gradients\tundra_gradients_carra_1991_2024.csv"
    # I can choose profile file of my choice
    profile_file = (
        r"data\reanalysis\carra_1991_2024\resampled\tundra_1991_2024_resampled.csv"
    )
    output_dir = r"plots\publication_plots"

    kmeans_results = km_clusters(
        gradient_file=gradient_file,
        profile_file=profile_file,
        output_dir=output_dir,
        n_clusters=5,
        exclude_lowest=100,
        ymin=0,
        ymax=500,
        random_state=42,  # set to 42 for deterministic approach
        max_iter=500,
        include_single_profiles=False,
    )

# Plot mean difference anomaly related to each cluster
if False:
    clusters_df_path = r"results\k_means\100_5_clusters.csv"
    difference_anomaly_file = r"data/reanalysis/carra_1991_2024/differences_anomalies/tundra-ice_differences_anomalies.csv"
    output_dir = r"plots/carra_1991_2024/KM_Clusters/mean_profiles"
    snow_fraction_file = (
        r"data/reanalysis/carra_1991_2024/raw/snow_fraction_1991_2024.csv"
    )
    plot_cluster_difference_anomaly(
        difference_anomaly_file,
        clusters_df_path,
        output_dir,
        snow_fraction_file,
        snow_thresh=0.3,  # 1 for no threshold
        include_single_profiles=False,
    )
if False:
    # Plot mean clusters synoptic conditions
    anomaly_file_path = (
        r"G:\LOWTROP_VRS\data\reanalysis\era5_1991_2024_anomalies_smooth2.nc"
    )
    clusters_df_path = r"results\k_means\100_5_clusters.csv"
    output_path = r"plots\carra_1991_2024\KM_Clusters\mean_synoptic"
    plot_mean_anomalies_by_cluster(
        clusters_df_path=clusters_df_path,
        anomaly_file_path=anomaly_file_path,
        output_path=output_path,
        vmin=-2.5,
        vmax=2.5,
    )

    # plot cluster occurrences
    output_dir = r"plots\carra_1991_2024\KM_Clusters\occurrences"
    profile_times_path = r"results/uav_time_map/profile_times.csv"
    clusters_df_path = r"results\k_means\100_5_clusters.csv"
    plot_cluster_occurrences(
        clusters_df_path=clusters_df_path,
        output_dir=output_dir,
        profile_times_path=profile_times_path,
    )


# Calculate mass changes for Flade Isblink north and full region
if False:
    # File paths
    flade_isblink_full = "data/glacier_mass_change/Flade_Isblink_full.csv"
    flade_isblink_north = "data/glacier_mass_change/Flade_Isblink_north.csv"
    mass_changes_file = (
        "data/glacier_mass_change/GRL_gla_mean-cal-mass-change-series.csv"
    )
    output_directory = "data/glacier_mass_change/mass_change"

    # Calculate mass changes and save to csv + return dataframe
    mass_df = calculate_mass_change(
        flade_isblink_full, flade_isblink_north, mass_changes_file, output_directory
    )

# Plot mass changes for Flade Isblink
if False:
    plot_mass_change(
        "data/glacier_mass_change/mass_change/Flade_Isblink_mass_change.csv",
        variable_to_plot="absolute_mass_change_FI",
    )

# Plot snow fraction vs in situ data
if False:
    plot_snow_fraction_vs_in_situ(
        "data\\met_stations\\VRS_snow_data.csv",
        "data\\reanalysis\\carra_1991_2024\\raw\\snow_fraction_1991_2024.csv",
        "plots\\carra_1991_2024\\snow\\in_situ_validation",
    )

# Compute correlation heatmap for snow fraction- and temperature difference anomalies
if False:
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    input_directory = r"data\reanalysis\carra_1991_2024\differences_anomalies"
    output_path = r"plots\carra_1991_2024\differences_anomalies"

    # Set parameters
    p_value_sigma = 0.7  # Gaussian smoothing for snow fraction data
    correlation_filter_sigma = 0  # Gaussian smoothing for correlation results
    p_value_thresh = 0.01  # Significance level for correlation
    ylim_max = 500  # Maximum value for height axis

    # Call the function
    correlation_heatmap(
        snow_fraction_file=snow_fraction_file,
        input_directory=input_directory,
        output_path=output_path,
        p_value_sigma=p_value_sigma,
        correlation_filter_sigma=correlation_filter_sigma,
        p_value_thresh=p_value_thresh,
        ylim_max=ylim_max,
    )

# Load mar file in batches, subsetting them roughly, converting their coordinates to m, subsetting them baed on shapefile
if False:  # subset and extract yearly files
    input_dir = r"G:\LOWTROP_VRS\data\mar_daily_10km"
    output_dir = r"G:\LOWTROP_VRS\data\mar_daily_10km_FI"
    shapefile_path = r"C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\GIS\area_of_interest_mar_bb_3413.shp"
    combined_output_path = r"G:\LOWTROP_VRS\data\mar_FI_1991_2024.nc"

    subset_and_extract_files(input_dir, output_dir, shapefile_path)
    # merge files by time
    merge_files_by_time(output_dir, combined_output_path)

# parse mar data into elevation bins
if False:
    input_file = r"G:\LOWTROP_VRS\data\mar_FI_1991_2024.nc"
    output_dir = r"data\mar_smb_1991_2024_FI"
    mar_to_elev_bins(file_path_mar=input_file, output_dir=output_dir, summer=False)

# convert smb data to csv
if False:
    input_file = r"G:\LOWTROP_VRS\data\mar_FI_1991_2024.nc"
    output_dir = r"data\mar_smb_1991_2024_FI"
    convert_smb(input_file, output_dir)

if False:
    print("merge mar smb and cluster assignments")
    # merge mar smb and cluster data and save as csv
    smb_csv_path = r"data\mar_smb_1991_2024_FI\FI_mar_smb.csv"
    cluster_csv_path = r"results\k_means\100_5_clusters.csv"
    output_dir = r"results\k_means"
    merge_mar_cluster(smb_csv_path, cluster_csv_path, output_dir)

# Plot smb violin/boxplots plots per cluster
if False:
    print("smb boxplots and violins per cluster")
    file_path_merged_df = r"results\k_means\mar_cluster_merged.csv"
    output_dir = r"plots\flade_isblink\clusters_smb"

    # violin plot for specific mass balance
    plot_smb_cluster(
        file_path_merged_df,
        output_dir,
        variable_to_plot="Specific Mass Balance (mm WE)",
        plot_type="violin",
    )

    # boxplot for specific mass balance
    plot_smb_cluster(
        file_path_merged_df,
        output_dir,
        variable_to_plot="Specific Mass Balance (mm WE)",
        plot_type="boxplot",
    )

    # violin plot for smb anomaly
    plot_smb_cluster(
        file_path_merged_df,
        output_dir,
        variable_to_plot="SMB Anomaly (mm WE)",
        plot_type="violin",
    )

    # boxplot for smb anomaly
    plot_smb_cluster(
        file_path_merged_df,
        output_dir,
        variable_to_plot="SMB Anomaly (mm WE)",
        plot_type="boxplot",
    )

if False:
    # plot mean SMB for DOY per cluster
    print("SMB over DOY per cluster")
    file_path_merged_df = r"results\k_means\mar_cluster_merged.csv"
    output_dir = r"plots\flade_isblink\clusters_smb"
    daily_means_per_cluster(
        file_path_merged_df, output_dir, gaussian_sigma=8, plot_individual_points=False
    )

if False:
    # summarize cluster contribution to total SMB in html file
    print("cluster contribution table and annual mean smb + trends")
    file_path_merged_df = r"results\k_means\mar_cluster_merged.csv"
    output_dir = r"results\k_means"
    cluster_contribution(file_path_merged_df, output_dir)

    # plot yearly mean smb + trendlines
    output_dir = r"plots\flade_isblink\clusters_smb"
    plot_yearly_smb_with_t2m_anomalies(file_path_merged_df, output_dir)

# Plot SMB values for Flade Isblink
if False:
    print("SMB Flade Isblink")
    file_path_smb = r"data\mar_smb_1991_2024_FI\FI_mar_smb.csv"
    output_dir = r"plots\flade_isblink\smb"
    plot_smb_mar(file_path_smb, output_dir)

if False:
    # plot smb vs elevation and calculate hypsometric effect
    print("Hypsometry Flade Isblink")
    input_file = r"data\mar_smb_1991_2024_FI\mar_1991_2024_elev_10_bins.csv"
    output_dir = r"plots\flade_isblink\smb"
    plot_smb_vs_elevation(input_file=input_file, output_dir=output_dir)

    output_csv = r"results\k_means"
    summary_df = summarize_ela_AAR(input_file, output_csv)
    print(summary_df)

if False:
    # SMB over DOY per cluster and elevation bins
    print("SMB over day per cluster and elev bin")
    mar_elev_bin_file = r"data\mar_smb_1991_2024_FI\mar_1991_2024_elev_10_bins.csv"
    cluster_file = r"results\k_means\100_5_clusters.csv"
    output_dir = r"plots\flade_isblink\smb"
    plot_smb_doy_cluster(
        mar_elev_bin_file=mar_elev_bin_file,
        cluster_file=cluster_file,
        output_dir=output_dir,
    )

if False:
    # Climatology of Difference and Snow fraction + Correaltion Heatmap difference and snow fraction anomaly
    input_file_climatology = (
        r"data\reanalysis\carra_1991_2024\differences\tundra-ice_differences.csv"
    )
    snow_fraction_file = (
        r"data\reanalysis\carra_1991_2024\raw\snow_fraction_1991_2024.csv"
    )
    input_file_anomalies = r"data\reanalysis\carra_1991_2024\differences_anomalies\tundra-ice_differences_anomalies.csv"
    output_dir = r"plots\publication_plots"

    fig = plot_combined_climatology_and_correlation(
        input_file_climatology=input_file_climatology,
        input_file_anomalies=input_file_anomalies,
        snow_fraction_file=snow_fraction_file,
        output_dir=output_dir,
        vmin=-2,
        vmax=2,
        ylim=500,
        smooth_window=5,
        p_value_sigma=0,
        correlation_filter_sigma=1,
        p_value_thresh=0.01,
    )

if True:
    # Plot synoptic conditions and cluster occurrences
    anomaly_file_path = (
        r"G:\LOWTROP_VRS\data\reanalysis\era5_1991_2024_anomalies_smooth2.nc"
    )
    clusters_df_path = r"results\k_means\100_5_clusters.csv"
    profile_times_path = r"results\uav_time_map\profile_times.csv"
    output_path = r"plots\publication_plots"

    plot_combined_synoptic_and_occurrences(
        clusters_df_path=clusters_df_path,
        anomaly_file_path=anomaly_file_path,
        profile_times_path=profile_times_path,
        output_path=output_path,
        vmin=-2.5,
        vmax=2.5,
    )

# SMB over DOY per Cluster and SMB per Cluster over elev bins
if True:
    df_mar_cluster_merged = r"results\k_means\mar_cluster_merged.csv"
    df_mar_elev = r"data\mar_smb_1991_2024_FI\mar_1991_2024_elev_10_bins.csv"
    cluster_file = r"results\k_means\100_5_clusters.csv"
    output_dir = r"plots\publication_plots"
    plot_smb_cluster_elevation(
        df_mar_cluster_merged, df_mar_elev, cluster_file, output_dir
    )

if True:
    # summarize cluster contribution to total SMB in html file
    file_path_merged_df = r"results\k_means\mar_cluster_merged.csv"
    output_dir = r"results\k_means"
    cluster_contribution(file_path_merged_df, output_dir)

    # plot yearly mean smb + trendlines and scatter JJA T and JJA SMB in 2nd function
    output_dir = r"plots\publication_plots"
    plot_smb_and_t2m_anomalies(file_path_merged_df, output_dir)
    plot_smb_t2m_anomalies_and_scatter(file_path_merged_df, output_dir)

if False:
    # plot scatter of yearly SMB vs ELA
    input_file = r"data\mar_smb_1991_2024_FI\mar_1991_2024_elev_10_bins.csv"
    output_dir_df = r"data\mar_smb_1991_2024_FI"
    output_dir_plot = r"plots\flade_isblink\smb"
    plot_smb_vs_ela(
        input_file=input_file,
        output_dir_df=output_dir_df,
        output_dir_plot=output_dir_plot,
        smoothing=True,
        frac=0.45,
    )


if False:
    # 4 mass balance publication plots combined
    file_path_smb = r"data\mar_smb_1991_2024_FI\FI_mar_smb.csv"
    file_path_elev = r"data\mar_smb_1991_2024_FI\mar_1991_2024_elev_10_bins.csv"
    file_path_zero = r"data\mar_smb_1991_2024_FI\ela_smb_df.csv"
    output_dir = r"plots\publication_plots"
    plot_smb_time_and_hyps(file_path_smb, file_path_elev, file_path_zero, output_dir)


# convert_smb add temp for average over ice cap: ttz_2m = ds["TTZ"].sel(ZTQLEV=2.0)
# make sure merge_mar_cluster function to merge cluster and smb also includes temp then
# plot_yearly_mean_smb code for plotting > Adding temp as 2nd plot then OR
# plot_yearly_smb_with_t2m_anomalies old code using carra ice t2m adapt
