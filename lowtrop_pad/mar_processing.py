import os
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
# import rioxarray


def subset_and_extract_files(input_dir, output_dir, shapefile_path):
    """
    Subset and clip all NetCDF files in the input directory to a specified region.
    Save the results in the output directory.

    Args:
        input_dir (str): Directory containing the original NetCDF files.
        output_dir (str): Directory to save the subsetted and clipped files.
        shapefile_path (str): Path to the shapefile for clipping.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the shapefile and ensure it has the correct CRS
    shapefile = gpd.read_file(shapefile_path)
    shapefile = shapefile.to_crs(epsg=3413)

    # Process each NetCDF file in the input directory
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith(".nc"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing: {file_path}")

            # Load the dataset
            ds = xr.open_dataset(file_path)

            # Set CRS for the dataset
            ds = ds.rio.write_crs("EPSG:3413")

            # Subset the dataset by longitude and latitude
            ds = ds.sel(x=slice(411, 525), y=slice(-918, -760))

            # Convert x and y coordinates from km to m
            ds = ds.assign_coords(x=ds.coords["x"] * 1000, y=ds.coords["y"] * 1000)

            # Set the spatial dimensions explicitly
            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

            # Filter out non-spatial variables
            spatial_vars = [
                var
                for var in ds.data_vars
                if "x" in ds[var].dims and "y" in ds[var].dims
            ]
            spatial_subset = ds[spatial_vars]

            # Clip the dataset using the shapefile
            clipped = spatial_subset.rio.clip(
                shapefile.geometry.apply(mapping), shapefile.crs, drop=True
            )

            # Save the clipped dataset to the output directory
            output_file = os.path.join(output_dir, file_name)
            clipped.to_netcdf(output_file)
            print(f"Processed and saved: {output_file}")


def merge_files_by_time(output_dir, combined_output_path):
    """
    Merge all NetCDF files in the output directory along the TIME dimension.

    Args:
        output_dir (str): Directory containing the subsetted and clipped NetCDF files.
        combined_output_path (str): Path to save the combined NetCDF file.
    """
    # List of all processed NetCDF files
    files = sorted(
        [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".nc")
        ]
    )

    # Open all files and combine along the TIME dimension
    datasets = [xr.open_dataset(file) for file in files]
    combined_ds = xr.concat(datasets, dim="TIME")

    # Save the combined dataset
    combined_ds.to_netcdf(combined_output_path)
    print(f"Combined dataset saved to: {combined_output_path}")


def convert_smb(input_file, output_dir):
    """
    Converts the MAR SMB data to a CSV file with specific SMB, total Mass Change and Anomaly.
    Parameters:
        input_file (str): Path to the input NetCDF file containing the MAR SMB data.
        output_dir (str): Directory to save the output CSV file.
    """
    ds = xr.open_dataset(input_file)
    ds = ds.sel(SECTOR=1)
    # mask, then sum, then divide by sum of mask ro get mean over ice cap
    smb = ds["SMB"]
    msk = ds["MSK"] / 100  # mask initially in percent
    t = ds["TTZ"].sel(ZTQLEV=2.0)
    # Apply the mask and sum SMB over the spatial domain for each day

    masked_smb = smb * msk
    masked_t = t * msk

    # Calculate the specific mass balance by summing SMB in mm over the spatial domain and dividing by sum of mask
    specific_mass_balance = masked_smb.sum(dim=["x", "y"]) / msk.sum(dim=["x", "y"])
    t_mean = masked_t.sum(dim=["x", "y"]) / msk.sum(dim=["x", "y"])

    # Calculate the area per grid cell (10 km x 10 km)
    area_per_cell = 10 * 10  # km^2
    area_total = (
        msk.sum(dim=["x", "y"])[1] * area_per_cell
    )  # Total area based on the ice mask, taking 1st value from time dim

    # total smb in m^3 > convert specific mb from mm to m and area from km^2 to m^2
    total_smb_daily = (specific_mass_balance / 1000) * (area_total * 1000000)

    # Select the time range for the climatology period (1991-2020) for smb and temperature
    climatology_period = specific_mass_balance.sel(
        TIME=slice("1991-01-01", "2020-12-31")
    )
    climatology_period_t = t_mean.sel(TIME=slice("1991-01-01", "2020-12-31"))

    # Extract the day of the year (DOY) from the TIME dimension
    climatology_period["dayofyear"] = climatology_period["TIME"].dt.dayofyear
    climatology_period_t["dayofyear"] = climatology_period_t["TIME"].dt.dayofyear

    # Group by day of the year and calculate the mean for each day of the year across all years
    climatology = climatology_period.groupby("dayofyear").mean(dim="TIME")
    climatology_t = climatology_period_t.groupby("dayofyear").mean(dim="TIME")

    # Step 5: Smooth the climatology using LOESS filtering
    climatology_smooth = lowess(
        climatology.values, climatology.dayofyear.values, frac=0.045
    )

    climatology_t_smooth = lowess(
        climatology_t.values, climatology_t.dayofyear.values, frac=0.045
    )

    # Convert smoothed climatology back to a pandas Series for easier manipulation
    climatology_smooth = pd.Series(
        climatology_smooth[:, 1], index=climatology["dayofyear"].values
    )

    climatology_t_smooth = pd.Series(
        climatology_t_smooth[:, 1], index=climatology_t["dayofyear"].values
    )

    # Calculate the daily SMB anomaly by subtracting the climatology from the daily SMB values
    daily_smb_anomaly = (
        specific_mass_balance.values
        - climatology_smooth.reindex(
            specific_mass_balance["TIME"].dt.dayofyear.values
        ).values
    )
    daily_t_anomaly = (
        t_mean.values
        - climatology_t_smooth.reindex(t_mean["TIME"].dt.dayofyear.values).values
    )

    # Prepare the DataFrame for the CSV file
    df = pd.DataFrame(
        {
            "Time": specific_mass_balance["TIME"].values,
            "SMB (m^3)": total_smb_daily.values,
            "Specific Mass Balance (mm WE)": specific_mass_balance.values,
            "SMB Anomaly (mm WE)": daily_smb_anomaly,
            "T2m": t_mean,
            "T2m_anom": daily_t_anomaly,
        }
    )

    # Convert time to the desired format (YYYY-MM-DD)
    df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%Y-%m-%d")

    # Save the DataFrame to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "FI_mar_smb.csv")
    df.to_csv(output_file, index=False)

    print(f"CSV file saved to: {output_file}")


def merge_mar_cluster(smb_csv_path, cluster_csv_path, output_dir):
    """
    Merge the MAR SMB and cluster dataset on the date column and save the result to a CSV file.
    Parameters:
        smb_csv_path (str): Path to the MAR SMB CSV file.
        cluster_csv_path (str): Path to the cluster CSV file.
        output_dir (str): Directory to save the merged CSV file.
    """

    smb_df = pd.read_csv(smb_csv_path, parse_dates=["Time"])
    cluster_df = pd.read_csv(cluster_csv_path, parse_dates=["Date"])

    # Merge datasets on the date column
    merged_df = pd.merge(
        smb_df.rename(columns={"Time": "Date"}),  # Align column names for merging
        cluster_df,
        on="Date",
        how="inner",
    )

    # Save the merged dataset to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mar_cluster_merged.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to: {output_file}")


def mar_to_elev_bins(file_path_mar, output_dir, summer=True):
    # Load dataset
    ds = xr.open_dataset(file_path_mar)

    # Filter for SECTOR = 1
    ds = ds.sel(SECTOR=1)

    # Select only the summer period (June, July, August)
    if summer:
        summer_months = [6, 7, 8]
        ds = ds.sel(TIME=ds["TIME"].dt.month.isin(summer_months))

    # Define elevation bins and labels
    elevation_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    elevation_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Extract surface height (SH) and digitize into elevation bins
    sh = ds["SH"].values
    digitized_data = np.digitize(sh, bins=elevation_bins)
    print("digitized")
    # Map elevation bins to labels
    elevation_bins_array = np.empty_like(digitized_data, dtype=object)
    elevation_bins_array[:] = "Invalid"  # Default for invalid values
    valid_mask = ~np.isnan(sh)  # Mask to handle valid SH values
    elevation_bins_array[valid_mask] = np.array(elevation_labels)[
        digitized_data[valid_mask] - 1
    ]
    print("elev_bins_array")
    # Add elevation bins to dataset
    ds["elev_bin"] = (("TIME", "y", "x"), elevation_bins_array)
    print("bins added")
    # Extract TTZ at ZTQLEV = 2.0 (temperature at 2m)
    ttz_2m = ds["TTZ"].sel(ZTQLEV=2.0)

    # Define cell area (10x10 km = 100 km² per cell)
    cell_area_km2 = 100

    # Group data by elevation bins and calculate means, counts, and area for each time step
    data = []
    for time in ds["TIME"].values:
        subset = ds.sel(TIME=time)
        ttz_2m_time = ttz_2m.sel(
            TIME=time
        )  # Extract temperature for the current time step
        print(time)

        # Define the weight using the MSK variable (assuming it's in percent)
        weight = subset["MSK"] / 100

        for elev_bin in elevation_labels:
            mask = subset["elev_bin"] == elev_bin

            # Calculate the total weight for the current bin
            total_weight = weight.where(mask).sum(dim=("y", "x"))

            # Weighted SMB mean
            weighted_sum_smb = (subset["SMB"] * weight).where(mask).sum(dim=("y", "x"))
            smb_weighted_mean = (
                (weighted_sum_smb / total_weight).item()
                if total_weight != 0
                else np.nan
            )

            # Weighted SHSN2 mean
            weighted_sum_shsn2 = (
                (subset["SHSN2"] * weight).where(mask).sum(dim=("y", "x"))
            )
            shsn2_weighted_mean = (
                (weighted_sum_shsn2 / total_weight).item()
                if total_weight != 0
                else np.nan
            )

            # Weighted TTZ (temperature) mean for 2m temperature
            weighted_sum_ttz = (ttz_2m_time * weight).where(mask).sum(dim=("y", "x"))
            ttz_weighted_mean = (
                (weighted_sum_ttz / total_weight).item()
                if total_weight != 0
                else np.nan
            )

            # Weighted mean elevation (SH)
            weighted_sum_sh = (subset["SH"] * weight).where(mask).sum(dim=("y", "x"))
            mean_elevation_weighted = (
                (weighted_sum_sh / total_weight).item() if total_weight != 0 else np.nan
            )

            # Calculate total area for the current bin
            weighted_cell_fraction = weight.where(mask).sum(dim=("y", "x")).item()
            total_area_km2 = weighted_cell_fraction * cell_area_km2

            # Append data for the current time and elevation bin
            data.append(
                [
                    pd.Timestamp(time).strftime("%Y-%m-%d"),
                    elev_bin,
                    smb_weighted_mean,
                    shsn2_weighted_mean,
                    ttz_weighted_mean,
                    mean_elevation_weighted,
                    weighted_cell_fraction,
                    total_area_km2,
                ]
            )

    # Create DataFrame
    columns = [
        "Time",
        "elev_bin",
        "SMB",  # Mean SMB (Specific) in mm we over that part of the glacier
        "Snow_above_ice",
        "Temp",
        "Mean_Elevation",
        "Data_Points",
        "Total_Area_km2",
    ]
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to CSV
    os.makedirs(output_dir, exist_ok=True)
    if summer:
        file_name = "mar_1991_2024_elev_bins_summer.csv"
    else:
        file_name = "mar_1991_2024_elev_10_bins.csv"
    output_file = os.path.join(output_dir, file_name)
    df.to_csv(output_file, index=False)

    # Display a sample of the results
    print(df.head())


def summarize_ela_AAR(input_file, output_dir):
    """
    Load MAR SMB data from a CSV file, compute the equilibrium line altitude (ELA)
    for three time periods (using zero-crossing of SMB) and the fractional
    Accumulation Area Ratio (AAR) based on 100 m elevation bins. The result is exported as a CSV.

    Parameters:
      input_file (str): Path to the input CSV file containing columns:
                        Time, SMB, Total_Area_km2, Mean_Elevation, elev_bin, etc.
      output_csv (str): Path to the output CSV file to save the summary.
    """
    # ---------------------- Load and Preprocess Data ----------------------
    df = pd.read_csv(input_file)

    # Convert 'Time' to datetime and extract Year
    df["Time"] = pd.to_datetime(df["Time"])
    df["Year"] = df["Time"].dt.year.astype(int)

    # Compute absolute SMB contribution (in mm·km²)
    df["SMB_abs_mm_km2"] = df["SMB"] * df["Total_Area_km2"]

    # Group by Year and elevation bin
    df_grouped = (
        df.groupby(["Year", "elev_bin"])
        .agg(
            SMB_abs=("SMB_abs_mm_km2", "sum"),  # Sum SMB contributions in that bin
            Total_Area_km2=("Total_Area_km2", "mean"),  # Mean area per elev_bin
            Mean_Elevation=("Mean_Elevation", "mean"),  # Mean elevation of the bin
        )
        .reset_index()
    )

    # Define bin edges and representative labels for the three time periods
    bins = [1990, 2002, 2013, 2024]  # right-inclusive edges
    labels = [1991, 2003, 2014]  # representative year labels

    # Create "Year_Bin" column by binning the "Year" column in df_grouped
    df_grouped["Year_Bin"] = pd.cut(
        df_grouped["Year"], bins=bins, labels=labels, right=True
    )
    df_grouped["Year_Bin"] = df_grouped["Year_Bin"].astype(int)
    # Group again by Year_Bin and elev_bin to get mean values (if there are multiple records per bin)
    df_bin_means = df_grouped.groupby(["Year_Bin", "elev_bin"]).mean().reset_index()
    # Convert aggregated SMB_abs from mm·km² to Gigatonnes (Gt)
    df_bin_means["SMB_abs_Gt"] = df_bin_means["SMB_abs"] * 1e-6

    def find_smb_zero_crossing_bin(df):
        """
        For each time period (Year_Bin), determine the equilibrium line altitude (ELA)
        where SMB (in Gt) crosses zero between adjacent elevation bins.
        If no crossing is found, defaults to 925 m.
        """
        zero_crossings = []
        for period, group in df.groupby("Year_Bin"):
            group = group.sort_values("Mean_Elevation")
            smb_values = group[["Mean_Elevation", "SMB_abs_Gt"]].values
            crossings = []
            for i in range(len(smb_values) - 1):
                # If sign change occurs between adjacent bins
                if smb_values[i, 1] * smb_values[i + 1, 1] < 0:
                    x1, y1 = smb_values[i]
                    x2, y2 = smb_values[i + 1]
                    # Linear interpolation to estimate the crossing elevation:
                    elev_crossing = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                    crossings.append(elev_crossing)
            abs_SMB = group["SMB_abs_Gt"].sum()
            if crossings:
                zero_crossings.append(
                    {"Year_Bin": period, "ELA": min(crossings), "SMB": abs_SMB}
                )
            else:
                # Print a warning if no crossing is found and default to 925 m
                print(
                    f"Attention: No ela found for time period {period}. Using default ELA=925 m."
                )
                zero_crossings.append({"Year_Bin": period, "ELA": 925, "SMB": abs_SMB})
        return pd.DataFrame(zero_crossings)

    ela_time_periods = find_smb_zero_crossing_bin(df_bin_means)

    # Create a table with unique elevation bin info (assuming each bin spans 100 m)
    df_unique = df_grouped[
        ["elev_bin", "Mean_Elevation", "Total_Area_km2"]
    ].drop_duplicates()

    def compute_AAR_for_ela(ELA, df_unique):
        """
        Given an equilibrium line altitude (ELA) and a DataFrame with unique elevation bin info,
        compute the accumulation and ablation areas assuming each bin spans 100 m and area is uniformly distributed.
        Returns (accum_area, ablation_area).
        """
        accum_area = 0.0
        ablation_area = 0.0
        for _, row in df_unique.iterrows():
            elev_bin = row["elev_bin"]
            area = row["Total_Area_km2"]
            lower_bound = elev_bin * 100.0
            upper_bound = (elev_bin + 1) * 100.0
            if ELA <= lower_bound:
                accum_fraction = 1.0
                ablation_fraction = 0.0
            elif ELA >= upper_bound:
                accum_fraction = 0.0
                ablation_fraction = 1.0
            else:
                # ELA falls within this bin: split fractionally
                accum_fraction = (upper_bound - ELA) / 100.0
                ablation_fraction = (ELA - lower_bound) / 100.0
            accum_area += accum_fraction * area
            ablation_area += ablation_fraction * area
        return accum_area, ablation_area

    AAR_list = []
    for idx, row in ela_time_periods.iterrows():
        ELA = row["ELA"]
        accum_area, ablation_area = compute_AAR_for_ela(ELA, df_unique)
        total_area = accum_area + ablation_area
        AAR = accum_area / total_area if total_area > 0 else np.nan
        AAR_list.append(AAR)

    ela_time_periods["AAR"] = AAR_list

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "aar_ela_summary.csv")
    ela_time_periods.to_csv(output_csv, index=False)
    print(f"AAR Summary exported to: {output_csv}")
    return ela_time_periods
