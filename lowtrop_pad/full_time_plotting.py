import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.stats import linregress
from matplotlib import cm
import pymannkendall as mk
from scipy.ndimage import uniform_filter1d
from matplotlib.colors import TwoSlopeNorm  # If you want center set to 0
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from cycler import cycler
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines


### SETTING PLOT DEFAULT SETTINGS
if True:
    # Set Arial font family for all elements
    plt.rc("font", family="sans-serif")
    plt.rcParams["font.sans-serif"] = ["Arial"]

    # Function to reduce saturation
    def adjust_saturation(color, sat_factor):
        rgb = mcolors.to_rgb(color)  # Convert hex to RGB
        hsv = mcolors.rgb_to_hsv(rgb)  # Convert RGB to HSV
        hsv = (hsv[0], hsv[1] * sat_factor, hsv[2])  # Reduce saturation
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))  # Convert back to hex

    # Adjust the saturation of the first color
    green = "green"
    red = "red"
    adjusted_green = adjust_saturation(green, 0.8)  # Reduce saturation by X%
    adjusted_red = adjust_saturation(red, 0.8)  # Reduce saturation by X%
    # Define new cycler for categorical colors for cluster data
    default_cycler = cycler(
        color=["#AF58BA", "#009ADE", adjusted_green, "#FFC61E", adjusted_red]
    ) + cycler(linestyle=["-", "--", (0, (1, 1)), "-.", (0, (3, 1, 1, 1, 1, 1))])

    # Apply settings
    plt.rc("lines", linewidth=2.5)
    plt.rc("axes", prop_cycle=default_cycler)
    plt.rcParams["hatch.linewidth"] = 1.5
    plt.rcParams["hatch.color"] = "yellow"


def plot_monthly(input_directory, output_path, vmin=-2.5, vmax=2.5, ylim=500):
    """
    This function generates heatmaps of monthly anomalies (June, July, August) and yearly mean anomalies
    for all CSV files in the input directory and saves the plots to the output path.
    """
    # Get all the CSV files in the directory
    input_files = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".csv")
    ]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over the input files
    for idx, file in enumerate(input_files):
        if "anomal" in file:
            title = "Anomalies"
            label = "Temperature Anomaly (°C)"
            file_ending = "monthly_anomalies"
        elif "differences" in file:
            title = "Differences"
            label = "Temperature Difference (°C)"
            file_ending = "monthly_differences"
        else:
            title = "Temperature Profiles"
            label = "Temperature (°C)"
            file_ending = "monthly_profiles"
        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)

        # Extract the year and month from the index
        df["year"] = df.index.year
        df["month"] = df.index.month

        # Calculate monthly mean anomalies (mean across all heights)
        monthly_mean = df.groupby(["year", "month"]).mean()

        # Pivot table to get the year as columns and the month as rows
        monthly_pivot = monthly_mean.unstack(level="month")

        # Ensure that the pivot data has the correct shape for plotting
        monthly_pivot = (
            monthly_pivot.T
        )  # Transpose to have months as rows and years as columns

        # Filter for June (6), July (7), August (8)
        months_to_plot = [6, 7, 8]

        # Calculate yearly mean over June, July, and August
        yearly_mean = (
            monthly_mean[
                monthly_mean.index.get_level_values("month").isin(months_to_plot)
            ]
            .groupby("year")
            .mean()
        )
        yearly_mean = yearly_mean.T

        # Create the figure for the subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

        # Plot for June (6)
        ax = axes[0, 0]
        month_data = monthly_pivot.xs(6, level="month")
        # Define y-tick positions and labels
        tick_positions = np.arange(
            0, ylim + 1, step=int(ylim / 10)
        )  # Modify the step as needed
        tick_labels = month_data.index[tick_positions]
        sns.heatmap(
            month_data,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": label},
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"June {title}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Height Above Ground (m)")
        ax.set_yticks(tick_positions)  # Set Y-axis ticks from height
        ax.set_yticklabels(
            tick_labels
        )  # Ensure that y-tick labels are from height values
        ax.invert_yaxis()  # Flip the y-axis
        ax.set_ylim(0, ylim)

        # Plot for July (7)
        ax = axes[0, 1]
        month_data = monthly_pivot.xs(7, level="month")
        tick_positions = np.arange(
            0, ylim + 1, step=int(ylim / 10)
        )  # Modify the step as needed
        tick_labels = month_data.index[tick_positions]
        sns.heatmap(
            month_data,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": label},
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"July {title}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Height Above Ground (m)")
        ax.set_yticks(tick_positions)  # Set Y-axis ticks from height
        ax.set_yticklabels(
            tick_labels
        )  # Ensure that y-tick labels are from height values
        ax.invert_yaxis()  # Flip the y-axis
        ax.set_ylim(0, ylim)

        # Plot for August (8)
        ax = axes[1, 0]
        month_data = monthly_pivot.xs(8, level="month")
        tick_positions = np.arange(
            0, ylim + 1, step=int(ylim / 10)
        )  # Modify the step as needed
        tick_labels = month_data.index[tick_positions]
        sns.heatmap(
            month_data,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": label},
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"August {title}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Height Above Ground (m)")
        ax.set_yticks(tick_positions)  # Set Y-axis ticks from height
        ax.set_yticklabels(
            tick_labels
        )  # Ensure that y-tick labels are from height values
        ax.invert_yaxis()  # Flip the y-axis
        ax.set_ylim(0, ylim)

        # Plot for yearly mean over June, July, and August
        ax = axes[1, 1]
        tick_positions = np.arange(
            0, ylim + 1, step=int(ylim / 10)
        )  # Modify the step as needed
        tick_labels = yearly_mean.index[tick_positions]
        sns.heatmap(
            yearly_mean,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": label},
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Summer Mean (June-August) {title}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Height Above Ground (m)")
        ax.set_yticks(tick_positions)  # Set Y-axis ticks from height
        ax.set_yticklabels(
            tick_labels
        )  # Ensure that y-tick labels are from height values
        ax.invert_yaxis()  # Flip the y-axis
        ax.set_ylim(0, ylim)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        output_file = os.path.join(
            output_path, f"{os.path.basename(file).split('.')[0]}_{file_ending}.png"
        )
        plt.savefig(output_file, dpi=300)

        # Close the plot to free up memory
        plt.close(fig)
        print("Plot saved to:", output_file)


def process_and_plot_trends(input_directory, output_directory, method="linear"):
    """
    Processes trend calculations for temperature anomalies or differences over time.
    The user can select the method for trend estimation:
        - 'linear': Standard linear regression (OLS)
        - 'mannkendall': Mann-Kendall trend test
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # File pattern for CSV files in the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    files = glob.glob(file_pattern)

    # Initialize a dictionary to store results by surface type and category
    all_trend_results = {"June": {}, "July": {}, "August": {}, "Summer": {}}

    # Process each file
    for file in files:
        if "differences" in file:
            title = "Differences"
        else:
            title = "Anomalies"

        # Extract the surface name from the filename
        surface = os.path.basename(file).split("_")[0]

        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)

        # Filter data by month and calculate yearly means
        monthly_data = {
            "June": df[df.index.month == 6],
            "July": df[df.index.month == 7],
            "August": df[df.index.month == 8],
            "Summer": df[df.index.month.isin([6, 7, 8])],
        }

        # Calculate trends for each month and summer season
        for category, monthly_df in monthly_data.items():
            # Calculate yearly means for each height in the DataFrame
            yearly_means = monthly_df.resample("YE").mean()
            trend_results = []

            # Calculate the trend for each height using yearly means
            for height in yearly_means.columns:
                x = np.arange(len(yearly_means))  # Create a sequential array for years
                y = yearly_means[height].values

                # Apply the selected method
                if method == "linear":
                    # Standard linear regression (OLS)
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    trend_results.append(
                        {
                            "height": int(height),
                            "coefficient": slope,
                            "p_value": p_value,
                        }
                    )

                elif method == "mannkendall":
                    # Mann-Kendall trend test (non-parametric)
                    result = mk.original_test(y)
                    trend_results.append(
                        {
                            "height": int(height),
                            "coefficient": result.slope,  # The slope from Mann-Kendall test
                            "p_value": result.p,
                        }
                    )

            # Store results by surface and category
            trend_results_df = pd.DataFrame(trend_results)
            trend_results_df.sort_values(by="height", inplace=True)
            if surface not in all_trend_results[category]:
                all_trend_results[category][surface] = trend_results_df

    # Plot the coefficients for each surface type in 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    categories = ["June", "July", "August", "Summer"]

    for ax, category in zip(axes.flatten(), categories):
        for surface, trend_df in all_trend_results[category].items():
            # Plot the coefficient (slope) vs height
            ax.plot(
                trend_df["coefficient"],
                trend_df["height"],
                linestyle="-",
                linewidth=3,
                alpha=0.7,
                label=surface,
            )

        # Set title and labels for each subplot
        ax.set_title(f"Linear Trend of Temperature {title} - {category}")
        ax.set_xlabel("Slope of Linear Trend (°C/Year)")
        ax.set_ylabel("Height Above Ground (m)")
        ax.legend(title="Surface")
        ax.grid()

    # Finalize plot layout
    plt.ylim(-5, 500)
    plt.tight_layout()

    # Save the plot to the output directory
    plot_file = os.path.join(output_directory, f"{title}_{method}_trends_plot.png")
    plt.savefig(plot_file, dpi=300)

    # Write the trend results to an HTML file
    all_trend_results_df = {}
    for category, surface_results in all_trend_results.items():
        for surface, trend_df in surface_results.items():
            trend_df["Time"] = category
            trend_df["Surface"] = surface
            if category not in all_trend_results_df:
                all_trend_results_df[category] = trend_df
            else:
                all_trend_results_df[category] = pd.concat(
                    [all_trend_results_df[category], trend_df], ignore_index=True
                )

    # Concatenate all trend results into a single DataFrame
    all_trends_combined = pd.concat(all_trend_results_df.values(), ignore_index=True)
    html_file = os.path.join(output_directory, f"{title}_{method}_trends.html")
    all_trends_combined.to_html(html_file, index=False)

    print(f"Plot saved to: {plot_file}")
    print(f"Trend results saved to HTML file: {html_file}")


def plot_slope_over_doy(
    input_directory, output_directory, height_level, method="linear", smooth_window=7
):
    """
    Calculate and plot the slope of temperature differences/anomalies at height = 0 over the day of the year
    for the summer months (June, July, August). The user can select the method for trend estimation.
    Smoothing is applied to the slopes using a moving average filter.

    Parameters:
    - input_directory: Path to the directory containing CSV files
    - method: Trend estimation method ('linear', 'mannkendall', 'lad')
    - smooth_window: Window size for smoothing (default is 7 days)
    """
    # create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Get all CSV files in the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    files = glob.glob(file_pattern)

    # Initialize a dictionary to store slope results by year
    all_slope_results = []

    # Process each file
    for file in files:
        # Extract the surface name from the filename
        surface = os.path.basename(file).split("_")[0]

        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)

        # Filter data for the summer months (June, July, August)
        summer_df = df[df.index.month.isin([6, 7, 8])]

        # Calculate the slope for height = 0 (ground level) for each day of the year
        slope_values = []

        # Loop over each day of the year (1 to 366)
        for doy in range(1, 367):
            # Extract the data for the specific day of year (doy)
            doy_data = summer_df[summer_df.index.dayofyear == doy]

            # Check if there is more than one data point for this day of the year
            if len(doy_data) > 1:
                # Apply the selected method to calculate the slope for this day of the year
                x = np.arange(len(doy_data))  # Time series (index as time variable)
                y = doy_data.iloc[
                    :, height_level
                ].values  # Get the ground level data (height = 0)

                if method == "linear":
                    # Standard linear regression (OLS)
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    slope_values.append((doy, slope))

                elif method == "mannkendall":
                    # Apply Mann-Kendall test (non-parametric) for trend estimation
                    from pymannkendall import mk

                    result = mk.original_test(y)
                    slope_values.append((doy, result.slope))

                elif method == "lad":
                    # Least Absolute Deviations (LAD) regression (RANSAC regression)
                    from sklearn.linear_model import RANSACRegressor

                    model = RANSACRegressor()
                    model.fit(x.reshape(-1, 1), y)
                    slope = model.estimator_.coef_[0]  # Get the slope from RANSAC
                    slope_values.append((doy, slope))

        # Store results for this surface
        slope_results_df = pd.DataFrame(slope_values, columns=["Day_of_Year", "Slope"])
        slope_results_df["Surface"] = surface
        all_slope_results.append(slope_results_df)

    # Concatenate all slope results into a single DataFrame
    all_slope_df = pd.concat(all_slope_results, ignore_index=True)

    # Apply smoothing to the slope values using a moving average filter
    all_slope_df["Smoothed_Slope"] = all_slope_df.groupby("Surface")["Slope"].transform(
        lambda x: uniform_filter1d(
            x, size=smooth_window
        )  # Apply smoothing for each surface
    )

    # Plot the smoothed slope over the day of the year
    plt.figure(figsize=(12, 6))
    for surface, surface_data in all_slope_df.groupby("Surface"):
        plt.plot(
            surface_data["Day_of_Year"],
            surface_data["Smoothed_Slope"],
            label=surface,
            linewidth=2,
        )

    plt.title(
        f"Smoothed Slope of Temperature Trend at Ground Level (Height = {height_level}) over Day of Year"
    )
    plt.xlabel("Day of Year")
    plt.ylabel(f"Smoothed Slope ({method})")
    plt.legend(title="Surface")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_directory, f"height_{height_level}_slope_over_doy_{method}.png"
        ),
        dpi=300,
    )
    print(f"Plot saved to: {output_directory}")


def plot_snow_trends(
    input_directory,
    snow_fraction_file,
    output_path,
    method="linear",
    vmin=-0.5,
    vmax=0.5,
):
    """
    This function generates heatmaps of the slope (Mann-Kendall or Linear Regression)
    of temperature differences (surfaces) or anomalies over the summer months (June, July, August)
    for all CSV files in the input directory and saves the plots to the output directory.
    The snow fraction (mean + trend) is also included in the plot.
    Parameters:
    - input_directory: Path to the directory containing CSV files
    - snow_fraction_file: Path to the snow fraction CSV file
    - output_path: Path to the output directory
    - method: Trend estimation method ('linear', 'mannkendall')
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load snow fraction data
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year

    # Calculate the typical snow fraction for each DOY
    typical_snow_fraction = snow_df.groupby("DOY")["snow_fraction"].mean()

    # Apply smoothing to the typical snow fraction
    typical_snow_fraction_smoothed = gaussian_filter(typical_snow_fraction, sigma=1)

    # Create a DataFrame from the smoothed typical values
    typical_snow_fraction_df = pd.DataFrame(
        {
            "DOY": typical_snow_fraction.index,
            "snow_fraction_typical": typical_snow_fraction_smoothed,
        }
    )

    # Merge smoothed typical values with the original DataFrame
    snow_df = snow_df.merge(typical_snow_fraction_df, on="DOY", how="left")

    # Calculate anomalies (actual - typical for the same DOY)
    snow_df["anomaly"] = snow_df["snow_fraction"] - snow_df["snow_fraction_typical"]

    # Calculate the slope of snow fraction anomaly for each DOY
    snow_slope = []
    for doy in snow_df["DOY"].unique():
        # Skip last DOY if it's a leap day (e.g., DOY=244)
        if doy == snow_df["DOY"].max():
            continue

        subset = snow_df[snow_df["DOY"] == doy]
        if subset.shape[0] > 1:  # Ensure there are enough data points
            if method == "linear":
                # Perform linear regression
                slope, _, _, _, _ = linregress(
                    subset["time"].dt.year, subset["anomaly"]
                )
                snow_slope.append(slope)
            elif method == "mannkendall":
                # Apply Mann-Kendall test for trend estimation
                result = mk.original_test(subset["anomaly"])
                snow_slope.append(
                    result.slope
                )  # Use the slope from the Mann-Kendall test
        else:
            snow_slope.append(np.nan)

    # Smooth the slope values
    snow_slope_smoothed = gaussian_filter(snow_slope, sigma=2)

    # Get all CSV files in the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    files = glob.glob(file_pattern)

    # Process each file
    for file in files:
        # Extract the surface name from the filename
        surface = os.path.basename(file).split("_")[0]

        # Load the data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)

        # Prepare the data for slope calculation
        df["Year"] = df.index.year
        df["DOY"] = df.index.day_of_year

        # Prepare empty DataFrames for slope and p-value
        slope_df = pd.DataFrame(
            index=np.unique(df["DOY"]), columns=df.columns[:-2]
        )  # Exclude 'Year' and 'DOY'
        pval_df = pd.DataFrame(index=np.unique(df["DOY"]), columns=df.columns[:-2])

        # Calculate slope and p-value for each DOY and column
        for doy in slope_df.index:
            subset = df[df["DOY"] == doy]
            for col in slope_df.columns:
                if subset.shape[0] > 1:  # Ensure enough points for regression
                    if method == "linear":
                        # Linear regression
                        slope, _, _, p_value, _ = linregress(
                            subset["Year"], subset[col]
                        )
                        slope_df.at[doy, col] = slope
                        pval_df.at[doy, col] = p_value
                    elif method == "mannkendall":
                        # Mann-Kendall test for trend estimation
                        result = mk.original_test(subset[col])
                        slope_df.at[doy, col] = result.slope
                        pval_df.at[doy, col] = result.p

        # Smooth the slope data
        slope_smoothed = gaussian_filter(slope_df.astype(float), sigma=2)

        # Create the heatmap plot with slopes and snow fraction
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot the heatmap
        extent = [
            snow_df["DOY"].min(),
            snow_df["DOY"].max(),
            0,
            slope_smoothed.shape[1],
        ]
        im = ax1.imshow(
            slope_smoothed.T * 10,
            aspect="auto",
            cmap="coolwarm",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        # Vertical colorbar on the left
        cbar = plt.colorbar(im, ax=ax1, orientation="vertical", shrink=0.7, pad=0.22)
        if "difference" in file:
            cbar.set_label("Slope of Temp. Difference (°C per Year)")
        else:
            cbar.set_label("Slope of Temp. Anomaly (°C per Year)")
        ax1.set_xlabel("Day of Year (June-August)")
        ax1.set_ylabel("Height above ground (m)")
        ax1.set_ylim(0, 100)  # Adjust as needed

        # Plot the smoothed snow fraction on a secondary y-axis (right, adjacent to the heatmap)
        ax2 = ax1.twinx()
        # Prepare DOY values without leap day
        doy_values = np.unique(snow_df["DOY"])
        ax2.plot(
            doy_values,
            typical_snow_fraction_smoothed * 100,
            color="darkgreen",
            label="Snow Fraction (%)",
            linewidth=2,
        )  # Scaled for visibility
        ax2.set_ylabel("Snow Fraction (%)", color="darkgreen")
        ax2.tick_params(axis="y", labelcolor="darkgreen")

        # Create a third y-axis for snow fraction trend slopes (offset outward)
        doy_values_leap = doy_values[doy_values != 244]  # Remove leap day if present
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 65))  # Offset the third axis
        ax3.plot(
            doy_values_leap,
            snow_slope_smoothed * 1000,
            color="limegreen",
            label="Snow Fraction Slope",
            linewidth=2,
        )  # conversion to %/Decade
        ax3.axhline(
            0, color="limegreen", linestyle="--", linewidth=2, label="Slope = 0"
        )  # Add a horizontal line
        ax3.set_ylabel("Slope of Snow Fraction (% per decade)", color="limegreen")
        ax3.tick_params(axis="y", labelcolor="limegreen")

        # Save the plot
        output_file = os.path.join(output_path, f"{surface}_plot_{method}.png")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def plot_monthly_differences(input_directory, output_directory, ylim=500):
    """
    Generates a 4-subplot figure of temperature profiles (June, July, August, and Summer Mean)
    for each CSV file in the input directory and saves it to the output directory.
    """
    # Get all CSV files in the directory
    input_files = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".csv")
    ]

    os.makedirs(output_directory, exist_ok=True)
    for file in input_files:
        title = "Differences"
        label = "Temperature Difference (°C)"
        surface = os.path.basename(file).split("_")[0]

        # Load data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)

        # Add 'year' and 'month' columns
        df["year"] = df.index.year
        df["month"] = df.index.month

        # Calculate monthly means and summer mean for June, July, August
        monthly_mean = df.groupby(["year", "month"]).mean()
        summer_mean = (
            monthly_mean.loc[
                monthly_mean.index.get_level_values("month").isin([6, 7, 8])
            ]
            .groupby("year")
            .mean()
            .T
        )

        # Setup figure with 4 subplots (3 for months and 1 for summer mean)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.patch.set_facecolor("white")  # Set background to white

        # Loop through months (June, July, August) and plot in subplots
        summer_months = [6, 7, 8]
        month_titles = ["June", "July", "August"]

        for i, month in enumerate(summer_months):
            ax = axes[i // 2, i % 2]
            month_data = monthly_mean.xs(month, level="month").T
            month_data_mean = month_data.mean(axis=1)
            for year in month_data.columns:
                if year == month_data.columns[0]:
                    ax.plot(
                        month_data[year],
                        month_data.index,
                        color="darkgrey",
                        alpha=0.9,
                        linewidth=0.7,
                        label="Yearly Temp. Differences",
                    )
                else:
                    ax.plot(
                        month_data[year],
                        month_data.index,
                        color="darkgrey",
                        alpha=0.9,
                        linewidth=0.7,
                    )
            ax.plot(
                month_data_mean,
                month_data_mean.index,
                color="blue",
                linewidth=2,
                label="Mean Temp. Difference",
            )
            ax.set_title(f"{month_titles[i]} {title}")
            ax.set_xlabel(label)
            ax.set_ylabel("Height Above Ground (m)")
            ax.invert_yaxis()  # Invert y-axis to have height increasing upwards
            # Adjust y-ticks for clarity
            ax.set_ylim(-5, ylim)
            ax.set_xlim(-0.6, 4)
            ax.set_yticks(range(0, ylim + 1, ylim // 10))  # Fewer y-ticks for clarity
            ax.grid()
            ax.legend()

        # Plot Summer Mean
        ax = axes[1, 1]
        summer_mean_mean = summer_mean.mean(axis=1)
        for year in summer_mean.columns:
            if year == month_data.columns[0]:
                ax.plot(
                    month_data[year],
                    month_data.index,
                    color="darkgrey",
                    alpha=0.9,
                    linewidth=0.7,
                    label="Yearly Temp. Differences",
                )
            else:
                ax.plot(
                    month_data[year],
                    month_data.index,
                    color="darkgrey",
                    alpha=0.9,
                    linewidth=0.7,
                )

        ax.plot(
            summer_mean_mean,
            summer_mean_mean.index,
            color="blue",
            linewidth=2,
            label="Mean Temp. Difference",
        )
        ax.set_title("Summer Mean (June-August)")
        ax.set_xlabel(label)
        ax.set_ylabel("Height Above Ground (m)")
        ax.invert_yaxis()
        ax.set_ylim(-5, ylim)
        ax.set_xlim(-0.6, 4)
        ax.set_yticks(range(0, ylim + 1, ylim // 10))
        ax.grid()
        ax.legend()

        # Adjust layout and save the plot
        plt.tight_layout()
        output_file = os.path.join(output_directory, f"{surface}_mean_differences.png")
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print("Plot saved to:", output_file)


def plot_climatology_temperature_difference(
    input_directory,
    snow_fraction_file,
    output_path,
    vmin=-2,
    vmax=2,
    ylim=500,
    smooth_window=5,
):
    """
    This function generates heatmaps of climatology (mean temperature differences) over the summer months (June, July, August)
    for all CSV files in the input directory, adds climatology of snow fraction, and saves the plots to the output path.

    Parameters:
    - input_directory: Path to the directory containing CSV files
    - snow_fraction_file: Path to the CSV file containing snow fraction data
    - output_path: Path to the directory where plots will be saved
    - vmin, vmax: Color scale for the heatmap
    - ylim: Maximum height (y-axis limit)
    - smooth_window: Window size for the smoothing of temperature differences
    """
    # Get all the CSV files in the directory
    input_files = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".csv")
    ]

    # Load snow fraction data
    df_snow = pd.read_csv(
        snow_fraction_file, index_col="time", parse_dates=True, dayfirst=True
    )

    # Ensure the index is datetime and filter to include only summer months (June, July, August)
    df_snow.index = pd.to_datetime(df_snow.index)
    df_snow_summer = df_snow[df_snow.index.month.isin([6, 7, 8])]

    # Group snow fraction by day of year and calculate the climatology (mean over all years for each day)
    df_snow_summer["doy"] = df_snow_summer.index.dayofyear
    df_daily_mean = df_snow_summer.groupby("doy")["snow_fraction"].mean()

    # Smooth the snow fraction climatology using a moving average (smooth over the days of the year)
    df_daily_mean_smoothed = uniform_filter1d(df_daily_mean.values, size=smooth_window)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over the input files
    for idx, file in enumerate(input_files):
        # Load the temperature difference data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)

        # Ensure the index is datetime
        df.index = pd.to_datetime(df.index)

        # Extract only summer months (June, July, August)
        df_summer = df[df.index.month.isin([6, 7, 8])]

        # Group by day of year and calculate the climatology (mean over all years for each day and height)
        climatology = df_summer.groupby(df_summer.index.dayofyear).mean()

        # Smooth the climatology using a moving average (smooth over the days of the year)
        climatology_smoothed = climatology.apply(
            lambda x: uniform_filter1d(x, size=smooth_window), axis=0
        )

        # Prepare the data for plotting
        X, Y = np.meshgrid(
            climatology_smoothed.index, climatology_smoothed.columns.astype(int)
        )
        Z = climatology_smoothed.values.T  # Transpose to align with X, Y

        # Create a figure for the plot
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot the heatmap of the climatology
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Centered colormap
        c = ax1.pcolormesh(X, Y, Z, cmap="coolwarm", shading="auto", norm=norm)

        cbar = plt.colorbar(
            c, ax=ax1, label="Temperature Difference (°C)", shrink=0.75, pad=0.09
        )
        cbar.ax.tick_params(labelsize=12)  # Adjust the size of colorbar ticks
        cbar.ax.set_ylabel("Temperature Difference (°C)", fontsize=14)  # Larger label

        # Plot the climatology of smoothed snow fraction on the second y-axis (for the correct doy)
        ax2 = ax1.twinx()
        ax2.plot(
            df_daily_mean.index,
            df_daily_mean_smoothed,
            color="darkgreen",
            label="Smoothed Snow Fraction",
            linewidth=2,
        )

        # Add labels and title
        # ax1.set_title("Temperature Differences & Snow Fraction") # Optional title
        ax1.set_xlabel("Day of Year (June-August)", fontsize=14)  # Larger x-label
        ax1.set_ylabel(
            "Height Above Ground (m)", color="black", fontsize=14
        )  # Larger y-label
        ax2.set_ylabel("Snow Fraction", color="darkgreen", fontsize=14)  # Green y-label

        # Adjust y-axis ticks
        ax1.set_yticks(np.arange(0, ylim + 1, step=ylim // 10))
        ax1.set_yticklabels(np.arange(0, ylim + 1, step=ylim // 10), fontsize=12)
        ax2.tick_params(axis="y", labelsize=12)

        # Set limits and grid
        ax1.set_ylim(0, ylim)
        ax2.set_ylim(0, 1)  # Snow fraction between 0 and 1
        ax1.grid(True)

        # Save the plot
        output_file = os.path.join(
            output_path, f"{os.path.basename(file).split('.')[0]}_climatology.png"
        )
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        # Close the plot to free up memory
        plt.close(fig)
        print(f"Plot saved to: {output_file}")


def plot_surface_signature_threshold(
    input_directory, output_directory, thresholds=np.arange(0.4, 1.6, 0.2), ylim=500
):
    """
    Generates a 4-subplot figure showing the ratio of datapoints outside given thresholds
    (0.4 to 1.4 in steps of 0.2) for each CSV file in the input directory.
    Plots are for June, July, August, and the entire summer.
    """
    # Get all CSV files in the directory
    input_files = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".csv")
    ]

    os.makedirs(output_directory, exist_ok=True)
    for file in input_files:
        surface = os.path.basename(file).split("_")[0]

        # Load data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)

        # Add 'year' and 'month' columns for filtering by month
        df["year"] = df.index.year
        df["month"] = df.index.month

        # Setup figure with 4 subplots (June, July, August, and Summer Mean), sharing both X and Y axes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

        # Define colormap
        colors = cm.viridis(
            np.linspace(0, 1, len(thresholds))
        )  # Generate a color for each threshold

        # Loop through months (June, July, August) and plot in subplots
        summer_months = [6, 7, 8]
        month_titles = ["June", "July", "August"]

        for i, month in enumerate(summer_months):
            ax = axes[i // 2, i % 2]
            # Filter data for the specified month
            month_data = df[df["month"] == month].iloc[
                :, :-2
            ]  # Exclude 'year' and 'month' columns

            # Calculate the ratio of values outside each threshold at each height level
            outside_ratios = {}
            for threshold in thresholds:
                # Calculate the ratio of values outside the threshold for each height level (column)
                outside_ratios[threshold] = (np.abs(month_data) > threshold).mean(
                    axis=0
                )

            # Convert dictionary to DataFrame for easier plotting
            outside_ratios_df = pd.DataFrame(outside_ratios)
            outside_ratios_df.index = outside_ratios_df.index.astype(
                float
            )  # Ensure height levels are treated as numbers

            # Plot each threshold's ratios with a sequential color scale
            for j, threshold in enumerate(thresholds):
                ax.plot(
                    outside_ratios_df[threshold],
                    outside_ratios_df.index,
                    color=colors[j],
                    label=f"Threshold {threshold:.1f}°C",
                    alpha=0.8,
                )

            ax.set_title(f"{month_titles[i]}")
            ax.set_xlabel("Ratio of Temp. Differences > Threshold")
            ax.set_ylabel("Height Above Ground (m)")
            ax.invert_yaxis()  # Invert y-axis to have height increasing upwards
            ax.set_ylim(0, ylim)
            ax.set_yticks(range(0, ylim + 1, ylim // 10))  # Fewer y-ticks for clarity
            ax.grid(True)

        # Plot Summer Mean
        ax = axes[1, 1]
        # Filter data for all summer months (June, July, August)
        summer_data = df[df["month"].isin(summer_months)].iloc[:, :-2]

        # Calculate the ratio of values outside each threshold at each height level for summer
        outside_ratios = {}
        for threshold in thresholds:
            outside_ratios[threshold] = (np.abs(summer_data) > threshold).mean(axis=0)

        outside_ratios_df = pd.DataFrame(outside_ratios)
        outside_ratios_df.index = outside_ratios_df.index.astype(float)

        # Plot each threshold's ratios for the summer mean with sequential color scale
        for j, threshold in enumerate(thresholds):
            ax.plot(
                outside_ratios_df[threshold],
                outside_ratios_df.index,
                color=colors[j],
                label=f"Threshold {threshold:.1f}°C",
                alpha=0.8,
            )

        ax.set_title("Summer Mean (June-August)")
        ax.set_xlabel("Ratio of Temp. Differences > Threshold")
        ax.set_ylabel("Height Above Ground (m)")
        ax.invert_yaxis()
        ax.set_ylim(0, ylim)
        ax.set_yticks(range(0, ylim + 1, ylim // 10))
        ax.grid(True)

        # Get the handles and labels from only the second subplot (top-right corner, axes[1, 0])
        handles, labels = axes[
            1, 0
        ].get_legend_handles_labels()  # Get handles and labels from only the top-right plot
        # Position the legend outside this specific subplot (next to it)
        axes[0, 1].legend(
            handles=handles,
            labels=labels,
            fontsize="medium",
            loc="upper left",
            bbox_to_anchor=(0.7, 1),
        )  # Position legend next to the 1,0 plot

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0, 0.96, 1])  # Adjust layout to fit everything
        output_file = os.path.join(
            output_directory, f"{surface}_outside_threshold_ratios.png"
        )
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        print("Plot saved to:", output_file)


def plot_snow_fraction_bins(snow_fraction_file, output_path, doy_list, num_bins=6):
    """
    Plots the yearly mean snow fraction for multiple DOY bins over time with a sequential color scheme.

    Parameters:
        snow_fraction_file (str): Path to the snow fraction CSV file.
        output_path (str): Directory to save the plot.
        doy_list (list[int]): List of DOYs to include in the analysis.
        num_bins (int): Number of bins to divide the DOYs into.
    """
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Load the snow fraction data
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year
    snow_df["Year"] = snow_df["time"].dt.year
    print(snow_df)
    # Filter the data to include only the specified DOYs
    snow_df = snow_df[snow_df["DOY"].isin(doy_list)]

    # Create bins for DOYs
    doy_bins = np.array_split(np.sort(doy_list), num_bins)

    # Initialize a DataFrame to store yearly means for each bin
    yearly_means = pd.DataFrame()

    # Calculate the yearly mean snow fraction for each bin
    for i, bin_doys in enumerate(doy_bins):
        bin_name = f"Bin {i+1} ({bin_doys[0]}-{bin_doys[-1]})"
        bin_data = snow_df[snow_df["DOY"].isin(bin_doys)]
        yearly_mean = bin_data.groupby("Year")["snow_fraction"].mean()
        yearly_means[bin_name] = yearly_mean

    # Define a sequential color map
    cmap = plt.cm.viridis  # Change this to your preferred colormap
    norm = Normalize(vmin=0, vmax=num_bins - 1)
    colors = [cmap(norm(i)) for i in range(num_bins)]

    # Plot the yearly means for each bin
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, column in enumerate(yearly_means.columns):
        ax.plot(
            yearly_means.index,
            yearly_means[column],
            label=column,
            marker="o",
            color=colors[i],
        )

    # Add labels, grid, and title
    ax.set_title("Yearly Mean Snow Fraction for DOY Bins")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Snow Fraction")
    ax.legend(title="DOY Bins", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    # Save the plot
    output_file = os.path.join(output_path, "snow_fraction_bins_colored.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_snow_free_days(file_path, output_dir, thresholds=np.arange(0.07, 0.57, 0.05)):
    """
    Analyze and visualize trends in snow-free days across thresholds for the entire summer.

    Parameters:
        file_path (str): Path to the CSV file containing snow fraction data.
        output_dir (str): Directory where the plot will be saved.
        thresholds (list): List of thresholds for snow-free conditions (default: 0.07 to 0.57, step 0.05).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    snow_df = pd.read_csv(file_path, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year
    snow_df["Year"] = snow_df["time"].dt.year

    # Filter for summer months (June, July, August)
    summer_data = snow_df[
        snow_df["DOY"].between(152, 244)
    ]  # DOY 152 to 244 (June 1 to Aug 31)

    # Initialize storage for slopes and p-values
    results = {"Threshold": [], "Slope": [], "P-value": []}

    # Analyze each threshold
    for threshold in thresholds:
        # Calculate snow-free days per year for the given threshold
        snow_free_days_per_year = (
            summer_data[summer_data["snow_fraction"] < threshold].groupby("Year").size()
        )

        # Perform linear regression
        if (
            len(snow_free_days_per_year) > 1
        ):  # Ensure there is sufficient data for regression
            years = snow_free_days_per_year.index
            snow_free_days = snow_free_days_per_year.values
            slope, intercept, r_value, p_value, std_err = linregress(
                years, snow_free_days
            )
        else:
            slope, p_value = np.nan, np.nan  # Insufficient data for regression

        # Store results
        results["Threshold"].append(threshold)
        results["Slope"].append(slope)
        results["P-value"].append(p_value)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Plotting slopes and p-values
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot slopes on the primary Y-axis
    color = "blue"
    ax1.set_xlabel("Snow-Free Threshold (on snow fraction)")
    ax1.set_ylabel("Slope (days/year)", color=color)
    ax1.plot(
        results_df["Threshold"],
        results_df["Slope"],
        marker="o",
        color=color,
        label="Slope",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    # Plot p-values on the secondary Y-axis
    ax2 = ax1.twinx()
    color = "red"
    ax2.set_ylabel("P-value", color=color)
    ax2.plot(
        results_df["Threshold"],
        results_df["P-value"],
        marker="s",
        linestyle="--",
        color=color,
        label="P-value",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Add a title and legends
    fig.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, "trends_snow_free_days.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to {plot_file}")
    return results_df


def plot_snow_anomaly_trend(file_path, output_dir, frac=0.1):
    """
    Calculate and plot daily snow fraction anomaly over time with LOESS smoothing,
    linear regression, slope, p-value, and a histogram of anomalies.

    Parameters:
        file_path (str): Path to the CSV file containing snow fraction data.
        output_dir (str): Directory where the plot will be saved.
        frac (float): The smoothing parameter for LOESS (default is 0.1).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    snow_df = pd.read_csv(file_path, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year
    snow_df["Year"] = snow_df["time"].dt.year

    # Calculate the typical snow fraction for each day of the year (across all years)
    typical_snow_fraction = snow_df.groupby("DOY")["snow_fraction"].mean()

    # Apply LOESS smoothing to the typical snow fraction
    smoothed_snow_fraction = sm.nonparametric.lowess(
        typical_snow_fraction, typical_snow_fraction.index, frac=frac
    )

    # Create a DataFrame from the smoothed values, ensuring DOY is a column
    smoothed_snow_fraction_df = pd.DataFrame(
        smoothed_snow_fraction, columns=["DOY", "snow_fraction_typical"]
    )

    # Merge smoothed values back with the original dataframe
    snow_df = snow_df.merge(
        smoothed_snow_fraction_df, on="DOY", suffixes=("", "_typical")
    )

    # Calculate the anomaly (difference between observed and smoothed snow fraction)
    snow_df["anomaly"] = snow_df["snow_fraction"] - snow_df["snow_fraction_typical"]

    # Perform linear regression on anomalies
    years = snow_df["Year"].unique()
    anomalies_by_year = snow_df.groupby("Year")["anomaly"].mean()
    slope, intercept, r_value, p_value, std_err = linregress(years, anomalies_by_year)

    # Plotting the anomalies over time with linear regression
    plt.figure(figsize=(6, 4))
    plt.plot(
        anomalies_by_year.index, anomalies_by_year.values, marker="o", label="Anomaly"
    )
    plt.plot(
        years,
        intercept + slope * years,
        linestyle="--",
        color="red",
        label=f"lin. regression (p-value = {p_value:.2f})",
    )

    plt.xlabel("Year")
    plt.ylabel("Snow Fraction Anomaly")
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_file = os.path.join(output_dir, "snow_fraction_anomaly_with_trend.png")
    plt.savefig(plot_file)
    plt.close()

    # Plot histogram of anomalies
    plt.figure(figsize=(10, 6))
    plt.hist(snow_df["anomaly"], bins=30, color="blue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Snow Fraction Anomalies")
    plt.xlabel("Snow Fraction Anomaly")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the histogram
    histogram_file = os.path.join(output_dir, "snow_fraction_anomaly_histogram.png")
    plt.savefig(histogram_file)
    plt.close()

    print(f"Plot saved to {plot_file}")
    print(f"Histogram saved to {histogram_file}")


def plot_snow_profiles(
    snow_fraction_file, tundra_file, output_dir, ylim_max=100, smooth_sigma=None
):
    """
    Plots temperature Anomalies for snow-free and snowy days as heatmaps.

    Parameters:
        snow_fraction_file (str): Path to the snow fraction CSV file.
        tundra_file (str): Path to the tundra temperature profile CSV file.
        output_dir (str): Directory to save the output plot.
        ylim_max (int, optional): Maximum height to display on the y-axis. Default is 100.
        smooth_sigma (float, optional): Sigma value for Gaussian smoothing. Default is None (no smoothing).
    """
    # Step 1: Load snow fraction data
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])

    # Step 2: Load the tundra temperature profile data
    df = pd.read_csv(tundra_file, index_col=0, parse_dates=True, dayfirst=True)
    df.sort_index(inplace=True)

    # Step 3: Merge snow fraction data
    df["time"] = df.index
    merged_df = pd.merge(df, snow_df[["time", "snow_fraction"]], on="time", how="left")

    # Step 4: Define snow-free and snowy days
    merged_df["snow_condition"] = np.where(
        merged_df["snow_fraction"] < 0.3, "snow_free", "snowy"
    )
    merged_df["DOY"] = merged_df["time"].dt.day_of_year

    # Step 5: Calculate mean temperature profiles for each DOY
    snow_free_profiles = {}
    snowy_profiles = {}

    for doy in merged_df["DOY"].unique():
        daily_data = merged_df[merged_df["DOY"] == doy]

        # Calculate mean profile for snow-free days
        snow_free_data = daily_data[daily_data["snow_condition"] == "snow_free"]
        if not snow_free_data.empty:
            snow_free_profiles[doy] = snow_free_data.drop(
                columns=["snow_fraction", "snow_condition", "time", "DOY"]
            ).mean()
        # Calculate mean profile for snowy days
        snowy_data = daily_data[daily_data["snow_condition"] == "snowy"]
        if not snowy_data.empty:
            snowy_profiles[doy] = snowy_data.drop(
                columns=["snow_fraction", "snow_condition", "time", "DOY"]
            ).mean()

    # Initialize matrices with DOY as index
    heights = [float(col) for col in df.columns if col != "time"]
    doys = sorted(merged_df["DOY"].unique())
    snow_free_matrix = pd.DataFrame(index=doys, columns=heights, dtype=float)
    snowy_matrix = pd.DataFrame(index=doys, columns=heights, dtype=float)

    for doy in doys:
        if doy in snow_free_profiles:
            snow_free_matrix.loc[doy] = snow_free_profiles[doy].values
        if doy in snowy_profiles:
            snowy_matrix.loc[doy] = snowy_profiles[doy].values

    # Apply Gaussian smoothing if specified
    if smooth_sigma is not None:
        snow_free_matrix = gaussian_filter(
            snow_free_matrix.fillna(0).values, sigma=smooth_sigma
        )
        snowy_matrix = gaussian_filter(
            snowy_matrix.fillna(0).values, sigma=smooth_sigma
        )
    else:
        snow_free_matrix = snow_free_matrix.values
        snowy_matrix = snowy_matrix.values

    # Step 6: Plot heatmaps
    plt.figure(figsize=(10, 12))

    # Heatmap for snow-free days
    plt.subplot(2, 1, 1)
    plt.imshow(
        snow_free_matrix.T,
        aspect="auto",
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
        vmin=-4,
        vmax=4,
    )
    plt.colorbar(label="Temperature Anomaly (°C)")
    plt.title("Snow-Free Temperature Anomalies")
    plt.xlabel("Day of Year")
    plt.ylabel("Height Above Ground (m)")
    plt.xticks(
        ticks=np.linspace(0, len(doys) - 1, 5),
        labels=np.round(np.linspace(doys[0], doys[-1], 5), 0).astype(int),
    )
    plt.yticks(
        ticks=np.linspace(0, ylim_max, 5),
        labels=np.round(np.linspace(0, ylim_max, 5), 0),
    )
    plt.ylim(0, ylim_max)

    # Heatmap for snowy days
    plt.subplot(2, 1, 2)
    plt.imshow(
        snowy_matrix.T,
        aspect="auto",
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
        vmin=-4,
        vmax=4,
    )
    plt.colorbar(label="Temperature Anomaly (°C)")
    plt.title("Snowy Temperature Anomalies")
    plt.xlabel("Day of Year")
    plt.ylabel("Height Above Ground (m)")
    plt.xticks(
        ticks=np.linspace(0, len(doys) - 1, 5),
        labels=np.round(np.linspace(doys[0], doys[-1], 5), 0).astype(int),
    )
    plt.yticks(
        ticks=np.linspace(0, ylim_max, 5),
        labels=np.round(np.linspace(0, ylim_max, 5), 0),
    )
    plt.ylim(0, ylim_max)

    plt.tight_layout()

    # Step 7: Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(
        output_dir, f"snow_no_snow_t_anomalies_tundra{smooth_sigma}.png"
    )
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")


def plot_average_snow_anomalies(
    snow_fraction_file, tundra_file, output_dir, ylim_max=100
):
    """
    Plots the average temperature profiles/Anomalies for snow-free and snowy conditions.

    Parameters:
        snow_fraction_file (str): Path to the snow fraction CSV file.
        tundra_file (str): Path to the tundra temperature profile CSV file.
        output_dir (str): Directory to save the output plot.
    """
    # Step 1: Load snow fraction data
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])

    # Step 2: Load the tundra temperature profile data
    df = pd.read_csv(tundra_file, index_col=0, parse_dates=True, dayfirst=True)
    df.sort_index(inplace=True)

    # Step 3: Merge snow fraction data
    df["time"] = df.index
    merged_df = pd.merge(df, snow_df[["time", "snow_fraction"]], on="time", how="left")

    # Step 4: Define snow-free and snowy days
    merged_df["snow_condition"] = np.where(
        merged_df["snow_fraction"] < 0.3, "snow_free", "snowy"
    )
    merged_df["DOY"] = merged_df["time"].dt.day_of_year

    # Step 5: Filter profiles based on snow conditions
    snow_free_profiles = merged_df[merged_df["snow_condition"] == "snow_free"]
    snowy_profiles = merged_df[merged_df["snow_condition"] == "snowy"]

    # Drop unnecessary columns and calculate mean profiles
    snow_free_mean_profile = snow_free_profiles.drop(
        columns=["snow_fraction", "snow_condition", "time", "DOY"]
    ).mean()
    snowy_mean_profile = snowy_profiles.drop(
        columns=["snow_fraction", "snow_condition", "time", "DOY"]
    ).mean()

    # Extract heights from column names
    heights = [float(col) for col in snow_free_mean_profile.index]

    # Step 6: Plot average temperature profiles
    plt.figure(figsize=(6, 8))
    plt.plot(snow_free_mean_profile, heights, label="Snow-Free", color="red")
    plt.plot(snowy_mean_profile, heights, label="Snow", color="blue")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)  # Zero line
    plt.xlabel("Temperature Anomaly (°C)")
    plt.ylabel("Height Above Ground (m)")
    plt.legend(loc="upper center")
    plt.grid()
    plt.ylim(0, ylim_max)  # Adjust as needed
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, "avg_snow_no_snow_anomalies_tundra.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")


def km_clusters(
    gradient_file,
    profile_file,
    output_dir,
    n_clusters=4,
    exclude_lowest=100,
    init="k-means++",
    ymin=0,
    ymax=500,
    random_state=None,
    max_iter=500,
    include_single_profiles=False,
):
    """
    Analyze and visualize temperature profiles by clustering.

    Parameters:
        gradient_file (str): Path to the CSV file containing gradient data.
        second_file (str): Path to the second file (e.g., anomaly, resampled, or gradient).
        output_dir (str): Directory to save the output plot.
        n_clusters (int): Number of clusters for KMeans. Default is 4.
        exclude_lowest (int): Number of lowest meters to exclude. Default is 100.
        ymin (int): Minimum y-axis value for the plot. Default is 0.
        ymax (int): Maximum y-axis value for the plot. Default is 500.
        random_state (int): Random state for reproducibility. Default is None.
        max_iter (int): Maximum number of iterations for KMeans. Default is 500.

    Returns:
        pd.DataFrame: DataFrame containing the cluster assignments for each timestamp.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and preprocess gradient data
    grad_df = pd.read_csv(gradient_file, index_col=0, parse_dates=True)
    grad_df_plot = grad_df
    grad_df = grad_df.iloc[
        :, exclude_lowest:
    ]  # Exclude the lowest `exclude_lowest` meters
    gradient_matrix = grad_df.values  # Convert to numpy array

    # Step 2: Apply KMeans clustering
    kmeans = KMeans(
        n_clusters=n_clusters, init=init, random_state=random_state, max_iter=max_iter
    )
    cluster_labels = kmeans.fit_predict(gradient_matrix)

    # Create a DataFrame with cluster assignments
    clusters = pd.DataFrame({"Date": grad_df.index, "Cluster": cluster_labels})
    clusters["Date"] = pd.to_datetime(clusters["Date"], dayfirst=True)

    # start counting clusters at 1
    clusters["Cluster"] = clusters["Cluster"] + 1

    # sort clusters in a meaningful order
    clusters.loc[clusters["Cluster"] == 1, "Cluster"] = 20
    clusters.loc[clusters["Cluster"] == 2, "Cluster"] = 40
    clusters.loc[clusters["Cluster"] == 3, "Cluster"] = 10
    clusters.loc[clusters["Cluster"] == 4, "Cluster"] = 30
    clusters.loc[clusters["Cluster"] == 5, "Cluster"] = 50
    clusters["Cluster"] = (clusters["Cluster"] / 10).astype(int)

    # Step 3: Load and preprocess the second file
    second_df = pd.read_csv(profile_file, index_col=0, parse_dates=True)
    second_df.index = pd.to_datetime(second_df.index, format="%d.%m.%Y")
    clusters["Date"] = pd.to_datetime(clusters["Date"], format="%Y-%m-%d")
    # Merge the clusters into the second_df DataFrame using index
    second_df = second_df.merge(
        clusters.set_index("Date"), left_index=True, right_index=True, how="inner"
    )
    # Count profiles per cluster (for the second file)
    cluster_sizes = second_df["Cluster"].value_counts()

    # Ensure grad_df_plot has its index converted to datetime using the proper format
    grad_df_plot = grad_df_plot.copy()
    grad_df_plot.index = pd.to_datetime(grad_df_plot.index, format="%d.%m.%Y")

    # Reset the index so that 'Date' becomes a column and rename it for consistency
    grad_df_plot = grad_df_plot.reset_index().rename(columns={"index": "Date"})

    # Merge grad_df_plot with clusters on the "Date" column (using a left join to keep all grad_df_plot rows)
    grad_df_plot = pd.merge(grad_df_plot, clusters, on="Date", how="left")

    # Compute mean profiles for each cluster from the second file
    mean_profiles = {}
    profile_cols = [col for col in second_df.columns if col.isdigit()]
    heights_profile = sorted([int(col) for col in profile_cols])
    for cluster in range(1, n_clusters + 1):
        cluster_data = (
            second_df[second_df["Cluster"] == cluster]
            .loc[:, profile_cols]
            .select_dtypes(include=[np.number])
        )
        mean_profiles[cluster] = cluster_data.mean(axis=0)

    grad_cols = [col for col in grad_df_plot.columns if col.isdigit()]
    heights_grad = sorted([int(col) for col in grad_cols])

    # Compute mean gradient profiles for each cluster
    mean_gradients = {}

    for cluster in range(1, n_clusters + 1):
        cluster_data = (
            grad_df_plot[grad_df_plot["Cluster"] == cluster]
            .loc[:, grad_cols]
            .select_dtypes(include=[np.number])
        )
        mean_gradients[cluster] = cluster_data.mean(axis=0)

    # Step 4: Visualize mean profiles from both files in a dual-panel plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 5))

    # Step 4.1: Plot individual profiles (if enabled)
    if include_single_profiles:
        # Individual profiles from profile_file
        for _, row in second_df.iterrows():
            single_profile = row[profile_cols].values
            cluster = row["Cluster"]
            ax2.plot(
                single_profile,
                heights_profile,
                color=f"C{int(cluster)}",
                linewidth=0.3,
                alpha=0.2,
            )

        # Individual profiles from gradient_file
        for _, row in grad_df.iterrows():
            single_gradient = row[grad_cols].values
            cluster = row["Cluster"]
            ax1.plot(
                single_gradient,
                heights_grad,
                color=f"C{int(cluster)}",
                linewidth=0.3,
                alpha=0.2,
            )

    # Step 4.2: Plot mean gradients on the left panel (gradient_file)
    for cluster in range(1, n_clusters + 1):
        ax1.plot(
            mean_gradients[cluster].values * 100,  # convert from °C/m to °C/100m
            heights_grad,
            linewidth=2,
        )

    ax1.set_xlabel("Temperature Gradient (°C/100m)", fontsize=12)
    ax1.set_xlim(-1.5, 1.5)  # Set limits for consistency
    ax1.axhspan(0, 100, color="grey", alpha=0.3, zorder=3)  # Shade lowest 100m
    ax1.grid()

    # Step 4.3: Plot mean profiles on the right panel (profile_file)
    for cluster in range(1, n_clusters + 1):
        ax2.plot(
            mean_profiles[cluster].values,
            heights_profile,
            label=f"CL{cluster}\nn = {cluster_sizes.get(cluster, 0)}",
            linewidth=2,
        )

    ax2.set_xlabel("Temperature (°C)", fontsize=12)
    ax2.axhspan(0, 100, color="grey", alpha=0.3, zorder=3)
    ax2.grid()

    # Step 4.4: Adjust y-axis limits
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel("Altitude Above Ground (m)", fontsize=12)

    plt.tight_layout()

    fig.subplots_adjust(bottom=0.2)

    # Collect legend handles and labels from ax2
    handles, labels = ax2.get_legend_handles_labels()

    ax1.text(
        -0.1,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.text(
        -0.02,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Add a figure-level legend below both plots, centered
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        handletextpad=0.9,
        columnspacing=1.0,
        frameon=False,
        fontsize=10,
    )

    # Save the combined figure
    out_file_png = os.path.join(
        output_dir,
        f"mean_profiles_{n_clusters}_cluster_single_{include_single_profiles}.png",
    )
    out_file_pdf = os.path.join(
        output_dir,
        f"mean_profiles_{n_clusters}_cluster_single_{include_single_profiles}.pdf",
    )

    plt.savefig(out_file_png, dpi=300)
    plt.savefig(out_file_pdf, dpi=300)
    plt.close()

    print(f"Plot saved to {out_file_png}")

    # Save clusters as CSV
    os.makedirs("results/k_means", exist_ok=True)
    cluster_file = os.path.join(
        "results", "k_means", f"{exclude_lowest}_{n_clusters}_clusters.csv"
    )
    clusters.to_csv(cluster_file, index=False)
    print(f"Clusters saved to {cluster_file}")

    return clusters


def evaluate_kmeans(
    data_file,
    output_dir,
    exclude_lowest=100,
    cluster_range=(2, 10),
    random_state=42,
    max_iter=500,
):
    """
    Evaluate KMeans clustering to determine the optimal number of clusters using
    the elbow method and silhouette score. Plots are saved to the output directory.

    Parameters:
        data_file (str): Path to the CSV file containing data for clustering.
        output_dir (str): Directory to save the output plots.
        exclude_lowest (int): Number of lowest meters to exclude. Default is 100.
        cluster_range (tuple): Range of cluster numbers to evaluate. Default is (2, 10).
        random_state (int): Random state for reproducibility. Default is 42.
        max_iter (int): Maximum number of iterations for KMeans. Default is 500.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and preprocess the data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.iloc[:, exclude_lowest:]  # Exclude the lowest `exclude_lowest` meters
    data_matrix = df.values  # Convert to numpy array

    # Standardize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_matrix)

    # Step 2: Initialize variables for evaluations
    cluster_range_values = range(cluster_range[0], cluster_range[1] + 1)
    inertia_values = []  # To store the inertia for each cluster count
    silhouette_scores = []  # To store the silhouette score for each cluster count

    # Step 3: Evaluate KMeans for each cluster count
    for n_clusters in cluster_range_values:
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, max_iter=max_iter
        )
        cluster_labels = kmeans.fit_predict(normalized_data)

        # Record inertia (for elbow curve)
        inertia_values.append(kmeans.inertia_)

        # Record silhouette score (for silhouette plot)
        silhouette_avg = silhouette_score(normalized_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Step 4: Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range_values, inertia_values, marker="o", label="Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of squared distances")
    plt.grid()
    plt.xticks(cluster_range_values)
    plt.tight_layout()
    elbow_plot_path = os.path.join(output_dir, "elbow_curve.png")
    plt.savefig(elbow_plot_path, dpi=300)
    plt.close()
    print(f"Elbow curve saved to {elbow_plot_path}")

    # Step 5: Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(
        cluster_range_values,
        silhouette_scores,
        marker="o",
        label="Silhouette Score",
        color="orange",
    )
    plt.title("Silhouette Score for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.xticks(cluster_range_values)
    plt.tight_layout()
    silhouette_plot_path = os.path.join(output_dir, "silhouette_scores.png")
    plt.savefig(silhouette_plot_path, dpi=300)
    plt.close()
    print(f"Silhouette score plot saved to {silhouette_plot_path}")


def plot_mean_anomalies_by_cluster(
    clusters_df_path,
    anomaly_file_path,
    temp_level=850,
    geopotential_level=500,
    vmin=-2.5,
    vmax=2.5,
    output_path=None,
):
    """
    Plots mean temperature and geopotential height anomalies for all clusters from K-means on one canvas.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    clusters_df = pd.read_csv(clusters_df_path)
    clusters_df["Date"] = pd.to_datetime(clusters_df["Date"])
    clusters_df["Date"] = clusters_df["Date"].dt.strftime("%Y-%m-%dT12:00:00")

    # Load the anomaly dataset
    ds = xr.open_dataset(anomaly_file_path)

    # Convert valid_time to string for consistent merging
    ds["valid_time"] = ds["valid_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Merge K-means results with the anomaly dataset valid_time
    merged_data = pd.DataFrame({"valid_time": ds["valid_time"].values}).merge(
        clusters_df,
        how="left",
        left_on="valid_time",
        right_on="Date",
    )

    # Assign clusters as a new coordinate in the dataset
    ds = ds.assign_coords(
        cluster=("valid_time", merged_data["Cluster"].fillna(-1).values)
    )

    # Calculate mean anomalies per cluster (excluding placeholder cluster -1)
    valid_clusters = np.sort(merged_data["Cluster"].dropna().unique())
    num_clusters = len(valid_clusters)

    # Count occurrences of each cluster
    cluster_counts = merged_data["Cluster"].value_counts()

    # Initialize the canvas for subplots
    fig, axes = plt.subplots(
        nrows=(num_clusters + 1) // 3,
        ncols=3,
        figsize=(12, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = axes.flatten()

    # Loop through clusters and create individual subplots
    for i, cluster_id in enumerate(valid_clusters):
        t_anomaly = (
            ds["t_anomaly"].where(ds["cluster"] == cluster_id).mean(dim="valid_time")
        )
        z_anomaly = (
            ds["z_anomaly"].where(ds["cluster"] == cluster_id).mean(dim="valid_time")
        )

        ax = axes[i]

        # Plot temperature anomaly as colored map
        im = t_anomaly.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_aspect("auto")

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAKES)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True)
        if i % 3 != 0:
            gl.left_labels = False  # Disable y-axis labels for middle/right plots
        if i < num_clusters - 3:
            gl.bottom_labels = False  # Disable x-axis labels for top plots

        # Contours for geopotential height anomaly
        contour_levels = np.arange(
            np.floor(z_anomaly.min().values / 5) * 5,
            np.ceil(z_anomaly.max().values / 5) * 5 + 5,
            5,
        )
        contours = ax.contour(
            z_anomaly.longitude,
            z_anomaly.latitude,
            z_anomaly.values,
            levels=contour_levels,
            colors="black",
            linewidths=1,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(contours, inline=True, fontsize=14, fmt="%1.0f")  # Larger font size

        # Mark the specific coordinate
        longitude = -16.636666666666667
        latitude = 81.59916666666666
        ax.plot(
            longitude,
            latitude,
            marker="o",
            color="black",
            markersize=10,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            longitude,
            latitude,
            " VRS",
            fontsize=15,
            color="black",  # Black color for annotation
            transform=ccrs.PlateCarree(),
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Annotate the number of occurrences for each cluster
        count = cluster_counts.get(cluster_id, 0)
        ax.text(
            0.95,
            0.95,  # Position in axis coordinates (top right corner)
            f"n={count}",
            transform=ax.transAxes,
            fontsize=12,
            color="black",
            ha="right",
            va="top",
            backgroundcolor="white",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        # Set the title for each subplot
        ax.set_title(f"Cluster {int(cluster_id)}")

    # Adjust layout and add a shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # Position of the colorbar

    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="vertical", label="850 hPa Temp. Anomaly (°C)"
    )
    # Set the font size for the colorbar label
    cbar.set_label(
        "850 hPa Temp. Anomaly (°C)", fontsize=14
    )  # Increase font size of label

    # Set the font size for the ticks on the colorbar
    cbar.ax.tick_params(labelsize=12)  # Increase font size of ticks

    # Hide any unused axes
    for ax in axes[num_clusters:]:
        ax.axis("off")

    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])  # Adjust layout to fit x/y labels

    # Save the plot
    output_file = os.path.join(output_path, "mean_anomalies_by_cluster.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")


def plot_cluster_occurrences(clusters_df_path, output_dir, profile_times_path):
    """
    Plot normalized cluster occurrences per day of year for the summer season (June-August).
    Also plot annual cluster occurrences over the years and a single stacked bar plot of cluster occurrences.

    Parameters:
    clusters_df (pd.DataFrame): DataFrame with 'Cluster' and datetime index containing summer data.
    profile_times_file (str): File path to the profile times data (CSV).
    output_dir (str): Directory to save the output plot.
    """

    # Convert 'Date' column to datetime and set it as index (if not already)
    clusters_df = pd.read_csv(clusters_df_path)
    clusters_df["Date"] = pd.to_datetime(clusters_df["Date"])
    clusters_df.set_index("Date", inplace=True)

    # Step 1: Add a column for the 'Day of Year'
    clusters_df["DayOfYear"] = clusters_df.index.dayofyear

    # Step 2: Count the occurrences of each cluster per day of year
    daily_occurrences = (
        clusters_df.groupby([clusters_df["DayOfYear"], "Cluster"])
        .size()
        .unstack(fill_value=0)
    )

    # Step 3: Calculate the number of occurrences for each DayOfYear across the entire DataFrame
    day_of_year_counts = clusters_df.groupby("DayOfYear").size()

    # Step 4: Divide the occurrences by the total count of each DayOfYear
    # This gives the normalized occurrences for each cluster per day of year
    normalized_daily_occurrences = daily_occurrences.div(day_of_year_counts, axis=0)

    # Step 5: Extract the year and create a 'Year' column for the annual plot
    clusters_df["Year"] = clusters_df.index.year

    # Step6: Count the occurrences of each cluster per year
    annual_occurrences = (
        clusters_df.groupby(["Year", "Cluster"]).size().unstack(fill_value=0)
    )

    # Step 7: Load profile time data for the third plot
    profile_times_df = pd.read_csv(profile_times_path)
    profile_times_df["Time"] = pd.to_datetime(profile_times_df["Time"])
    profile_times_df["Date"] = profile_times_df["Time"].dt.date
    unique_dates = profile_times_df["Date"].unique()
    unique_dates_df = pd.DataFrame(unique_dates, columns=["Date"])

    # Merge the clusters with the profile times based on the date
    clusters_df.index = pd.to_datetime(clusters_df.index).date
    merged_df = pd.merge(
        clusters_df, unique_dates_df, left_index=True, right_on="Date", how="inner"
    )

    # Calculate the number of occurrences for each cluster
    cluster_counts = merged_df["Cluster"].value_counts()

    # Normalize the counts to get the ratios
    total_count = cluster_counts.sum()
    cluster_ratios = cluster_counts / total_count

    # Create a DataFrame for plotting (normalized counts as ratios)
    cluster_ratios_df = pd.DataFrame(cluster_ratios).T
    # Sort the numeric column headers
    sorted_columns = sorted(cluster_ratios_df.columns)

    # Reindex the DataFrame with the sorted columns
    cluster_ratios_df = cluster_ratios_df.reindex(columns=sorted_columns)

    # Step 9: Create a figure and set up subplots with adjusted widths
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 7)  # Divide the figure grid into X columns
    ax1 = fig.add_subplot(gs[0, :3])  # Normalized occurrences plot takes 5 columns
    ax2 = fig.add_subplot(gs[0, 3:6])  # Annual occurrences plot takes another 5 columns
    ax3 = fig.add_subplot(gs[0, 6:])  # Single bar plot takes the last column

    # Define a consistent color palette for the clusters
    unique_clusters = sorted(clusters_df["Cluster"].unique())
    palette = sns.color_palette(n_colors=len(unique_clusters))
    cluster_palette = {
        cluster: color for cluster, color in zip(unique_clusters, palette)
    }

    # Plot 1: Normalized Cluster Occurrences Per Day of Year
    normalized_daily_occurrences.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        width=0.9,
        color=[cluster_palette.get(x) for x in normalized_daily_occurrences.columns],
    )
    ax1.set_ylabel("Mean Relative Occurrence Per Day")
    ax1.set_xlabel("Day of Year")
    ax1.set_xticks(range(0, len(normalized_daily_occurrences), 5))
    ax1.tick_params(axis="x")

    # Plot 2: Cluster Occurrence Ratio Per Year
    annual_occurrences.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        width=0.9,
        color=[cluster_palette.get(x) for x in annual_occurrences.columns],
    )
    ax2.set_xticks(range(0, len(annual_occurrences), 5))
    ax2.set_ylabel("Cluster Occurrences")
    ax2.set_xlabel("Year")

    # Plot 3: Single Stacked Bar Plot for Cluster Occurrences
    cluster_ratios_df.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        width=0.9,
        color=[cluster_palette.get(x) for x in [0, 2, 3, 4]],
    )  # Reordered stacking

    # Add labels to the bar segments (absolute count for each cluster)
    bottom = 0  # Start position for stacking
    for i, cluster in enumerate([0, 2, 3, 4]):  # Reordered stack order
        count = cluster_counts.get(cluster, 0)
        ax3.text(
            0,
            bottom + (count / total_count) / 2,
            f"n={count}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
        bottom += count / total_count  # Update the position for the next segment

    # Label the axes and title for the third plot
    ax3.set_ylabel("Cluster Occurrence Ratio")
    ax3.set_xlabel("Clusters on\nField Days")
    ax3.set_xticklabels([])

    # Gather handles and labels for the legend
    handles, labels = ax1.get_legend_handles_labels()

    # Add the word 'Cluster' before each label
    labels = [f"Cluster {label}" for label in labels]

    # Remove individual legends from all subplots
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    # Create the combined legend below the plots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        fontsize=12,
    )

    # Adjust layout for the combined figure
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for the legend
    plt.subplots_adjust(wspace=0.6, left=0.0, right=1)  # Adjust for space for y-labels

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "cluster_occurrences_seasonal_and_annual.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_difference_anomaly(
    difference_anomaly_file,
    clusters_df_path,
    output_dir,
    snow_fraction_file,
    snow_thresh=0.3,
    include_single_profiles=False,
):
    """
    Plots the mean difference anomaly profiles for all clusters in a single plot,
    filtered by snow fraction threshold. Optionally includes single anomaly profiles.

    Parameters:
        difference_anomaly_file (str): Path to the difference anomaly file (CSV).
        kmeans_results (pd.DataFrame): DataFrame with 'Date' and 'Cluster' columns indicating cluster assignments.
        output_dir (str): Directory to save the generated plot.
        snow_fraction_file (str): Path to the snow fraction file (CSV).
        snow_thresh (float): Threshold for snow fraction to include data points (default: 0.3).
        include_single_profiles (bool): Whether to include single anomaly profiles as thin, transparent lines (default: False).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the difference anomaly data
    df = pd.read_csv(
        difference_anomaly_file, index_col=0, parse_dates=True, dayfirst=True
    )

    # Merge difference anomalies with cluster assignments
    kmeans_results = pd.read_csv(clusters_df_path, parse_dates=["Date"])

    df = df.merge(kmeans_results, left_index=True, right_on="Date", how="inner")

    # Load the snow fraction data
    snow_fraction_data = pd.read_csv(snow_fraction_file)
    snow_fraction_data["time"] = pd.to_datetime(
        snow_fraction_data["time"], format="%Y-%m-%d"
    )

    # Merge with the main DataFrame based on the Date/time column
    merged_df = df.merge(
        snow_fraction_data, left_on="Date", right_on="time", how="inner"
    )

    # Filter rows where snow fraction is below the threshold
    filtered_df = merged_df[merged_df["snow_fraction"] < snow_thresh]

    # Group by cluster and calculate the mean anomaly for each cluster
    grouped = filtered_df.groupby("Cluster").mean()

    # Exclude the 'Date' and 'time' columns explicitly
    numeric_columns = [
        col for col in grouped.columns if col not in ["Date", "time", "snow_fraction"]
    ]
    grouped = grouped[numeric_columns]

    # Convert column names (heights) to floats for plotting
    heights = [float(col) for col in grouped.columns]

    # Plot mean profiles for all clusters in one plot
    plt.figure(figsize=(6, 8))

    if include_single_profiles:
        # Plot single anomaly profiles as thin, transparent lines
        for _, row in filtered_df.iterrows():
            cluster = row["Cluster"]
            single_profile = row[numeric_columns].values
            plt.plot(
                single_profile,
                heights,
                color=f"C{int(cluster)}",
                linewidth=0.3,
                alpha=0.3,
            )
        plt.xlim(-0.5, 2)

    for cluster in grouped.index:
        plt.plot(
            grouped.loc[cluster].values,
            heights,
            label=f"Cluster {cluster} ({len(filtered_df[filtered_df['Cluster'] == cluster])} profiles)",
            linewidth=2,
        )

    # Add formatting and labels
    plt.xlabel("Mean Temperature Anomaly (°C)")
    plt.ylabel("Height Above Ground (m)")
    plt.title(f"Mean Temp. Diff. Anomaly per Cluster (Snow Frac. < {snow_thresh})")
    plt.grid()
    plt.ylim(0, 500)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()

    # Save the plot
    if include_single_profiles:
        out_file = os.path.join(
            output_dir,
            f"mean_diff_anomaly_profiles_per_cluster_snow_thresh_{snow_thresh}_single.png",
        )
    else:
        out_file = os.path.join(
            output_dir,
            f"mean_diff_anomaly_profiles_per_cluster_snow_thresh_{snow_thresh}.png",
        )
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Plot saved to {out_file}")


def plot_mass_change(
    file_path,
    output_dir="plots/flade_isblink",
    variable_to_plot="absolute_mass_change_FI",
):
    """
    Loads the mass change data from the given CSV file and plots the absolute mass change
    for Flade Isblink Full as a bar plot. Bars are colored blue for positive values and red for negative values.

    Parameters:
    - file_path (str): The path to the CSV file containing the mass change data.
    - output_dir (str): The directory where the plot will be saved.
    - variable_to_plot (str): The variable to plot (e.g., 'absolute_mass_change_FI').
    """
    # Step 1: Load the data from the CSV file
    mass_df = pd.read_csv(file_path)

    # Step 2: Convert Year to integers for plotting
    mass_df["Year"] = mass_df["Year"].astype(int)

    # Step 3: Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(7, 4))

    # Plot the bars with conditional coloring
    colors = [
        "blue" if value > 0 else "red" for value in mass_df[f"{variable_to_plot}_full"]
    ]
    ax.bar(mass_df["Year"], mass_df[f"{variable_to_plot}_full"], color=colors)

    # Set labels and title
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass Change (Gt)")
    plt.tight_layout()
    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color="black", linewidth=1, linestyle="--")

    # Add grid
    ax.grid(linestyle="--", alpha=0.7)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{variable_to_plot}_full_mass_change_bar_plot.png"
    )
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved to {output_file}")


def plot_snow_fraction_vs_in_situ(vrs_file, snow_fraction_file, output_dir):
    """
    Plots Snow Fraction vs SWE and Snow Fraction vs HS, color-coded by Day of Year (DOY),
    and saves the resulting figure to the specified output directory.

    Parameters:
    - vrs_file (str): Path to the VRS snow data CSV file.
    - snow_fraction_file (str): Path to the snow fraction data CSV file.
    - output_dir (str): Directory to save the output plot.
    """
    # Load the datasets
    vrs_data = pd.read_csv(vrs_file)
    snow_fraction_data = pd.read_csv(snow_fraction_file)

    # Convert date columns to datetime format
    vrs_data["Date"] = pd.to_datetime(vrs_data["Date"])
    snow_fraction_data["time"] = pd.to_datetime(
        snow_fraction_data["time"], format="%Y-%m-%d"
    )

    # Add day of the year (DOY) for color-coding
    vrs_data["DOY"] = vrs_data["Date"].dt.dayofyear

    # Merge the datasets on the date column
    data_merged = pd.merge(
        vrs_data, snow_fraction_data, left_on="Date", right_on="time"
    )

    # Add DOY to the merged data for color-coding
    data_merged["DOY"] = data_merged["Date"].dt.dayofyear

    # Create a figure with two subplots side-by-side
    fig, axs = plt.subplots(
        1, 2, figsize=(8, 4), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )

    # Plot snow fraction vs SWE (snow water equivalent)
    sc1 = axs[0].scatter(
        data_merged["swe"],
        data_merged["snow_fraction"],
        c=data_merged["DOY"],
        cmap="viridis",
        alpha=0.7,
    )
    axs[0].set_xlabel("SWE mm (in-situ)")
    axs[0].set_ylabel("Snow Fraction (CARRA)")
    axs[0].grid(True)

    # Calculate and annotate correlation metrics for SWE
    swe_corr, swe_pval = pearsonr(data_merged["swe"], data_merged["snow_fraction"])
    p_annotation_swe = "p < 0.05" if swe_pval < 0.05 else f"p = {swe_pval:.3f}"
    axs[0].annotate(
        f"r = {swe_corr:.3f}\n{p_annotation_swe}",
        xy=(0.05, 0.86),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot snow fraction vs HS (snow height)
    axs[1].scatter(
        data_merged["hs"],
        data_merged["snow_fraction"],
        c=data_merged["DOY"],
        cmap="viridis",
        alpha=0.7,
    )
    axs[1].set_xlabel("Snow Height m (in-situ)")
    axs[1].grid(True)

    # Calculate and annotate correlation metrics for HS
    hs_corr, hs_pval = pearsonr(data_merged["hs"], data_merged["snow_fraction"])
    p_annotation_hs = "p < 0.05" if hs_pval < 0.05 else f"p = {hs_pval:.3f}"
    axs[1].annotate(
        f"r = {hs_corr:.3f}\n{p_annotation_hs}",
        xy=(0.05, 0.86),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    cbar_ax = fig.add_axes(
        [0.999, 0.15, 0.02, 0.7]
    )  # Adjust [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Day of Year")

    # Adjust layout and save the plot
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "snow_fraction_vs_swe_hs.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {output_path}")


def correlation_heatmap(
    snow_fraction_file,
    input_directory,
    output_path,
    p_value_sigma=0.5,
    correlation_filter_sigma=1.2,
    p_value_thresh=0.05,
    ylim_max=None,
):
    """
    Generate correlation heatmaps between snow fraction anomalies and temperature difference anomalies.

    Parameters:
        snow_fraction_file (str): Path to the snow fraction data file (CSV).
        input_directory (str): Directory containing temperature difference anomaly files (CSV).
        output_path (str): Directory to save the generated plots.
        p_value_sigma (float): Gaussian filter strength for p-value smoothing.
        correlation_filter_sigma (float): Gaussian filter strength for correlation smoothing.
        p_value_thresh (float): P-value threshold for significance.
        ylim_max (float): Maximum y-axis limit for the plots.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load snow fraction data
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year

    # Calculate the typical snow fraction for each DOY
    typical_snow_fraction = snow_df.groupby("DOY")["snow_fraction"].mean()

    # Apply LOESS smoothing to the typical snow fraction
    typical_snow_fraction_smoothed = sm.nonparametric.lowess(
        typical_snow_fraction, typical_snow_fraction.index, frac=0.1
    )

    # Extract only the smoothed values (2nd column)
    typical_snow_fraction_smoothed = typical_snow_fraction_smoothed[:, 1]

    # Create a DataFrame from the smoothed typical values
    typical_snow_fraction_df = pd.DataFrame(
        {
            "DOY": typical_snow_fraction.index,
            "snow_fraction_typical": typical_snow_fraction_smoothed,
        }
    )

    # Merge smoothed typical values with the original DataFrame
    snow_df = snow_df.merge(typical_snow_fraction_df, on="DOY", how="left")

    # Calculate anomalies (actual - typical for the same DOY)
    snow_df["anomaly"] = snow_df["snow_fraction"] - snow_df["snow_fraction_typical"]

    # Get all CSV files in the input directory
    input_files = [
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if file.endswith(".csv")
    ]

    # Function to smooth a DataFrame while preserving its structure
    def smooth_dataframe(df, sigma):
        smoothed_array = gaussian_filter(df.to_numpy(), sigma=sigma)
        return pd.DataFrame(smoothed_array, index=df.index, columns=df.columns)

    # Loop through each file and process temperature anomalies
    for idx, file in enumerate(input_files):
        # Load the temperature difference data
        df = pd.read_csv(file, index_col=0, parse_dates=True, dayfirst=True)
        df.sort_index(inplace=True)

        # Ensure 'DOY' is added to the DataFrame
        df["DOY"] = df.index.day_of_year

        # Reindex snow_df based on df.index to align the timestamps
        snow_df_reindexed = snow_df.set_index("time").reindex(
            df.index, fill_value=np.nan
        )

        # Merge snow anomaly into temperature data
        df["snow_anomaly"] = snow_df_reindexed["anomaly"]

        # Dynamically determine the range of DOY in the data
        min_doy = df["DOY"].min()
        max_doy = df["DOY"].max()

        # Exclude non-relevant columns explicitly
        columns_to_correlate = [
            col for col in df.columns if col not in ["DOY", "snow_anomaly"]
        ]

        # Prepare DataFrames using the determined DOY range and relevant columns
        correlation_results = pd.DataFrame(
            index=range(min_doy, max_doy + 1), columns=columns_to_correlate
        )
        correlation_p = pd.DataFrame(
            index=range(min_doy, max_doy + 1), columns=columns_to_correlate
        )

        # Loop through each DOY (152 to 243)
        for doy, group in df.groupby("DOY"):
            group = group.dropna(subset=["snow_anomaly"])

            # Calculate correlation of 'snow_anomaly' with each column
            for col in group.columns[:-2]:
                corr, p_value = pearsonr(group["snow_anomaly"], group[col])
                correlation_results.loc[doy, col] = corr
                correlation_p.loc[doy, col] = p_value

        # Convert results to numeric for further processing
        correlation_results = correlation_results.astype(float)
        correlation_p = correlation_p.astype(float)

        # Smooth correlation results and p-values
        correlation_results_smoothed = smooth_dataframe(
            correlation_results, sigma=correlation_filter_sigma
        )
        correlation_p_smoothed = smooth_dataframe(correlation_p, sigma=p_value_sigma)

        # Plot the correlation heatmap
        plt.figure(figsize=(6, 4))
        plt.imshow(
            correlation_results_smoothed.T,
            aspect="auto",
            cmap="bwr",
            vmin=-0.8,
            vmax=0.8,
            extent=[
                correlation_results.index.min(),
                correlation_results.index.max(),
                correlation_results.shape[1],
                0,
            ],
        )
        plt.colorbar(label="Correlation Coefficient")

        # Indicate significant p-values with small yellow dots
        significant_mask = correlation_p_smoothed < p_value_thresh
        y_indices, x_indices = np.where(significant_mask.T)

        plt.scatter(
            x_indices + 152,
            y_indices - 1,
            color="yellow",
            s=0.3,
            marker="*",
            zorder=1.5,
            alpha=0.4,
        )

        # Set labels and title
        plt.xlabel("Day of Year")
        plt.ylabel("Height Above Ground (m)")
        plt.title("Corr(Snow Frac. Anomaly, Temp. Diff. Anomaly)")

        # Adjust y-axis limits if ylim_max is provided
        if ylim_max is not None:
            plt.ylim(0, ylim_max)

        # Adjust layout and save the plot
        plt.tight_layout()
        plot_filename = os.path.join(
            output_path,
            f'corr_{os.path.basename(file).split(".")[0]}_p_sigma_{p_value_sigma}_p_value_{p_value_thresh}_r_sigma{correlation_filter_sigma}.png',
        )
        plt.savefig(plot_filename, dpi=300)

        print(f"Correlation plot saved for {file}")


def plot_smb_cluster(
    file_path_merged_df, output_dir, variable_to_plot, plot_type="violin"
):
    """
    Plots a violin or boxplot plot of the specified variable grouped by clusters, with counts annotated.

    Parameters:
        smb_csv_path (str): Path to the CSV file containing SMB data.
        cluster_csv_path (str): Path to the CSV file containing cluster assignments.
        variable_to_plot (str): The column name in the SMB dataset to plot.
        plot_type (str): The type of plot to create ('violin' or 'boxplot'). Default is 'violin'.
    """

    # Load the merged data
    merged_df = pd.read_csv(file_path_merged_df, parse_dates=["Date"])

    if plot_type == "violin":
        # Create the violin plot
        plt.figure(figsize=(8, 5))
        ax = sns.violinplot(
            data=merged_df,
            x="Cluster",
            y=variable_to_plot,
            color="dodgerblue",
            linewidth=1.5,
        )
    if plot_type == "boxplot":
        # Create the boxplots
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(
            data=merged_df,
            x="Cluster",
            y=variable_to_plot,
            color="dodgerblue",
            linewidth=1.5,
        )

    # Annotate the count of observations for each cluster
    cluster_counts = merged_df["Cluster"].value_counts().sort_index()
    for i, count in enumerate(cluster_counts):
        ax.text(
            x=i,
            y=merged_df[variable_to_plot].min()
            * 1.28,  # Position near the top of the violins
            s=f"n={count}",
            ha="center",
            fontsize=12,
            color="black",
            bbox=dict(
                facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"
            ),  # Add a box around the text
        )

    # Add labels and title
    plt.xlabel("Cluster", fontsize=13)
    plt.ylabel(variable_to_plot, fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=1)
    plt.ylim(
        merged_df[variable_to_plot].min() * 1.35,
        merged_df[variable_to_plot].max() * 1.15,
    )
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(
        output_dir, f"{variable_to_plot}_by_cluster_{plot_type}.png"
    )
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def plot_smb_mar(file_path_smb, output_dir):
    """
    Takes csv with SMB data and plots the Specific Mass Balance (SMB) for each hydrological year parsed in winter/summer balance.
    Parameters:
    - file_path_smb (str): Path to the CSV file containing the SMB data.
    - output_dir (str): Directory to save the output plot.
    """
    # Load data
    smb_df = pd.read_csv(file_path_smb)
    smb_df["Time"] = pd.to_datetime(smb_df["Time"])

    # Filter data to include only from 01.09.1991 to 31.08.2024
    start_date = "1991-09-01"
    end_date = "2024-08-31"
    smb_df = smb_df[(smb_df["Time"] >= start_date) & (smb_df["Time"] <= end_date)]

    # Add Year and Month columns
    smb_df["Year"] = smb_df["Time"].dt.year
    smb_df["Month"] = smb_df["Time"].dt.month

    # Define hydrological year
    smb_df["HydroYear"] = smb_df["Year"]
    smb_df.loc[smb_df["Month"] >= 9, "HydroYear"] += 1

    # Add Season column
    smb_df["Season"] = smb_df["Month"].apply(
        lambda x: "Summer" if x in [6, 7, 8] else "Winter"
    )

    # Group by Hydrological Year and Season, then sum Specific Mass Balance
    seasonal_sums = (
        smb_df.groupby(["HydroYear", "Season"])["Specific Mass Balance (mm WE)"]
        .sum()
        .unstack(fill_value=0)
    )

    # Calculate total mass balance for each hydrological year
    seasonal_sums["Total"] = seasonal_sums.sum(axis=1)

    # Convert SMB (m^3) to Gigatonnes and group by hydrological year
    smb_df["SMB (Gt)"] = smb_df["SMB (m^3)"] / 1000000000
    yearly_smb_gt = smb_df.groupby("HydroYear")["SMB (Gt)"].sum()

    # Compute cumulative sum
    cumulative_smb_gt = yearly_smb_gt.cumsum()

    # Prepare data for plotting by converting von mm to m
    hydro_years = seasonal_sums.index
    summer = seasonal_sums["Summer"] / 1000
    winter = seasonal_sums["Winter"] / 1000
    total = seasonal_sums["Total"] / 1000

    # Create the plots
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # First plot: Stacked seasonal bars and total bars
    bar_width = 0.4  # Width of each bar group
    x = np.arange(len(hydro_years))  # positions for hydrological years

    axs[0].bar(
        x,
        winter,
        width=bar_width,
        label="Winter (Sept-May)",
        color="dodgerblue",
        zorder=3,
        alpha=0.4,
    )
    axs[0].bar(
        x,
        summer,
        width=bar_width,
        label="Summer (June-Aug.)",
        color="red",
        zorder=3,
        alpha=0.55,
    )
    axs[0].bar(
        x + bar_width, total, width=bar_width, label="Annual", color="black", zorder=3
    )

    # Customize first plot
    axs[0].set_ylabel("Specific SMB (m we)", fontsize=12)
    axs[0].set_xlim(-0.7, len(hydro_years) - 0.3)
    axs[0].legend(
        fontsize=10,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=3,
        handletextpad=0.6,
        columnspacing=0.6,
    )
    axs[0].grid(linestyle="--", alpha=0.7, zorder=0)

    # Second plot: Cumulative mass balance
    axs[1].plot(
        x + bar_width / 2, cumulative_smb_gt.values, color="green", marker="o", zorder=3
    )
    axs[1].set_ylabel("Cumulative SMB (Gt)", fontsize=12)
    axs[1].set_xlabel("Hydrological Year", fontsize=12)
    axs[1].grid(linestyle="--", alpha=0.7, zorder=0)

    # Set shared x-axis labels
    axs[1].set_xticks(x + bar_width / 2)
    axs[1].set_xticklabels(hydro_years, rotation=45)

    # Adjust layout and show
    plt.tight_layout()

    # save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "winter_summer_smb_mar.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def cluster_contribution(file_path_merged_df, output_dir):
    """
    Calculate the cluster contribution to the total specific mass balance and occurrence ratio in summer, saves table as html file.
    Parameters:
    - file_path_merged_df (str): Path to the CSV file containing the mar and cluster merged data.
    - output_dir (str): Directory to save the output plot.
    """
    # Load the merged data
    merged_df = pd.read_csv(file_path_merged_df, parse_dates=["Date"])
    # Calculate cluster contribution
    cluster_contribution = (
        merged_df.groupby("Cluster")["Specific Mass Balance (mm WE)"].sum().sort_index()
    )
    total_mass_change = cluster_contribution.sum()
    cluster_contribution_percentage = (cluster_contribution / total_mass_change) * 100

    # Calculate occurrence ratio
    cluster_occurrence = merged_df["Cluster"].value_counts().sort_index()
    total_occurrences = cluster_occurrence.sum()
    cluster_occurrence_percentage = (cluster_occurrence / total_occurrences) * 100

    # Combine metrics into a DataFrame
    cluster_summary = pd.DataFrame(
        {
            "Total Contribution (m WE)": cluster_contribution / 1000,
            "Days Count": cluster_occurrence,
            "Contribution Percentage (%)": cluster_contribution_percentage,
            "Occurrence Percentage (%)": cluster_occurrence_percentage,
        }
    )

    # Add a row for the sums
    sums_row = pd.DataFrame(
        {
            "Total Contribution (m WE)": [
                cluster_summary["Total Contribution (m WE)"].sum()
            ],
            "Days Count": [cluster_summary["Days Count"].sum()],
            "Contribution Percentage (%)": [
                cluster_summary["Contribution Percentage (%)"].sum()
            ],
            "Occurrence Percentage (%)": [
                cluster_summary["Occurrence Percentage (%)"].sum()
            ],
        },
        index=["Total"],
    )

    # Append the sums row to the original cluster summary
    cluster_summary = pd.concat([cluster_summary, sums_row])

    # Reset the index to move it into a separate column
    cluster_summary.reset_index(inplace=True)
    cluster_summary.rename(columns={"index": "Cluster"}, inplace=True)

    # round values in summary table
    cluster_summary = cluster_summary.round(1)
    # Convert to HTML table
    html_table = cluster_summary.to_html(index=False)

    # Save the table to a file
    tablefile = os.path.join(output_dir, "cluster_contribution_summary.html")
    with open(tablefile, "w") as file:
        file.write(html_table)

    print(f"Cluster contribution summary saved to {tablefile}")


def plot_smb_t2m_anomalies_and_scatter(file_path_merged_df, output_dir):
    """
    Plot the yearly summer mean Surface Mass Balance (SMB) and Temperature Anomalies (T2m_anom)
    for each cluster with trend lines and significance markers in the legend.
    Additionally, add a scatterplot of all yearly data (SMB vs. T2m_anom).

    Parameters:
    - file_path_merged_df (str): Path to the CSV file containing the merged data.
    - output_dir (str): Directory to save the output plot.
    """
    # Load merged_df and extract the year
    merged_df = pd.read_csv(file_path_merged_df, parse_dates=["Date"])
    merged_df["Year"] = merged_df["Date"].dt.year

    # Compute yearly means per cluster (unstacked DataFrames)
    yearly_means_smb = (
        merged_df.groupby(["Year", "Cluster"])["Specific Mass Balance (mm WE)"]
        .mean()
        .unstack()
    )
    yearly_means_t2m = (
        merged_df.groupby(["Year", "Cluster"])["T2m_anom"].mean().unstack()
    )

    # Compute overall yearly means (all clusters combined)
    all_yearly_means_smb = merged_df.groupby(["Year"])[
        "Specific Mass Balance (mm WE)"
    ].mean()
    all_yearly_means_t2m = merged_df.groupby(["Year"])["T2m_anom"].mean()
    r_all, p_all = pearsonr(all_yearly_means_smb, all_yearly_means_t2m)

    # Set up the figure using GridSpec:
    # Left column (60% width): two stacked plots sharing the x-axis (ax1 and ax2)
    # Right column (40% width): one scatter plot (ax3) spanning both rows.
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(
        nrows=2, ncols=2, width_ratios=[0.6, 0.4], wspace=0.23, hspace=0.1
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    # Remove x tick labels from ax1 since it shares the x-axis with ax2
    ax1.tick_params(labelbottom=False)

    prop_cycle = plt.rcParams["axes.prop_cycle"]

    cycle_styles = list(prop_cycle)

    # Extract colors and linestyles separately
    default_colors = [s["color"] for s in cycle_styles]
    default_linestyles = [s["linestyle"] for s in cycle_styles]

    # -------------------------
    # Plot 1: Yearly SMB by Cluster (ax1)
    smb_legend_handles = []

    for cluster in yearly_means_smb.columns:
        x = yearly_means_smb.index
        y = yearly_means_smb[cluster]
        valid_mask = ~y.isna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        color = default_colors[cluster - 1]
        print("check", cluster)

        linestyle = default_linestyles[cluster - 1]
        # Plot main SMB line with dynamic linestyle
        (line,) = ax1.plot(
            x_valid, y_valid, color=color, linestyle=linestyle, alpha=0.6, linewidth=1.3
        )

        label = f"CL{cluster}"

        # If there's enough data, compute and plot trendline
        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
            (trend_line,) = ax1.plot(
                x_valid,
                slope * x_valid + intercept,
                linestyle="-",
                color=color,
                linewidth=2,
            )
            if p_value < 0.05:
                label += "*"  # Mark significance

        # Create legend handle with extracted linestyle
        smb_legend_handles.append(
            mlines.Line2D(
                [], [], linestyle=linestyle, color=color, linewidth=2, label=label
            )
        )

    ax1.set_ylim(-20, 4)
    ax1.set_ylabel("Specific SMB (mm w.e.)")
    ax1.grid(linestyle="--", alpha=0.7)

    # Use extracted styles in the legend
    ax1.legend(
        handles=smb_legend_handles,
        fontsize=8,
        loc=[0.01, 0.01],
        handletextpad=0.3,
        columnspacing=1,
        frameon=True,
        ncol=5,
    )

    ax1.text(
        -0.1,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    # -------------------------
    # Plot 2: Yearly T2m Anomaly by Cluster (ax2)
    t2m_legend_handles = []

    for cluster in yearly_means_t2m.columns:
        x = yearly_means_t2m.index
        y = yearly_means_t2m[cluster]
        valid_mask = ~y.isna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        color = default_colors[cluster - 1]
        linestyle = default_linestyles[cluster - 1]

        # Plot main T2m line with dynamic linestyle
        (line,) = ax2.plot(
            x_valid, y_valid, color=color, linestyle=linestyle, alpha=0.6, linewidth=1.3
        )

        label = f"CL{cluster}"

        # If there's enough data, compute and plot trendline
        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
            (trend_line,) = ax2.plot(
                x_valid,
                slope * x_valid + intercept,
                linestyle="-",
                color=color,
                linewidth=2,
            )
            if p_value < 0.05:
                label += "*"  # Mark significance

        # Create legend handle with extracted linestyle
        t2m_legend_handles.append(
            mlines.Line2D(
                [], [], linestyle=linestyle, color=color, linewidth=2, label=label
            )
        )

    ax2.set_ylim(-3, 3)
    ax2.set_ylabel("T2m Anomaly (°C)")
    ax2.set_xlabel("Year")
    ax2.grid(linestyle="--", alpha=0.7)

    # Use extracted styles in the legend
    ax2.legend(
        handles=t2m_legend_handles,
        fontsize=8,
        loc=[0.01, 0.01],
        handletextpad=0.3,
        columnspacing=1,
        frameon=True,
        ncol=5,
    )

    ax2.text(
        -0.12,
        1.01,
        "(b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    # -----------
    # Plot 3: Scatterplot of Yearly SMB vs. T2m Anomaly (ax3)
    # Set ax3 to be quadratic.
    scatter_handles = []
    cluster_corr = {}
    for cluster in yearly_means_smb.columns:
        y = yearly_means_smb[cluster]
        x = yearly_means_t2m[cluster]
        valid_mask = x.notna() & y.notna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        color = default_colors[cluster - 1]
        if cluster == 1:
            alpha = 0.9
        else:
            alpha = 0.7
        ax3.scatter(x_valid, y_valid, marker="o", color=color, alpha=alpha, s=12)
        if len(x_valid) > 1:
            r, p = pearsonr(x_valid, y_valid)
            significance = "*" if p < 0.05 else ""
            cluster_corr[cluster] = r
        else:
            cluster_corr[cluster] = np.nan
            significance = ""
        label = f"CL{cluster}: r = {cluster_corr[cluster]:.2f}{significance}"
        scatter_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color=color,
                markersize=6,
                label=label,
            )
        )

    overall_significance = "*" if p_all < 0.05 else ""
    overall_label = f"All Cluster: r = {r_all:.2f}{overall_significance}"
    overall_handle = mlines.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        color="grey",
        markersize=6,
        label=overall_label,
    )
    # Append the overall handle to the scatter handles.
    scatter_handles.append(overall_handle)

    # Now create the legend.
    ax3.legend(
        handles=scatter_handles,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        handletextpad=0.3,
        columnspacing=0.3,
        frameon=True,
    )
    ax3.set_xlabel("T2m Anomaly (°C)")
    ax3.set_ylabel("Specific SMB (mm w.e.)")
    ax3.grid(linestyle="--", alpha=0.7)

    # Place the legend centered below ax3.
    ax3.legend(
        handles=scatter_handles,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.3, -0.2),
        handletextpad=0.3,
        columnspacing=0.3,
        frameon=False,
    )

    # Move ax3 upward so its top aligns with ax1.
    pos = ax3.get_position()
    new_pos = [pos.x0, pos.y0 + 0.3, pos.width, pos.height - 0.3]
    ax3.set_position(new_pos)

    ax3.text(
        -0.05,
        1.05,
        "(c)",
        transform=ax3.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    # Save the complete plot to file.
    os.makedirs(output_dir, exist_ok=True)
    plot_filename_png = os.path.join(
        output_dir, "yearly_smb_and_t2m_anomalies_with_scatter.png"
    )
    plot_filename_pdf = os.path.join(
        output_dir, "yearly_smb_and_t2m_anomalies_with_scatter.pdf"
    )
    plt.savefig(plot_filename_png, dpi=300, bbox_inches="tight")
    plt.savefig(plot_filename_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_filename_png}")


def plot_smb_and_t2m_anomalies(file_path_merged_df, output_dir):
    """
    Plot the yearly summer mean Surface Mass Balance (SMB) and Temperature Anomalies (T2m_anom)
    for each cluster with trend lines and significance markers in the legend.

    Parameters:
    - file_path_merged_df (str): Path to the CSV file containing the merged data.
    - output_dir (str): Directory to save the output plot.
    """

    # Load merged_df
    merged_df = pd.read_csv(file_path_merged_df, parse_dates=["Date"])
    merged_df["Year"] = merged_df["Date"].dt.year

    # Compute yearly means for SMB and T2m_anom
    yearly_means_smb = (
        merged_df.groupby(["Year", "Cluster"])["Specific Mass Balance (mm WE)"]
        .mean()
        .unstack()
    )
    yearly_means_t2m = (
        merged_df.groupby(["Year", "Cluster"])["T2m_anom"].mean().unstack()
    )

    # Set up figure with two subplots sharing x-axis
    fig, axs = plt.subplots(
        nrows=2, figsize=(6, 6), sharex=True, gridspec_kw={"hspace": 0.05}
    )

    # Get default colors from matplotlib
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # SMB Plot (Top)
    ax1 = axs[0]
    smb_legend_handles = []
    for cluster in yearly_means_smb.columns:
        x = yearly_means_smb.index
        y = yearly_means_smb[cluster]

        # Remove NaN values
        valid_mask = ~y.isna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]

        if cluster == 1:
            alpha = 0.7
        else:
            alpha = 0.5
        # Scatter plot (consistent marker style)
        ax1.scatter(
            x_valid,
            y_valid,
            marker="o",
            color=default_colors[cluster % len(default_colors)],
            alpha=alpha,
        )

        # Trend line and significance check
        label = f"Cluster {cluster}"
        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
            ax1.plot(
                x_valid,
                np.poly1d(np.polyfit(x_valid, y_valid, 1))(x_valid),
                linestyle="--",
                color=default_colors[cluster % len(default_colors)],
            )
            if p_value < 0.05:
                label += "*"  # Mark significance

        smb_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor=default_colors[cluster % len(default_colors)],
                markersize=6,
                label=label,
            )
        )

    ax1.set_ylabel("SMB (mm WE)")
    ax1.grid(linestyle="--", alpha=0.7)
    ax1.legend(
        handles=smb_legend_handles,
        fontsize=9,
        loc=[1.01, 0.54],
        handletextpad=0.3,
        columnspacing=0.3,
        frameon=True,
    )

    # T2m Anomaly Plot (Bottom)
    ax2 = axs[1]
    t2m_legend_handles = []
    for cluster in yearly_means_t2m.columns:
        x = yearly_means_t2m.index
        y = yearly_means_t2m[cluster]

        # Remove NaN values
        valid_mask = ~y.isna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        if cluster == 1:
            alpha = 0.7
        else:
            alpha = 0.5
        # Scatter plot (same marker style)
        ax2.scatter(
            x_valid,
            y_valid,
            marker="o",
            color=default_colors[cluster % len(default_colors)],
            alpha=alpha,
        )

        # Trend line and significance check
        label = f"Cluster {cluster}"
        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
            ax2.plot(
                x_valid,
                np.poly1d(np.polyfit(x_valid, y_valid, 1))(x_valid),
                linestyle="--",
                color=default_colors[cluster % len(default_colors)],
            )
            if p_value < 0.05:
                label += "*"  # Mark significance

        t2m_legend_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor=default_colors[cluster % len(default_colors)],
                markersize=6,
                label=label,
            )
        )

    ax2.set_ylabel("T2m Anomaly (°C)")
    ax2.set_xlabel("Year")
    ax2.grid(linestyle="--", alpha=0.7)
    ax2.legend(
        handles=t2m_legend_handles,
        fontsize=9,
        loc=[1.01, 0.54],
        handletextpad=0.3,
        columnspacing=0.3,
        frameon=True,
    )

    # Add bold subplot labels "a" and "b" above the plots
    ax1.text(
        -0.1,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
    )
    ax2.text(
        -0.1,
        1.03,
        "(b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
    )

    correlations = {}
    p_values = {}

    all_yearly_means_smb = merged_df.groupby(["Year"])[
        "Specific Mass Balance (mm WE)"
    ].mean()
    all_yearly_means_t2m = merged_df.groupby(["Year"])["T2m_anom"].mean()
    r_all, p_all = pearsonr(all_yearly_means_smb, all_yearly_means_t2m)

    for cluster in yearly_means_smb.columns:
        smb_series = yearly_means_smb[cluster].dropna()
        t2m_series = yearly_means_t2m[cluster].dropna()
        if len(smb_series) > 1 and len(t2m_series) > 1:
            r, p = pearsonr(smb_series, t2m_series)
            correlations[cluster] = r
            p_values[cluster] = p
        else:
            correlations[cluster] = np.nan
            p_values[cluster] = np.nan

    # Create a DataFrame for the results
    correlation_df = pd.DataFrame(
        {
            "Cluster": list(correlations.keys()) + ["All"],
            "Correlation (r)": list(correlations.values()) + [r_all],
            "p-value": list(p_values.values()) + [p_all],
        }
    )

    # Convert DataFrame to HTML
    table_html = correlation_df.to_html(index=False, float_format="%.3f")

    # Save the table to an HTML file
    with open("results\\k_means\\annual_JJA_corr_t2m_smb.html", "w") as f:
        f.write(table_html)

    print("HTML table saved as correlation_results.html")

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename_png = os.path.join(output_dir, "yearly_smb_and_t2m_anomalies.png")
    plot_filename_pdf = os.path.join(output_dir, "yearly_smb_and_t2m_anomalies.pdf")
    plt.savefig(plot_filename_png, dpi=300, bbox_inches="tight")
    plt.savefig(plot_filename_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_filename_png}")


def plot_yearly_smb_with_t2m_anomalies(file_path_merged_df, output_dir):
    """
    Plots yearly mean specific mass balance (SMB) for each cluster, marking points by
    t2m anomaly with a colormap and adding trend lines with significance stars.

    Parameters:
        file_path_merged_df (str): Path to the merged data CSV file.
        t2m_anomaly_file (str): Path to the t2m anomaly CSV file.
        output_dir (str): Directory where the plot will be saved.
    """

    merged_t_df = pd.read_csv(file_path_merged_df, parse_dates=["Date"])

    # Add year column
    merged_t_df["Year"] = merged_t_df["Date"].dt.year

    # Calculate yearly means
    yearly_means = merged_t_df.groupby(["Year", "Cluster"]).mean().unstack()

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))

    # Get the default color cycle from rcParams
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Assign default colors to clusters based on the predefined color cycle
    cluster_colors = {
        cluster: default_colors[cluster % len(default_colors)]
        for cluster in yearly_means["Specific Mass Balance (mm WE)"].columns
    }

    # Dictionary to store significance for each cluster
    significant_clusters = {}

    # Define marker styles for clusters
    markers = ["o", "s", "D", "^", "*", "h"]
    cmap = plt.cm.bwr
    for cluster in yearly_means["Specific Mass Balance (mm WE)"].columns:
        # Define marker style and color
        marker_style = markers[cluster % len(markers)]
        cluster_color = cluster_colors[cluster]  # Assign pre-defined color

        # Extract data
        x = yearly_means.index
        y = yearly_means["Specific Mass Balance (mm WE)", cluster]
        t2m_anom = yearly_means["T2m_anom", cluster]

        valid_mask = ~y.isna()
        x_valid, y_valid, t2m_anom_valid = (
            x[valid_mask],
            y[valid_mask],
            t2m_anom[valid_mask],
        )

        # Scatter plot
        scatter = ax.scatter(
            x_valid,
            y_valid,
            c=t2m_anom_valid,
            cmap=cmap,
            vmin=-4,
            vmax=4,
            edgecolor="black",
            linewidth=0.5,
            marker=marker_style,
            label=f"Cluster {cluster}",
        )

        # Add trend line with its specific color
        if len(x_valid) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
            significant_clusters[cluster] = p_value < 0.05

            ax.plot(
                x_valid,
                np.poly1d(np.polyfit(x_valid, y_valid, 1))(x_valid),
                linestyle="--",
                color=cluster_color,  # Keep trendline color unique
            )

    # Create legend handles with filled markers matching trendline colors
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker=markers[cluster % len(markers)],
            color="black",
            markerfacecolor=cluster_colors[cluster],  # Use trendline color
            markersize=6,
            linestyle="None",
            label=f"Cluster {cluster}{'*' if significant_clusters.get(cluster, False) else ''}",
        )
        for cluster in yearly_means["Specific Mass Balance (mm WE)"].columns
    ]

    ax.legend(handles=legend_handles, fontsize=10, loc="lower left")

    # Colorbar for t2m anomaly
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("t 2m Anomaly [°C]", fontsize=12)

    # Final plot adjustments
    ax.set_ylabel("Summer Mean SMB (mm WE)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(
        output_dir, "yearly_smb_by_cluster_with_trends_with_t_anom.png"
    )
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_filename}")


def daily_means_per_cluster(
    input_file, output_dir, gaussian_sigma=8, plot_individual_points=False
):
    """
    Reads a CSV file with mass balance and cluster data, calculates daily means per cluster,
    and saves scatter plot.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_dir (str): Directory where the plot will be saved.
    - gaussian_sigma (float): Standard deviation of the Gaussian filter for smoothing.
    - plot_individual_points (bool): Whether to plot all individual data points in cluster colors.
    """

    # Load data
    merged_df = pd.read_csv(input_file, parse_dates=["Date"])

    # Add Day of Year (DOY) column
    merged_df["DOY"] = merged_df["Date"].dt.day_of_year

    # Calculate daily means
    daily_means = (
        merged_df.groupby(["DOY", "Cluster"])["Specific Mass Balance (mm WE)"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for cluster in daily_means.columns:
        # Define color for the cluster
        cluster_color = plt.cm.tab10(
            cluster % 10
        )  # Use tab10 colormap to cycle through colors

        # Extract data for the current cluster
        x = daily_means.index
        y = daily_means[cluster]

        # Remove NaN values for trend line calculation
        valid_mask = ~y.isna()
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # Smooth the data using a Gaussian filter
        y_smoothed = gaussian_filter(y_valid, sigma=gaussian_sigma)

        # Plot smoothed lines
        ax.plot(
            x_valid,
            y_smoothed,
            label=f"Cluster {cluster}",
            linewidth=1.8,
            color=cluster_color,
        )

        # Optionally plot raw data points
        if plot_individual_points:
            cluster_data = merged_df[merged_df["Cluster"] == cluster]
            ax.scatter(
                cluster_data["DOY"],
                cluster_data["Specific Mass Balance (mm WE)"],
                marker="o",
                s=8,  # Smaller markers
                color=cluster_color,
                alpha=0.3,  # More transparent
            )
        else:
            ax.scatter(
                x,
                y,
                marker="o",
                s=18,
                color=cluster_color,
                alpha=0.5,
            )

    # Customize plot
    ax.legend()
    ax.set_ylabel("Daily Mean SMB (mm WE)", fontsize=12)
    ax.set_xlabel("Day of Year", fontsize=12)
    ax.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure with appropriate file name
    file_suffix = "_with_individual_points" if plot_individual_points else ""
    output_path = os.path.join(output_dir, f"daily_means_scatter_plot{file_suffix}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


def plot_smb_vs_elevation(input_file, output_dir):
    """
    Reads a CSV file, processes the data, and saves a plot of Mean SMB vs. Mean Elevation.

    Parameters:
        csv_file (str): Path to the CSV file.
        save_path (str): Path to save the output plot (e.g., 'output.png')
        hypso_stat (bool): Whether to calculate hypsometric effect statistics.
    """
    # Load data
    df = pd.read_csv(input_file)

    # Convert 'Time' column to datetime
    df["Time"] = pd.to_datetime(df["Time"])

    # Define bin edges and labels
    bins = [1990, 2002, 2013, 2024]  # Bin edges (right-inclusive)
    labels = [1991, 2003, 2014]  # Representative year labels

    # Create the 'Year_Bin' column
    df["Year_Bin"] = pd.cut(df["Time"].dt.year, bins=bins, labels=labels, right=True)

    # Convert to integer for easier handling
    df["Year_Bin"] = df["Year_Bin"].astype(int)
    df_grouped = (
        df.groupby(["Year_Bin", "elev_bin"])
        .agg(
            {
                "SMB": "sum",  # Sum for SMB
                "Total_Area_km2": "mean",  # Mean for Total_Area_km2
                "Mean_Elevation": "mean",
            }
        )
        .reset_index()
    )

    # Define the number of years in each bin
    years_in_bin = {1991: 12, 2003: 11, 2014: 11}

    # Apply division to normalize SMB annually by years in each bin
    df_grouped["SMB_per_year"] = df_grouped.apply(
        lambda row: row["SMB"] / years_in_bin[row["Year_Bin"]], axis=1
    )

    # Define manual elevation bins
    elevation_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Compute bin midpoints and widths
    midpoints = [
        (elevation_bins[i] + elevation_bins[i + 1]) / 2
        for i in range(len(elevation_bins) - 1)
    ]
    widths = np.diff(elevation_bins)  # Compute bar widths

    # Create a dictionary mapping each elev_bin to its midpoint
    elev_bin_to_midpoint = {i: midpoints[i] for i in range(len(midpoints))}

    # Assign midpoints and widths to df_grouped based on elev_bin
    df_grouped["Bar_Center"] = df_grouped["elev_bin"].map(elev_bin_to_midpoint)
    df_grouped["Bar_Width"] = df_grouped["elev_bin"].map(
        lambda x: widths[x] - 2 if x < len(widths) else widths[-1]
    )

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot bars for Total Area at midpoints
    ax1.bar(
        df_grouped["Bar_Center"],
        df_grouped["Total_Area_km2"],
        width=df_grouped["Bar_Width"],
        alpha=1,
        color="grey",
        label="Area",
    )

    # Create secondary y-axis for SMB
    ax2 = ax1.twinx()

    # Create a sequential color palette
    palette = sns.color_palette("cool", as_cmap=True)

    # Normalize colors based on Year_Bin
    norm = plt.Normalize(df_grouped["Year_Bin"].min(), df_grouped["Year_Bin"].max())

    # Loop through each 5-year bin and plot SMB vs. bin midpoints
    for year_bin, group in df_grouped.groupby("Year_Bin"):
        if year_bin == 1991:
            label = "1991-2002"
        else:
            label = f"{year_bin}-{year_bin+10}"
        ax2.plot(
            group["Mean_Elevation"],
            group["SMB_per_year"],
            marker="o",
            linestyle="-",
            color=palette(norm(year_bin)),
            label=label,
        )

    # Labels and title
    ax1.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Annual mean SMB (mm we)")
    ax1.set_ylabel("Area (km²)")
    ax1.set_ylim(0, 2000)

    # Legends
    ax1.legend(loc="upper right", bbox_to_anchor=(0.2, 0.95))

    # Move ax2 legend below the plot (horizontal layout)
    ax2.legend(
        title="SMB Periods", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4
    )
    # Add horizontal line at ax2 == 0
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.7)

    # Grid and layout
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "annual_smb_vs_elevation.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_smb_doy_cluster(mar_elev_bin_file, cluster_file, output_dir):
    """
    Plots the mean SMB over Day of Year (DOY) for each elevation bin and plot SMB per cluster over elevation.
    Parameters:
    - mar_elev_bin_file (str): Path to the CSV file containing the MAR data with elevation bins.
    - cluster_file (str): Path to the CSV file containing the cluster assignments.
    - output_dir (str): Directory to save the output plot.
    """

    # Load data
    df_cluster = pd.read_csv(cluster_file)
    df_mar = pd.read_csv(mar_elev_bin_file)

    # Convert time columns to datetime format
    df_mar["Time"] = pd.to_datetime(df_mar["Time"])
    df_cluster["Date"] = pd.to_datetime(df_cluster["Date"])

    # Merge on matching dates, keeping all rows from df_mar for those dates
    merged_df = df_mar.merge(
        df_cluster, left_on="Time", right_on="Date", how="inner"
    ).drop(columns=["Date"])
    merged_df["DOY"] = merged_df["Time"].dt.day_of_year

    # Define elevation bin labels
    elev_bin_labels = {i: f"{i*100}-{(i+1)*100}m" for i in range(10)}

    # Compute mean SMB per DOY for each elevation bin
    df_grouped = merged_df.groupby(["elev_bin", "DOY"])["SMB"].mean().reset_index()

    # Normalize elevation bins for viridis colormap
    norm = mcolors.Normalize(vmin=0, vmax=9)  # Elevation bins range from 0 to 9
    cmap = cm.viridis  # Use the viridis colormap

    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot SMB over DOY for each elevation bin on ax1
    for elev_bin, data in df_grouped.groupby("elev_bin"):
        color = cmap(norm(elev_bin))  # Assign color using viridis colormap
        label = elev_bin_labels.get(elev_bin, f"Elev {elev_bin}")  # Use custom label
        ax1.plot(data["DOY"], data["SMB"], label=label, color=color)

    ax1.set_xlabel("Day of Year (June-August)")
    ax1.set_ylabel("Mean SMB (mm we)")
    ax1.grid(True)

    # Compute mean SMB per Elevation Bin and Cluster for ax2
    df_cluster_grouped = (
        merged_df.groupby(["Cluster", "Mean_Elevation"])["SMB"].mean().reset_index()
    )

    # Plot SMB per Cluster vs Elevation on ax2
    for cluster, data in df_cluster_grouped.groupby("Cluster"):
        ax2.plot(
            data["Mean_Elevation"],
            data["SMB"],
            label=f"Cluster {cluster}",
            # linestyle="-",
            marker="o",
        )

    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Mean SMB (mm we)")
    ax2.grid(True)
    # Create common legend below both plots
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(
        handles1,
        labels1,
        loc="lower center",
        bbox_to_anchor=(0.25, -0.1),
        ncol=5,
        fontsize="small",
        frameon=True,
        columnspacing=0.5,
    )

    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles2,
        labels2,
        loc="lower center",
        bbox_to_anchor=(0.75, -0.1),
        ncol=3,
        fontsize="small",
        frameon=True,
        columnspacing=0.5,
    )
    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "smb_doy_cluster.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {save_path}")


def plot_combined_climatology_and_correlation(
    input_file_climatology,
    input_file_anomalies,
    snow_fraction_file,
    output_dir,
    vmin=-2,
    vmax=2,
    ylim=500,
    smooth_window=5,
    p_value_sigma=0.5,
    correlation_filter_sigma=1.2,
    p_value_thresh=0.05,
):
    """
    Creates a combined figure with two subplots:
      - Left subplot: Climatology (mean temperature differences) heatmap with an overlaid plot
                      of smoothed snow fraction climatology.
      - Right subplot: Correlation heatmap between snow fraction anomalies and temperature difference anomalies.

    Parameters:
      input_directory_climatology : Directory containing CSV files for the climatology plot.
      input_directory_anomalies    : Directory containing CSV files for the correlation plot.
      snow_fraction_file           : Path to the CSV file containing snow fraction data.
      vmin, vmax                   : Color limits for the temperature difference heatmap.
      ylim                         : Maximum height value for the y-axis.
      smooth_window                : Window size for smoothing the climatology data.
      p_value_sigma                : Gaussian filter sigma for p-value smoothing (correlation plot).
      correlation_filter_sigma     : Gaussian filter sigma for correlation smoothing.
      p_value_thresh               : P-value threshold for marking significance in the correlation plot.
    """

    # Create a figure with 2 subplots (left and right) sharing the y-axis.
    # Adjust the width ratios (here left=1.1 and right=0.9) as desired.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]}
    )

    # function for gaussian smoothing
    def smooth_dataframe(df, sigma):
        smoothed_array = gaussian_filter(df.to_numpy(), sigma=sigma)
        return pd.DataFrame(smoothed_array, index=df.index, columns=df.columns)

    # -------------------------------
    # Left subplot: Climatology Plot
    # -------------------------------
    # Load snow fraction data for the overlay.
    df_snow = pd.read_csv(
        snow_fraction_file, index_col="time", parse_dates=True, dayfirst=True
    )
    df_snow.index = pd.to_datetime(df_snow.index)
    df_snow_summer = df_snow[df_snow.index.month.isin([6, 7, 8])]
    df_snow_summer["doy"] = df_snow_summer.index.dayofyear
    df_daily_mean = df_snow_summer.groupby("doy")["snow_fraction"].mean()
    df_daily_mean_smoothed = df_daily_mean.rolling(
        window=smooth_window, center=True
    ).mean()

    # Read climatology file.
    df_clim = pd.read_csv(
        input_file_climatology, index_col=0, parse_dates=True, dayfirst=True
    )
    df_clim.index = pd.to_datetime(df_clim.index)
    df_summer = df_clim[df_clim.index.month.isin([6, 7, 8])]
    climatology = df_summer.groupby(df_summer.index.dayofyear).mean()
    climatology_smoothed = smooth_dataframe(climatology, sigma=correlation_filter_sigma)

    # Create meshgrid for pcolormesh.
    X_left, Y_left = np.meshgrid(
        climatology_smoothed.index, climatology_smoothed.columns.astype(int)
    )
    Z_left = climatology_smoothed.values.T

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    c = ax1.pcolormesh(
        X_left, Y_left, Z_left, cmap="coolwarm", shading="auto", norm=norm
    )

    cbar = plt.colorbar(
        c, ax=ax1, label="Temperature Difference (°C)", shrink=0.65, pad=0.11
    )
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("Temperature Difference (°C)", fontsize=14)

    # Overlay the snow fraction on a twin axis.
    ax1_twin = ax1.twinx()
    (line_sf,) = ax1_twin.plot(
        df_daily_mean.index,
        df_daily_mean_smoothed,
        color="darkgreen",
        label="Snow Fraction",
        linewidth=2,
    )
    ax1_twin.set_ylabel("Snow Fraction", color="darkgreen", fontsize=14)
    ax1_twin.set_ylim(0, 1)
    ax1_twin.legend(handles=[line_sf], loc="upper right")

    ax1.set_xlabel("Day of Year (June-August)", fontsize=14)
    ax1.set_ylabel("Altitude Above Ground (m)", color="black", fontsize=14)
    ax1.set_yticks(np.arange(0, ylim + 1, step=ylim // 10))
    ax1.set_ylim(0, ylim)
    ax1.grid(True)

    # ----------------------------------
    # Right subplot: Correlation Heatmap
    # ----------------------------------
    # Compute anomalies from snow fraction data.
    snow_df = pd.read_csv(snow_fraction_file, parse_dates=["time"])
    snow_df["DOY"] = snow_df["time"].dt.day_of_year
    typical_snow_fraction = snow_df.groupby("DOY")["snow_fraction"].mean()
    typical_snow_fraction_smoothed = sm.nonparametric.lowess(
        typical_snow_fraction, typical_snow_fraction.index, frac=0.1
    )
    typical_snow_fraction_smoothed = typical_snow_fraction_smoothed[:, 1]
    typical_snow_fraction_df = pd.DataFrame(
        {
            "DOY": typical_snow_fraction.index,
            "snow_fraction_typical": typical_snow_fraction_smoothed,
        }
    )
    snow_df = snow_df.merge(typical_snow_fraction_df, on="DOY", how="left")
    snow_df["anomaly"] = snow_df["snow_fraction"] - snow_df["snow_fraction_typical"]

    # Read anomalies file.
    df_anom = pd.read_csv(
        input_file_anomalies, index_col=0, parse_dates=True, dayfirst=True
    )
    df_anom.sort_index(inplace=True)
    df_anom["DOY"] = df_anom.index.day_of_year
    snow_df_reindexed = snow_df.set_index("time").reindex(
        df_anom.index, fill_value=np.nan
    )
    df_anom["snow_anomaly"] = snow_df_reindexed["anomaly"]

    min_doy = df_anom["DOY"].min()
    max_doy = df_anom["DOY"].max()
    columns_to_correlate = [
        col for col in df_anom.columns if col not in ["DOY", "snow_anomaly"]
    ]

    # Build dataframes for correlation coefficients and p-values.
    correlation_results = pd.DataFrame(
        index=range(min_doy, max_doy + 1), columns=columns_to_correlate
    )
    correlation_p = pd.DataFrame(
        index=range(min_doy, max_doy + 1), columns=columns_to_correlate
    )

    for doy, group in df_anom.groupby("DOY"):
        group = group.dropna(subset=["snow_anomaly"])
        for col in columns_to_correlate:
            corr, p_value = pearsonr(group["snow_anomaly"], group[col])
            correlation_results.loc[doy, col] = corr
            correlation_p.loc[doy, col] = p_value

    correlation_results = correlation_results.astype(float)
    correlation_p = correlation_p.astype(float)
    correlation_results_smoothed = smooth_dataframe(
        correlation_results, sigma=correlation_filter_sigma
    )
    correlation_p_smoothed = smooth_dataframe(correlation_p, sigma=p_value_sigma)

    # Plot the correlation heatmap.
    im = ax2.imshow(
        correlation_results_smoothed.T,
        aspect="auto",
        cmap="bwr",
        vmin=-1,
        vmax=1,
        extent=[
            correlation_results.index.min(),
            correlation_results.index.max(),
            correlation_results.shape[1],
            0,
        ],
    )
    # Create the colorbar
    cbar2 = plt.colorbar(
        im, ax=ax2, label="Correlation Coefficient", shrink=0.65, pad=0.12
    )

    # Set the label size
    cbar2.ax.tick_params(labelsize=12)
    cbar2.ax.set_ylabel("Correlation Coefficient", fontsize=14)

    # Add the right-hand y-axis for altitude on the right subplot
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Altitude Above Ground (m)", color="black", fontsize=14)
    ax2_twin.set_ylim(0, ylim)  # Ensure the y-axis limits match ax1
    ax2_twin.set_yticks(np.arange(0, ylim + 1, step=ylim // 10))  # Match ticks with ax1

    # Overlay a contourf with diagonal hatching on areas where p < threshold.
    mask = (correlation_p_smoothed < p_value_thresh).astype(float)
    cf = ax2.contourf(
        X_left, Y_left, mask.T, levels=[0.5, 1.5], hatches=["///"], alpha=0
    )

    # Add a legend entry for the hatched (significant) areas.
    signif_patch = Patch(
        facecolor="none",
        hatch="///",
        edgecolor="yellow",
        label="Significant (p < {:.2f})".format(p_value_thresh),
    )
    ax2.legend(handles=[signif_patch], loc="upper right")

    ax2.set_xlabel("Day of Year (June-August)", fontsize=14)
    ax2.set_ylabel("")
    ax2.axhline(y=100, color="grey", linestyle="--", linewidth=1.5, zorder=3)

    if ylim:
        ax2.set_ylim(0, ylim)

    ax1.text(
        -0.1,
        1.05,
        "(a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.text(
        -0.02,
        1.05,
        "(b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.grid(True)
    # tight layout
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.01)

    # save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path_png = os.path.join(output_dir, "combined_climatology_correlation.png")
    save_path_pdf = os.path.join(output_dir, "combined_climatology_correlation.pdf")
    plt.savefig(save_path_png, bbox_inches="tight", dpi=300)
    plt.savefig(save_path_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {save_path_png}")
    plt.close()


def plot_combined_synoptic_and_occurrences(
    clusters_df_path,
    anomaly_file_path,
    profile_times_path,
    output_path,
    vmin=-2.5,
    vmax=2.5,
):
    """
    Combines two figures into one canvas with two rows.

    Top row: Mean anomalies per cluster (one row, 5 columns).
    Bottom row: Three subplots for cluster occurrence metrics (normalized daily, annual, and a single stacked bar plot)
    """
    # ---------------------------
    # Top Row: Anomalies per Cluster
    # ---------------------------
    # Read clusters and prepare time string
    clusters_df = pd.read_csv(clusters_df_path)
    clusters_df["Date"] = pd.to_datetime(clusters_df["Date"])
    clusters_df["Date"] = clusters_df["Date"].dt.strftime("%Y-%m-%dT12:00:00")

    # Start counting Clusters at 1
    # Load anomaly dataset and convert valid_time to string
    ds = xr.open_dataset(anomaly_file_path)
    ds["valid_time"] = ds["valid_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Merge K-means results with the anomaly dataset based on valid_time/Date
    merged_data = pd.DataFrame({"valid_time": ds["valid_time"].values}).merge(
        clusters_df,
        how="left",
        left_on="valid_time",
        right_on="Date",
    )

    # Assign clusters as a new coordinate in the dataset
    ds = ds.assign_coords(
        cluster=("valid_time", merged_data["Cluster"].fillna(-1).values)
    )

    # Determine valid clusters (assumed to be 5 clusters)
    valid_clusters = np.sort(merged_data["Cluster"].dropna().unique())
    cluster_counts = merged_data["Cluster"].value_counts()

    # Compute the data extent (longitude and latitude bounds) from the dataset
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    data_extent = [lon_min, lon_max, lat_min, lat_max]

    # ---------------------------
    # Bottom Row: Cluster Occurrence Plots
    # ---------------------------
    # For the occurrence plots, read the clusters file again and set datetime index.
    clusters_occ = pd.read_csv(clusters_df_path)
    clusters_occ["Date"] = pd.to_datetime(clusters_occ["Date"])
    clusters_occ.set_index("Date", inplace=True)
    clusters_occ["DayOfYear"] = clusters_occ.index.dayofyear

    # Count occurrences per DayOfYear and cluster
    daily_occurrences = (
        clusters_occ.groupby([clusters_occ["DayOfYear"], "Cluster"])
        .size()
        .unstack(fill_value=0)
    )
    day_of_year_counts = clusters_occ.groupby("DayOfYear").size()
    normalized_daily_occurrences = daily_occurrences.div(day_of_year_counts, axis=0)

    # Annual occurrences per cluster
    clusters_occ["Year"] = clusters_occ.index.year
    annual_occurrences = (
        clusters_occ.groupby(["Year", "Cluster"]).size().unstack(fill_value=0)
    )

    # Load profile time data for the single stacked bar plot
    profile_times_df = pd.read_csv(profile_times_path)
    profile_times_df["Time"] = pd.to_datetime(profile_times_df["Time"])
    profile_times_df["Date"] = profile_times_df["Time"].dt.date
    unique_dates = profile_times_df["Date"].unique()
    unique_dates_df = pd.DataFrame(unique_dates, columns=["Date"])

    # Merge clusters with profile times (only on field days)
    clusters_occ.index = pd.to_datetime(clusters_occ.index).date
    merged_occ_df = pd.merge(
        clusters_occ, unique_dates_df, left_index=True, right_on="Date", how="inner"
    )
    occurrence_cluster_counts = merged_occ_df["Cluster"].value_counts()
    total_count = occurrence_cluster_counts.sum()
    cluster_ratios = occurrence_cluster_counts / total_count
    cluster_ratios_df = pd.DataFrame(cluster_ratios).T
    sorted_columns = sorted(cluster_ratios_df.columns)
    cluster_ratios_df = cluster_ratios_df.reindex(columns=sorted_columns)

    # Define a consistent color palette for the clusters (assumes clusters are numeric)
    unique_clusters = sorted(clusters_occ["Cluster"].unique())
    palette = sns.color_palette(n_colors=len(unique_clusters))
    cluster_palette = {
        cluster: color for cluster, color in zip(unique_clusters, palette)
    }

    # ---------------------------
    # Create the Combined Figure with Two Rows
    # ---------------------------
    # Create a figure and a GridSpec with 2 rows.
    fig = plt.figure(figsize=(10, 6))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)

    # --- Top row: 5 columns for anomalies per cluster ---
    gs_top = outer_gs[0].subgridspec(1, 5, wspace=0.2)

    # Loop over each valid cluster to create an anomalies map
    im = None  # To capture the last image for the shared colorbar
    for i, cluster_id in enumerate(valid_clusters):
        # Use PlateCarree for the map projection
        ax = fig.add_subplot(gs_top[0, i], projection=ccrs.PlateCarree())

        # Restrict the map to your data region and set a white background
        ax.set_extent(data_extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("white")

        # Calculate the mean anomalies for the given cluster
        t_anomaly = (
            ds["t_anomaly"].where(ds["cluster"] == cluster_id).mean(dim="valid_time")
        )
        z_anomaly = (
            ds["z_anomaly"].where(ds["cluster"] == cluster_id).mean(dim="valid_time")
        )

        # Plot temperature anomaly with PlateCarree transform (data are in lat/lon)
        im = t_anomaly.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_aspect("auto")

        # Add geographic features only within the data extent
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, edgecolor="black")

        # Add gridlines; disable labels on inner panels for clarity
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle="--"
        )
        gl.top_labels = False
        gl.buttom_labels = True
        gl.right_labels = False
        if i != 0:
            gl.left_labels = False

        # Contour geopotential height anomaly
        z_min = np.floor(z_anomaly.min().values / 5) * 5
        z_max = np.ceil(z_anomaly.max().values / 5) * 5 + 5
        contour_levels = np.arange(z_min, z_max, 5)
        contours = ax.contour(
            z_anomaly.longitude,
            z_anomaly.latitude,
            z_anomaly.values,
            levels=contour_levels,
            colors="black",
            linewidths=1,
            transform=ccrs.PlateCarree(),
            # only solid lines
            linestyles="solid",
        )
        ax.clabel(contours, inline=True, fontsize=9, fmt="%1.0f", zorder=2)

        # Mark a specific coordinate (e.g., VRS)
        longitude = -16.636666666666667
        latitude = 81.59916666666666
        ax.plot(
            longitude,
            latitude,
            marker="*",
            color="yellow",
            markersize=8,
            transform=ccrs.PlateCarree(),
        )

        # custom annotations VRS
        if i == 2 or i == 3:
            ax.text(
                longitude,
                latitude,
                " VRS",
                fontsize=12,
                color="black",
                transform=ccrs.PlateCarree(),
                verticalalignment="top",
                horizontalalignment="left",
            )
        else:
            ax.text(
                longitude,
                latitude,
                " VRS",
                fontsize=12,
                color="black",
                transform=ccrs.PlateCarree(),
                verticalalignment="bottom",
                horizontalalignment="left",
            )

        count = cluster_counts.get(cluster_id, 0)
        # Annotation with both the count and GpH anomaly line
        annotation = f"n = {count}"
        ax.text(
            0.98,
            0.985,  # top-right corner
            annotation,
            transform=ax.transAxes,
            fontsize=11,
            color="black",
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white", edgecolor="black", alpha=1, boxstyle="round,pad=0.2"
            ),
        )
        # Define a dictionary to map cluster IDs to their descriptions
        cluster_descriptions = {
            1: "Low Pressure",
            2: "Zonal",
            3: "High Pressure",
            4: "Strong Zonal",
            5: "Strong High Pressure",
        }
        description = cluster_descriptions.get(cluster_id, "")

        ax.set_title(
            f"CL{int(cluster_id)}\n'{description}'",
            fontsize=12,
            fontweight="bold",
            color=cluster_palette.get(cluster_id, "black"),
        )

        # Add bold 'a' to the upper-left of the first-row subplots
        if i == 0:  # Assuming 5 subplots per row
            ax.text(
                -0.15,
                1.1,  # Slightly outside the top-left corner
                "(a)",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="center",
            )

    # Add a shared vertical colorbar for the anomalies maps.
    cbar_ax = fig.add_axes([0.91, 0.53, 0.015, 0.35])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="vertical", label="850 hPa Temp. Anomaly (°C)"
    )
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("850 hPa Temp. Anomaly (°C)", fontsize=10)

    # --- Bottom row: Occurrence Plots (3 panels) ---
    # Reduce the horizontal spacing since we no longer need room for y-axis labels.
    gs_bot = outer_gs[1].subgridspec(1, 7, wspace=0.3)  # Reduced from 0.9 to 0.1
    ax_norm = fig.add_subplot(gs_bot[0, :3])
    ax_ann = fig.add_subplot(
        gs_bot[0, 3:6], sharey=ax_norm
    )  # share y-axis with ax_norm
    ax_bar = fig.add_subplot(gs_bot[0, 6:], sharey=ax_norm)  # share y-axis with ax_norm

    # Add bold subplot labels "b", "c", and "d" above the three occurrence plots.
    ax_norm.text(
        -0.032,
        1.1,
        "(b)",
        transform=ax_norm.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_ann.text(
        -0.03,
        1.1,
        "(c)",
        transform=ax_ann.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax_bar.text(
        -0.035,
        1.1,
        "(d)",
        transform=ax_bar.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Plot 1: Normalized Cluster Occurrences Per Day of Year
    normalized_daily_occurrences.plot(
        kind="bar",
        stacked=True,
        ax=ax_norm,
        width=0.9,
        color=[cluster_palette.get(x) for x in normalized_daily_occurrences.columns],
    )
    ax_norm.set_ylabel("Relative Cluster Occurrence", fontsize=10)
    ax_norm.set_xlabel("Day of Year (June-August)", fontsize=10)
    ax_norm.set_xticks(range(0, len(normalized_daily_occurrences), 5))
    ax_norm.tick_params(axis="x", labelsize=10)

    # Prepare relative data for annual occurrences by dividing by 92.
    annual_occurrences_relative = annual_occurrences / 92

    # Average the cluster counts for first and second halve of period

    first_half_abs = (
        annual_occurrences.loc[annual_occurrences.index < 2008].mean().round(2)
    )
    second_half_abs = (
        annual_occurrences.loc[annual_occurrences.index >= 2008].mean().round(2)
    )

    first_halve_rel = (
        annual_occurrences_relative.loc[annual_occurrences_relative.index < 2008]
        .mean()
        .round(2)
    )
    second_halve_rel = (
        annual_occurrences_relative.loc[annual_occurrences_relative.index >= 2008]
        .mean()
        .round(2)
    )

    # writing average clusters to dataframe
    average_clusters = pd.DataFrame(
        {
            "1991-2007 Abs": first_half_abs,
            "2008-2024 Abs": second_half_abs,
            "1991-2007 Rel": first_halve_rel,
            "2008-2024 Rel": second_halve_rel,
        }
    ).T

    # save average clusters to html table
    average_clusters.to_html("results\\k_means\\cluster_occurence_over_time.html")

    # Plot 2: Annual Cluster Occurrence (Relative)
    annual_occurrences_relative.plot(
        kind="bar",
        stacked=True,
        ax=ax_ann,
        width=0.9,
        color=[cluster_palette.get(x) for x in annual_occurrences_relative.columns],
    )
    ax_ann.set_xticks(range(0, len(annual_occurrences_relative), 5))
    ax_ann.set_xlabel("Year", fontsize=10)
    ax_ann.tick_params(axis="x", labelsize=10)
    # Remove the redundant y-axis label on ax_ann since y-axis is shared.
    ax_ann.set_ylabel("")

    # Plot 3: Single Stacked Bar Plot for Cluster Occurrence on Fielddays
    cluster_order = [1, 2, 3, 5]  # use a specific stacking order
    cluster_colors = [cluster_palette.get(x) for x in cluster_order]
    cluster_ratios_df = cluster_ratios_df[cluster_order]  # reorder columns if needed
    cluster_ratios_df.plot(
        kind="bar",
        stacked=True,
        ax=ax_bar,
        width=0.9,
        color=cluster_colors,
    )
    bottom_val = 0
    for cluster in cluster_order:
        count = occurrence_cluster_counts.get(cluster, 0)
        ratio = count / total_count if total_count != 0 else 0
        ax_bar.text(
            0,
            bottom_val + ratio / 2,
            f"n={count}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        bottom_val += ratio
    ax_bar.set_xlabel("Clusters on\nField Days", fontsize=10)
    ax_bar.set_xticklabels([])

    # Remove the redundant y-axis label on ax_bar since y-axis is shared.
    ax_bar.set_ylabel("")

    # (Optionally) adjust the y-axis limits so that all values are between 0 and 1.
    ax_norm.set_ylim(0, 1)

    # Create a combined legend using handles from one of the occurrence plots
    handles, labels = ax_norm.get_legend_handles_labels()
    labels = [f"CL{label}" for label in labels]
    ax_norm.get_legend().remove()
    ax_ann.get_legend().remove()
    ax_bar.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(labels),
        fontsize=11,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

    # Ensure the output directory exists and save the figure
    os.makedirs(output_path, exist_ok=True)
    output_file_png = os.path.join(output_path, "combined_synoptic_and_occurrences.png")
    output_file_pdf = os.path.join(output_path, "combined_synoptic_and_occurrences.pdf")
    plt.savefig(output_file_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {output_file_png}")


def plot_smb_cluster_elevation(
    df_mar_cluster_merged, df_mar_elev, cluster_file, output_dir
):
    """
    plots Daily Mean SMB per cluster over DOY left and Mean SMB over elevation per cluster right.
    Parameters:
    - df_mar_cluster_merged (str): Path to the CSV file containing the merged MAR cluster data.
    - df_mar_elev (str): Path to the CSV file containing the MAR elevation bins data.
    - cluster_file (str): Path to the CSV file containing the cluster assignments.
    - output_dir (str): Directory to save the output plot.
    """

    # Load mar cluster merged file
    merged_df = pd.read_csv(df_mar_cluster_merged, parse_dates=["Date"])
    merged_df["DOY"] = merged_df["Date"].dt.day_of_year

    # Calculate daily means for Cluster SMB over DOY
    daily_means = (
        merged_df.groupby(["DOY", "Cluster"])["Specific Mass Balance (mm WE)"]
        .mean()
        .unstack()
    )

    # Load cluster file and mar elevation bins file and merge them
    df_cluster = pd.read_csv(cluster_file)
    df_mar = pd.read_csv(df_mar_elev)
    df_mar["Time"] = pd.to_datetime(df_mar["Time"])
    df_cluster["Date"] = pd.to_datetime(df_cluster["Date"])

    # Merge data and compute necessary metrics
    merged_df = df_mar.merge(
        df_cluster, left_on="Time", right_on="Date", how="inner"
    ).drop(columns=["Date"])
    merged_df["DOY"] = merged_df["Time"].dt.day_of_year
    df_cluster_grouped = (
        merged_df.groupby(["Cluster", "Mean_Elevation"])["SMB"].mean().reset_index()
    )

    def custom_rolling_mean(series, full_window, min_points):
        """
        Compute a custom rolling mean with a flexible window size.

        For indices where a full centered window (of length `full_window`) is available,
        that full window is used.

        For indices near the edges where a full window is not available, a fallback window is used.
        If the number of data points in the fallback window is less than `min_points`,
        the output for that index is set to np.nan.

        For example, with min_points=3:
        - At index 0, if less than 3 data points are available, np.nan is returned.
        - At index 1, only indices 0–2 (3 points) are averaged (if exactly 3 are available).

        Parameters:
        series (array-like): 1D data (e.g. a Pandas Series or numpy array)
        full_window (int): Desired window size for the center of the data (should be odd)
        min_points (int): Minimum number of points required for smoothing at an index.

        Returns:
        np.ndarray: The smoothed data (with np.nan where not enough data is available).
        """
        arr = np.asarray(series)
        n = len(arr)

        # Pre-allocate output.
        smoothed = np.empty(n)

        # Calculate half window size (integer division)
        half = full_window // 2  # e.g., if full_window=25, half=12

        for i in range(n):
            # For indices where a full window is available:
            if i >= half and i < n - half:
                window_data = arr[i - half : i + half + 1]
            # For indices near the beginning:
            elif i < half:
                window_data = arr[0 : 2 * i + 1]
            # For indices near the end:
            else:  # i >= n - half
                window_data = arr[n - (2 * (n - i) - 1) : n]

            # Only compute the average if we have at least `min_points` data points.
            if len(window_data) < min_points:
                smoothed[i] = np.nan
            else:
                smoothed[i] = np.mean(window_data)

        return smoothed

    # Create a combined figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [1, 1]}, sharey=False
    )

    # Plot Left: Daily Mean SMB per Cluster
    for cluster in daily_means.columns:
        x = daily_means.index
        y = daily_means[cluster]
        valid_mask = ~y.isna()
        x_valid, y_valid = x[valid_mask], y[valid_mask]
        y_smoothed = custom_rolling_mean(y_valid, full_window=25, min_points=5)
        ax1.plot(x_valid, y_smoothed, label=f"CL{cluster}", linewidth=2)
        ax1.scatter(x, y, marker="o", s=18, alpha=0.6)
    ax1.set_xlabel("Day of Year (June-August)")
    ax1.set_ylabel("Mean Specific SMB (mm w.e.)")
    ax1.grid(True)

    # Plot Right: Mean SMB per Cluster vs Elevation
    for cluster, data in df_cluster_grouped.groupby("Cluster"):
        ax2.plot(
            data["Mean_Elevation"],
            data["SMB"],
            label=f"CL{cluster}",
            marker="o",
            linewidth=2,
        )
    ax2.set_xlabel("Elevation (m)")
    ax2.set_ylabel("Mean Specific SMB (mm w.e.)")
    ax2.grid(True)

    # Add bold 'b' above the right plot
    ax2.text(
        -0.05,
        1.03,
        "(b)",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="center",
    )
    # Add bold 'a' above the left plot
    ax1.text(
        -0.05,
        1.03,
        "(a)",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="center",
    )

    # Create a common legend below the plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend
    os.makedirs(output_dir, exist_ok=True)
    save_path_png = os.path.join(output_dir, "combined_smb_cluster.png")
    save_path_pdf = os.path.join(output_dir, "combined_smb_cluster.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined plot saved to {save_path_pdf}")


def plot_smb_vs_ela(
    input_file, output_dir_df, output_dir_plot, smoothing=False, frac=0.45
):
    """
    Reads the MAR SMB data, computes ELA and JJA temperature anomalies, and plots SMB vs ELA color coded by temperature anomaly.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_dir_df (str): Directory where the processed dataframe will be saved.
        output_dir_plot (str): Directory where the plot will be saved (if needed).
        smoothing (bool): If True, adds a LOWESS smoothing line to the scatter plot.
        frac (float): The fraction of data used when computing LOWESS smoothing.
    """

    # Ensure output directories exist
    os.makedirs(output_dir_df, exist_ok=True)
    os.makedirs(output_dir_plot, exist_ok=True)

    # Load data
    df = pd.read_csv(input_file)

    # Convert 'Time' column to datetime
    df["Time"] = pd.to_datetime(df["Time"])

    # Extract year from 'Time'
    df["Year"] = df["Time"].dt.year.astype(int)

    # Filter for summer months (June, July, August)
    summer_df = df[df["Time"].dt.month.isin([6, 7, 8])]

    # Group by 'Year' and calculate the mean temperature for summer months
    yearly_summer_temp = summer_df.groupby("Year")["Temp"].mean()

    # Calculate the baseline climatology for the period 1991-2020
    baseline = yearly_summer_temp.loc[1991:2020].mean()

    # Compute the temperature anomaly for each year relative to the baseline
    yearly_temp_anomaly = yearly_summer_temp - baseline

    # Create a new column that represents the absolute SMB contribution (in mm·km²)
    df["SMB_abs_mm_km2"] = df["SMB"] * df["Total_Area_km2"]

    # Group by Year and elevation bin to aggregate the data
    df_grouped = (
        df.groupby(["Year", "elev_bin"])
        .agg(
            SMB_abs=("SMB_abs_mm_km2", "sum"),  # Sum of SMB contributions for the bin
            Total_Area_km2=("Total_Area_km2", "sum"),  # Sum of areas for the bin
            Mean_Elevation=("Mean_Elevation", "mean"),  # Mean elevation of the bin
        )
        .reset_index()
    )

    # Convert the absolute SMB from mm·km² to Gigatonnes (Gt)
    df_grouped["SMB_abs_Gt"] = df_grouped["SMB_abs"] * 1e-6

    def find_smb_zero_crossing(df):
        """
        Find the elevation where the SMB (in Gt) crosses zero for each year.
        If no crossing is found, defaults to 925 m.
        """
        zero_crossings = []
        for year, group in df.groupby("Year"):
            group = group.sort_values("Mean_Elevation")
            # Use the absolute SMB values in Gt for interpolation
            smb_values = group[["Mean_Elevation", "SMB_abs_Gt"]].values
            crossings = []
            for i in range(len(smb_values) - 1):
                if smb_values[i, 1] * smb_values[i + 1, 1] < 0:
                    x1, y1 = smb_values[i]
                    x2, y2 = smb_values[i + 1]
                    elev_crossing = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                    crossings.append(elev_crossing)
            # Calculate the overall absolute SMB for the year (in Gt)
            abs_SMB = group["SMB_abs_Gt"].sum()
            if crossings:
                zero_crossings.append(
                    {"Year": year, "ELA": min(crossings), "SMB": abs_SMB}
                )
            else:
                zero_crossings.append({"Year": year, "ELA": 925, "SMB": abs_SMB})
        return pd.DataFrame(zero_crossings)

    zero_crossing_df = find_smb_zero_crossing(df_grouped)
    zero_crossing_df["t_anom"] = zero_crossing_df["Year"].map(yearly_temp_anomaly)

    # Calculate Accumulation Area Ratio (AAR)
    aar_all = []
    for i, row in zero_crossing_df.iterrows():
        ela = row["ELA"]
        yearly = df_grouped[df_grouped["Year"] == row["Year"]]
        accumulation_area = yearly[yearly["Mean_Elevation"] > ela][
            "Total_Area_km2"
        ].sum()
        ablation_area = yearly[yearly["Mean_Elevation"] < ela]["Total_Area_km2"].sum()
        aar = (
            accumulation_area / (accumulation_area + ablation_area)
            if (accumulation_area + ablation_area)
            else 0
        )
        aar_all.append(aar)
    zero_crossing_df["AAR"] = aar_all

    # Save the processed dataframe
    df_path = os.path.join(output_dir_df, "ela_smb_df.csv")
    zero_crossing_df.to_csv(df_path, index=False)
    print(f"Dataframe saved to {df_path}")

    # path to zero_crossing_df: data\mar_smb_1991_2024_FI\ela_smb_df.csv
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the symmetric color limits so that 0°C is centered
    vlim = np.max(np.abs(zero_crossing_df["t_anom"]))
    vmin, vmax = -vlim, vlim

    # Create the scatter plot (SMB vs ELA)
    sc = ax.scatter(
        zero_crossing_df["SMB"],
        zero_crossing_df["ELA"],
        c=zero_crossing_df["t_anom"],
        cmap="coolwarm",  # blue for cold, red for warm
        vmin=vmin,
        vmax=vmax,
        edgecolor="k",  # adds a black border around markers
        s=100,  # adjust marker size as needed
    )

    # Label the axes and add grid
    ax.set_xlabel("Annual SMB (Gt)")
    ax.set_ylabel("ELA (m)")
    ax.grid(True)

    # Add a colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("JJA Temperature Anomaly (°C)")

    # Add LOWESS smoothing if desired
    if smoothing:
        x = zero_crossing_df["SMB"].values
        y = zero_crossing_df["ELA"].values
        smoothed = sm.nonparametric.lowess(y, x, frac=frac)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color="green",
            linestyle="-",
            linewidth=2,
            label="LOWESS Smoothing",
        )
        ax.legend()

    # Save the plot
    plot_path = os.path.join(output_dir_plot, "smb_vs_ela.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


def plot_smb_time_and_hyps(file_path_smb, file_path_elev, file_path_zero, output_dir):
    """
    Combines four separate plots into one figure arranged in two columns and two rows.

    Left column (nested vertically):
      - Top: Winter/Summer SMB plot (stacked seasonal bars and annual bars)
      - Bottom: Cumulative SMB plot
    Right column:
      - Top: Hypsometric plot (SMB vs. Elevation)
      - Bottom: Zero-crossing plot (scatter plot with ELA on x-axis and SMB on y-axis)

    Parameters:
      file_path_smb (str): CSV file for seasonal SMB data (for winter/summer and cumulative plots).
      file_path_elev (str): CSV file for hypsometric data (SMB vs. Elevation).
      file_path_zero (str): CSV file for zero-crossing data (e.g., output of ela_smb_df.csv).
      output_dir (str): Directory to save the combined plot.
      smoothing (bool): Whether to add LOWESS smoothing to the zero-crossing scatter.
      frac (float): The LOWESS smoothing fraction.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    ##########################
    # LEFT COLUMN: First Function Plots (Seasonal SMB and Cumulative SMB)
    ##########################
    # Load seasonal SMB data
    smb_df = pd.read_csv(file_path_smb)
    smb_df["Time"] = pd.to_datetime(smb_df["Time"])
    # Filter data between 01.09.1991 and 31.08.2024
    start_date = "1991-09-01"
    end_date = "2024-08-31"
    smb_df = smb_df[(smb_df["Time"] >= start_date) & (smb_df["Time"] <= end_date)]

    # Add Year and Month columns
    smb_df["Year"] = smb_df["Time"].dt.year
    smb_df["Month"] = smb_df["Time"].dt.month
    # Define hydrological year: if Month >= 9, then HydroYear = Year + 1
    smb_df["HydroYear"] = smb_df["Year"]
    smb_df.loc[smb_df["Month"] >= 9, "HydroYear"] += 1
    # Define Season column: Summer for June-Aug, else Winter.
    smb_df["Season"] = smb_df["Month"].apply(
        lambda x: "Summer" if x in [6, 7, 8] else "Winter"
    )

    # Group by Hydrological Year and Season, summing Specific Mass Balance (in mm WE)
    seasonal_sums = (
        smb_df.groupby(["HydroYear", "Season"])["Specific Mass Balance (mm WE)"]
        .sum()
        .unstack(fill_value=0)
    )
    seasonal_sums["Total"] = seasonal_sums.sum(axis=1)

    # Compute SMB (Gt) and cumulative SMB (Gt)
    smb_df["SMB (Gt)"] = smb_df["SMB (m^3)"] / 1e9
    yearly_smb_gt = smb_df.groupby("HydroYear")["SMB (Gt)"].sum()
    cumulative_smb_gt = yearly_smb_gt.cumsum()

    # Prepare data for plotting (convert mm to m for seasonal sums)
    hydro_years = seasonal_sums.index
    summer = seasonal_sums["Summer"] / 1000  # m we
    winter = seasonal_sums["Winter"] / 1000  # m we
    total = seasonal_sums["Total"] / 1000  # m we

    ##########################
    # RIGHT TOP: Second Function Plot (Hypsometric plot)
    ##########################
    # Load hypsometric data
    df_elev = pd.read_csv(file_path_elev)
    df_elev["Time"] = pd.to_datetime(df_elev["Time"])
    # Create Year_Bin using the defined bins:
    bins_elev = [1990, 2002, 2013, 2024]
    labels_elev = [1991, 2003, 2014]
    df_elev["Year_Bin"] = pd.cut(
        df_elev["Time"].dt.year, bins=bins_elev, labels=labels_elev, right=True
    )
    df_elev["Year_Bin"] = df_elev["Year_Bin"].astype(int)

    # Group by Year_Bin and elev_bin, and aggregate
    df_elev_grouped = (
        df_elev.groupby(["Year_Bin", "elev_bin"])
        .agg(
            {
                "SMB": "sum",
                "Total_Area_km2": "mean",
                "Mean_Elevation": "mean",
            }
        )
        .reset_index()
    )
    # Normalize SMB per year (using a provided dictionary of years in each bin)
    years_in_bin = {1991: 12, 2003: 11, 2014: 11}
    df_elev_grouped["SMB_per_year"] = df_elev_grouped.apply(
        lambda row: row["SMB"] / years_in_bin[row["Year_Bin"]], axis=1
    )
    # Define manual elevation bins and assign bar midpoints/widths
    elevation_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    midpoints = [
        (elevation_bins[i] + elevation_bins[i + 1]) / 2
        for i in range(len(elevation_bins) - 1)
    ]
    widths = np.diff(elevation_bins)
    elev_bin_to_midpoint = {i: midpoints[i] for i in range(len(midpoints))}
    df_elev_grouped["Bar_Center"] = df_elev_grouped["elev_bin"].map(
        elev_bin_to_midpoint
    )
    df_elev_grouped["Bar_Width"] = df_elev_grouped["elev_bin"].map(
        lambda x: widths[x] - 2 if x < len(widths) else widths[-1]
    )

    ##########################
    # RIGHT BOTTOM: Third Function Plot (ELA SMB scatter)
    ##########################
    # Load ELA SMB data
    df_zero = pd.read_csv(file_path_zero)
    # Determine color limits based on t_anom.
    vlim = np.max(np.abs(df_zero["t_anom"]))
    vmin, vmax = -vlim, vlim

    ##########################
    # Now: Combine all plots into one canvas using GridSpec
    ##########################
    # Create the main figure
    fig = plt.figure(figsize=(11, 6.5))
    # Create a GridSpec with 1 row and 2 columns.
    gs_main = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.24)

    # Left column: nested GridSpec (2 rows, 1 column) with height ratios [2,1]
    gs_left = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1, subplot_spec=gs_main[0], height_ratios=[1.2, 1], hspace=0.2
    )
    # Right column: nested GridSpec (2 rows, 1 column) for top and bottom plots
    gs_right = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1, subplot_spec=gs_main[1], hspace=0.45
    )

    # LEFT TOP: First function, seasonal SMB stacked bars
    ax_left_top = fig.add_subplot(gs_left[0])
    x = np.arange(len(hydro_years))
    bar_width = 0.4
    ax_left_top.bar(
        x,
        winter,
        width=bar_width,
        label="Winter",
        color="dodgerblue",
        alpha=0.4,
        zorder=3,
    )
    ax_left_top.bar(
        x, summer, width=bar_width, label="Summer", color="red", alpha=0.55, zorder=3
    )
    ax_left_top.bar(
        x + bar_width, total, width=bar_width, label="Annual", color="black", zorder=3
    )
    ax_left_top.set_ylabel("Specific SMB (m we)", fontsize=12)
    ax_left_top.set_xlim(-0.8, len(hydro_years) - 0.3)
    ax_left_top.set_ylim(-0.84, 0.6)

    # Set tick labels: show every 2nd year
    tick_labels = [
        str(year) if i % 2 == 0 else "" for i, year in enumerate(hydro_years)
    ]

    # Set x-ticks and hide labels on the top axis
    ax_left_top.set_xticks(x + bar_width / 2)
    ax_left_top.set_xticklabels(hydro_years, rotation=45)

    ax_left_top.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.13),
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.5,
        labelspacing=0.2,
    )
    ax_left_top.grid(linestyle="--", alpha=0.7, zorder=0)

    # LEFT BOTTOM: First function, cumulative SMB plot
    ax_left_bottom = fig.add_subplot(gs_left[1], sharex=ax_left_top)
    ax_left_bottom.plot(
        x + bar_width / 2,
        cumulative_smb_gt.values,
        color="green",
        marker="o",
        zorder=3,
        linewidth=2,
    )
    print("Cumulative SMB (Gt):", cumulative_smb_gt)
    ax_left_bottom.set_ylabel("Cumulative SMB (Gt)", fontsize=12)
    ax_left_bottom.set_xlabel("Hydrological Year", fontsize=12)
    ax_left_bottom.grid(linestyle="--", alpha=0.7, zorder=0)
    ax_left_bottom.set_xticks(x + bar_width / 2)

    ax_left_bottom.set_xticklabels(tick_labels, rotation=45)

    # RIGHT TOP: Second function, hypsometric plot (SMB vs Elevation)
    ax_right_top = fig.add_subplot(gs_right[0])
    # Plot area bars on left y-axis:
    ax_right_top.bar(
        df_elev_grouped["Bar_Center"],
        df_elev_grouped["Total_Area_km2"],
        width=df_elev_grouped["Bar_Width"],
        alpha=1,
        color="grey",
        label="Area",
    )
    ax_right_top.set_xlabel("Elevation (m)", fontsize=12)
    ax_right_top.set_ylabel("Area (km²)", fontsize=12)
    ax_right_top.set_ylim(0, 2000)
    ax_right_top.grid(True, axis="x", linestyle="--", alpha=0.7)
    ax_right_top.legend(loc="upper right", bbox_to_anchor=(0.3, 0.95))

    # Add minor ticks to the primary x-axis
    ax_right_top.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # Enable vertical grid lines for both major and minor ticks on the x-axis
    ax_right_top.grid(which="major", axis="x", linestyle="--", alpha=0.7)
    ax_right_top.grid(which="minor", axis="x", linestyle=":", alpha=0.5)

    # Create secondary y-axis for SMB
    ax_right_top_2 = ax_right_top.twinx()
    palette = sns.color_palette("cool", as_cmap=True)
    norm = plt.Normalize(
        df_elev_grouped["Year_Bin"].min(), df_elev_grouped["Year_Bin"].max()
    )
    for year_bin, group in df_elev_grouped.groupby("Year_Bin"):
        if year_bin == 1991:
            label = "1991-2002"
        else:
            label = f"{year_bin}-{year_bin+10}"
        ax_right_top_2.plot(
            group["Mean_Elevation"],
            group["SMB_per_year"] / 1000,
            marker="o",
            linestyle="-",
            color=palette(norm(year_bin)),
            label=label,
            linewidth=2,
        )
    ax_right_top_2.set_ylabel("Annual Specific SMB (m w.e.)", fontsize=12)
    ax_right_top_2.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax_right_top_2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=10,
        handletextpad=0.2,
        columnspacing=0.5,
        labelspacing=0.2,
        frameon=False,
    )
    ax_right_top_2.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Also add minor ticks on the secondary x-axis (they share the same x-axis)
    ax_right_top_2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # RIGHT BOTTOM: Third function, ELA SMB scatter plot
    ax_right_bottom = fig.add_subplot(gs_right[1])
    pos = ax_right_bottom.get_position()

    correlation = df_zero["SMB"].corr(df_zero["t_anom"])
    print("Pearson Correlation between SMB and t_anom:", correlation)
    sc = ax_right_bottom.scatter(
        df_zero["ELA"],
        df_zero["SMB"],
        c=df_zero["t_anom"],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolor="k",
        s=100,
    )
    # Annotate the correlation in the top-left corner of the axis
    ax_right_bottom.annotate(
        f"r(SMB, JJA T_anom) = {correlation:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    ax_right_bottom.set_xlabel("ELA (m)", fontsize=12)
    ax_right_bottom.set_ylabel("Annual SMB (Gt)", fontsize=12)
    # Enable grid (both horizontal and vertical)
    ax_right_bottom.grid(True, linestyle="--", alpha=0.7)

    # Get the position of the top right axis
    pos_top = ax_right_top.get_position()  # [x0, y0, width, height]
    # Get the current position of the right-bottom axis
    pos_bottom = ax_right_bottom.get_position()  # [x0, y0, width, height]

    # Define a fixed space for the colorbar (in figure coordinate width)
    cb_space = 0.025  # adjust as needed

    # Define a vertical offset to lower the right-bottom plot (in figure coordinates)
    vertical_offset = 0.03  # adjust as needed

    # Set the new position for the right-bottom axis so its right border aligns with the top axis's right border minus cb_space
    new_y = pos_bottom.y0 - vertical_offset  # shift downward
    new_pos = [pos_top.x0, new_y, pos_top.width - cb_space, pos_bottom.height]
    ax_right_bottom.set_position(new_pos)

    # Create a new axes for the colorbar, placing it to the right of ax_right_bottom
    # A small gap (0.01) is left between the plot and the colorbar.
    cbar_ax = fig.add_axes(
        [new_pos[0] + new_pos[2] + 0.01, new_pos[1], cb_space - 0.01, new_pos[3]]
    )
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label("JJA Temp. Anomaly (°C)", fontsize=12)

    # Increase x-axis tick density by adding minor ticks:
    ax_right_bottom.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # Enable vertical grid lines for both major and minor ticks on the x-axis:
    ax_right_bottom.grid(which="major", axis="x", linestyle="--", alpha=0.7)
    ax_right_bottom.grid(which="minor", axis="x", linestyle=":", alpha=0.5)

    # Now, create the colorbar in the new axes
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label("JJA Temp. Anomaly (°C)", fontsize=12)
    # Now add labels (a, b, c, d) to each subplot in the top left corner:
    ax_left_top.text(
        -0.1,
        1.03,
        "(a)",
        transform=ax_left_top.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_left_bottom.text(
        -0.1,
        1.03,
        "(c)",
        transform=ax_left_bottom.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_right_top.text(
        -0.0,
        1.03,
        "(b)",
        transform=ax_right_top.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_right_bottom.text(
        -0.0,
        1.03,
        "(d)",
        transform=ax_right_bottom.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    plt.tight_layout()

    # Save the combined plot
    combined_plot_path_png = os.path.join(output_dir, "FI_mass_balance_combined.png")
    combined_plot_path_pdf = os.path.join(output_dir, "FI_mass_balance_combined.pdf")
    plt.savefig(combined_plot_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(combined_plot_path_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {combined_plot_path_png}")
