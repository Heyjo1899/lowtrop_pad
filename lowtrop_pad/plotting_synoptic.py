import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_era5_synoptic(
    date,
    time,
    path_to_file1,
    path_to_file2,
    temp_level,
    geopotential_level,
    output_path,
    output_filename=None,
):
    """
    Plots ERA5 data on a map for a specific date and time, visualizing temperature
    as colors and geopotential height as isolines.

    Parameters:
    date (str): Date in the format 'YYYYMMDD'.
    time (str): Time in the format 'HH'.
    path_to_file1 (str): Path to the first ERA5 NetCDF file (contains variables `z` and `t`).
    path_to_file2 (str): Path to the second ERA5 NetCDF file (contains the variable `t2m`).
    temp_level (int): Level of the temperature to plot.
    geopotential_level (int): Level of the geopotential height to plot.
    output_path (str): Path to the output folder for saving the plot.
    output_filename (str): Optional filename for the output plot.
    """

    # Convert date from YYYYMMDD to YYYY-MM-DDTHH:MM:SS
    specific_date = f"{date[:4]}-{date[4:6]}-{date[6:]}T{time}:00:00"

    # Load the datasets
    dataset1 = xr.open_dataset(path_to_file1)
    dataset2 = xr.open_dataset(path_to_file2)

    # Gravitational acceleration in m/s²
    g = 9.80665

    # Extract temperature data from the first dataset and convert from Kelvin to Celsius
    temp_data = dataset1["t"].sel(level=temp_level, time=specific_date) - 273.15

    # Extract geopotential height data from the first dataset and convert from geopotential to geopotential height
    geopotential_data = (
        dataset1["z"].sel(level=geopotential_level, time=specific_date) / g
    )

    # Extract temperature data at 1000 hPa from the second dataset and convert from Kelvin to Celsius
    t2m = dataset2["t2m"].sel(time=specific_date).squeeze() - 273.15

    # Coordinate to mark
    longitude = -16.636666666666667
    latitude = 81.59916666666666

    # Plotting
    fig, ax = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot the temperature data from the first dataset
    temp_plot = temp_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        cbar_kwargs={"label": "850 hPa Temperature (°C)"},
    )
    ax.set_aspect("auto")

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)

    # Add gridlines
    ax.gridlines(draw_labels=True)

    # Contours and Labels for geopotential height
    contour_levels = np.arange(
        np.floor(geopotential_data.min().values / 15) * 15,
        np.ceil(geopotential_data.max().values / 15) * 15 + 15,
        15,
    )
    contours = ax.contour(
        geopotential_data.longitude,
        geopotential_data.latitude,
        geopotential_data.values,
        levels=contour_levels,
        colors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%1.0f")

    # Ensure t2m has 2D shape by selecting the specific time
    t2m_2d = t2m.sel(time=specific_date) if "time" in t2m.dims else t2m

    # Contours and Labels for temperature at 1000 hPa from the second dataset
    t_contour_levels = np.arange(
        np.floor(t2m_2d.min().values / 2) * 2,
        np.ceil(t2m_2d.max().values / 2) * 2 + 2,
        2,
    )
    t_contours = ax.contour(
        t2m_2d.longitude,
        t2m_2d.latitude,
        t2m_2d.values,
        levels=t_contour_levels,
        colors="green",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(t_contours, inline=True, fontsize=8, fmt="%1.0f")

    # Add legend for both geopotential height and temperature at 1000 hPa
    geopotential_legend_label = "Geopotential Height (gpdm)"
    t2m_legend_label = "Temperature 2m (°C)"

    # Create legend handles
    geopotential_handle = plt.Line2D(
        [0], [0], color="black", lw=0.5, label=geopotential_legend_label
    )
    t2m_handle = plt.Line2D([0], [0], color="green", lw=0.5, label=t2m_legend_label)

    # Add combined legend
    ax.legend(handles=[geopotential_handle, t2m_handle], loc="lower left", fontsize=10)

    # Plot the specific coordinate with a marker and label
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
        color="red",
        transform=ccrs.PlateCarree(),
        verticalalignment="top",
        horizontalalignment="left",
    )

    # Set title
    ax.set_title(
        f"Temperature at {temp_level} hPa and Geopotential Height at {geopotential_level} hPa on {specific_date}"
    )

    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    if output_filename:
        print(f"Saving plot to {output_path}/{output_filename}")
        plt.savefig(f"{output_path}/{output_filename}", bbox_inches="tight")
    else:
        plt.show()
