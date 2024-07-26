import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

def plot_era5_synoptic(date, time, path_to_file, variable_to_plot, level, output_path, output_filename=None):
    """
    Plots ERA5 data on a map for a specific date and time.

    Parameters:
    date (str): Date in the format 'YYYYMMDD'.
    time (str): Time in the format 'HH'.
    path_to_file (str): Path to the ERA5 NetCDF file.
    variable_to_plot (str): Variable to plot.
    level (int): Level of the variable to plot.
    output_path (str): Path to the output folder for saving the plot.
    output_filename (str): Optional filename for the output plot.
    """
    
    # Convert date from YYYYMMDD to YYYY-MM-DDTHH:MM:SS
    specific_date = f"{date[:4]}-{date[4:6]}-{date[6:]}T{time}:00:00"

    # Load the dataset
    dataset = xr.open_dataset(path_to_file)

    # Gravitational acceleration in m/s²
    g = 9.80665
    
    # Extract the data for the specific level and date
    data = dataset[variable_to_plot].sel(level=level, time=specific_date)

    # Convert geopotential to geopotential height
    data_height = data / g

    # Coordinate to mark
    longitude = -16.636666666666667
    latitude = 81.59916666666666

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the data
    contour_data = data_height.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', cbar_kwargs={'label': 'Geopotential Height (m)'})
    ax.set_aspect('auto')

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)

    # Add gridlines
    ax.gridlines(draw_labels=True)

    # Contours and Labels
    # Calculate contours and labels
    contour_levels = np.linspace(data_height.min().values, data_height.max().values, num=10)
    contours = ax.contour(data_height.longitude, data_height.latitude, data_height.values, levels=contour_levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    # Plot the specific coordinate with a marker and label
    ax.plot(longitude, latitude, marker='o', color='black', markersize=10, transform=ccrs.PlateCarree())
    ax.text(longitude, latitude, ' VRS', fontsize=15, color='black', transform=ccrs.PlateCarree(), verticalalignment='top', horizontalalignment='left')
    
    # Set title
    ax.set_title(f'Variable {variable_to_plot} at {level} hPa on {specific_date}')

    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    if output_filename:
        print(f"Saving plot to {output_path}/{output_filename}")
        plt.savefig(f"{output_path}/{output_filename}", bbox_inches='tight')
    else:
        plt.show()

