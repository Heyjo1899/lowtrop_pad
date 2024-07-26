### USE DBASE ENVIRONMENT ###

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import re  

# Define your working directory and file paths
new_directory = r'C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\Analysis\lowtrop_pad'
os.chdir(new_directory)

file_ending = '.csv'
x_varname = 'T'
y_varname = 'alt_ag'
date = '20230801'  # Date is no longer used for filtering
xq_2_path = "data/xq2/averaged_profiles_custom_3_5_10_20/make_Gif"
carra_path = "data/reanalysis/CARRA_extracted_profiles/make_Gif"
asiaq_path = "data\\met_stations\\Asiaq_met_VRS.csv"
output_path = f"plots\\GIF_{date}"
images_folder = os.path.join(output_path, 'images')

def split_and_concatenate(file):
    file = file.replace(".csv", "")
    first_underscore = file.find('_')
    last_hyphen = file.rfind('-')
    part_before_underscore = file[:first_underscore]
    if part_before_underscore == "avg":
        part_before_underscore = "XQ2"
    part_after_hyphen = file[last_hyphen + 1:]
    result = f"{part_before_underscore} {part_after_hyphen}"
    return result

def get_time_from_csv(filepath, time_col_name='time'):
    df = pd.read_csv(filepath)
    return pd.to_datetime(df[time_col_name].iloc[0])

def extract_label_part(filename):
    # Extract part after '-' and before '.'
    match = re.search(r'-(.*)\.csv', filename)
    return match.group(1) if match else ''

def read_and_plot(filepath, linestyle, color, linewidth, alpha, ax, label_prefix='', time_col_name='time', include_first_timestamp=False):
    df = pd.read_csv(filepath)
    if include_first_timestamp:
        first_time = pd.to_datetime(df[time_col_name].iloc[0]).strftime('%H:%M')
        label_part = extract_label_part(os.path.basename(filepath))
        label_prefix = f"{first_time} XQ2 {label_part}"
    ax.plot(df[x_varname], df[y_varname], linestyle=linestyle, color=color,
            linewidth=linewidth, alpha=alpha, label=label_prefix)

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

df_asiaq = pd.read_csv(asiaq_path, sep=';', parse_dates=['DateTime'], dayfirst=True)
df_asiaq.set_index('DateTime', inplace=True)

def get_wind_direction_at_time(time, df_asiaq, tolerance=pd.Timedelta(minutes=5)):
    nearest_times = df_asiaq.index[(df_asiaq.index >= (time - tolerance)) & (df_asiaq.index <= (time + tolerance))]
    if not nearest_times.empty:
        nearest_time = nearest_times[np.argmin(np.abs(nearest_times - time))]
        return df_asiaq.loc[nearest_time]['VD(degrees 9m)']
    else:
        return 'N/A'

carra_files_with_times = []
for file in os.listdir(carra_path):
    if file.endswith(file_ending):
        time = get_time_from_csv(os.path.join(carra_path, file))
        carra_files_with_times.append((file, time))

xq2_files_with_times = []
for file in os.listdir(xq_2_path):
    if file.endswith(file_ending):
        time = get_time_from_csv(os.path.join(xq_2_path, file))
        xq2_files_with_times.append((file, time))

# Sort the files by the numeric part of the filenames
carra_files_with_times.sort(key=lambda x: extract_number_from_filename(x[0]))
xq2_files_with_times.sort(key=lambda x: extract_number_from_filename(x[0]))

images = []

# Create the images subfolder if it does not exist
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

for i, (carra_file, carra_time) in enumerate(carra_files_with_times):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find the corresponding xq2 file and time for current carra file
    xq2_file = f"{carra_file.split('_', 1)[1]}"
    xq2_filepath = os.path.join(xq_2_path, xq2_file)
    
    if os.path.exists(xq2_filepath):
        xq2_time = get_time_from_csv(xq2_filepath)
        wind_direction = get_wind_direction_at_time(xq2_time, df_asiaq)  # Use xq2 time for wind direction
        read_and_plot(xq2_filepath, 'solid', 'blue', 2.5, 1.0, ax, time_col_name='time', include_first_timestamp=True)

    for past_file, _ in carra_files_with_times[:i]:
        read_and_plot(os.path.join(carra_path, past_file), 'solid', 'red', 1.0, 0.5, ax)

    for past_file, _ in xq2_files_with_times[:i]:
        read_and_plot(os.path.join(xq_2_path, past_file), 'solid', 'blue', 1.0, 0.5, ax)

    current_carra_label = read_and_plot(os.path.join(carra_path, carra_file), 'solid', 'red', 2.5, 1.0, ax,
                                        label_prefix=f"{pd.to_datetime(carra_time).strftime('%H:%M')} CARRA {extract_label_part(carra_file)}")

    ax.annotate(f'Wind Dir: {wind_direction}°', xy=(0.85, 0.90), xycoords='axes fraction', ha='left', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=2))

    ax.grid()
    ax.set_ylabel("Altitude above ground [m]")
    ax.set_xlabel(f"{x_varname} [°C]")
    ax.set_title(f"Profiles on {date}")

    # Show only the current labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-2], handles[-1]], [labels[-2], labels[-1]], loc='upper right')

    plt.tight_layout()
    plt.xlim(-2, 5)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    
    # Save the individual image
    image_filename = os.path.join(images_folder, f"image_{i+1}.png")
    plt.imsave(image_filename, image)
    plt.close(fig)

# Save GIF
gif_filename = f"profiles_{date}.gif"
if output_path:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gif_filename = os.path.join(output_path, gif_filename)
imageio.mimsave(gif_filename, images, fps=0.7, loop=0)
print(f"GIF saved as {gif_filename}")


