import pandas as pd
from pathlib import Path
import re

def process_csv_files(base_input_dir, base_output_dir):
    """
    Processes CSV files in a given directory structure, selects/renames columns,
    and saves them to a new directory with a modified naming convention.

    Args:
        base_input_dir (str): The path to the root input directory containing
                               the date-named subfolders.
        base_output_dir (str): The path to the root output directory where
                                the processed files will be saved.
    """
    input_path = Path(base_input_dir)
    output_path = Path(base_output_dir)

    # Ensure output base directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the columns to extract and their new names
    columns_to_extract = {
        'alt_ag': 'ALT_ag',
        'xq2_T': 'AT',
        'xq2_time': 'TIME', # This column will be formatted
        'xq2_lon': 'LON',
        'xq2_lat': 'LAT',
        'xq2_alt': 'ALT'
    }

    # Iterate through each date-named subfolder
    for date_folder in input_path.iterdir():
        # Check if it's a directory named like YYYYMMDD
        if date_folder.is_dir() and re.fullmatch(r'\d{8}', date_folder.name):
            print(f"Processing folder: {date_folder.name}")

            # Create the corresponding output date folder
            output_date_folder = output_path / date_folder.name
            output_date_folder.mkdir(exist_ok=True)

            # Iterate through each CSV file in the current date folder
            for csv_file in date_folder.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)

                    missing_columns = [col for col in columns_to_extract.keys() if col not in df.columns]
                    if missing_columns:
                        print(f"  Warning: Skipping {csv_file.name} in {date_folder.name} due to missing columns: {', '.join(missing_columns)}")
                        continue

                    # Select columns
                    df_processed = df[list(columns_to_extract.keys())].copy() # Use .copy() to avoid SettingWithCopyWarning

                    
                    # Convert to datetime, then format to 'YYYY-MM-DD HH:MM'
                    if 'xq2_time' in df_processed.columns:
                        try:
                            # Attempt to convert to datetime. coerce errors will set invalid parsing to NaT
                            df_processed['xq2_time'] = pd.to_datetime(df_processed['xq2_time'], errors='coerce')
                            # Drop rows where conversion failed (NaT) for 'xq2_time'
                            df_processed = df_processed.dropna(subset=['xq2_time'])
                            df_processed['xq2_time'] = df_processed['xq2_time'].dt.strftime('%Y-%m-%d %H:%M')
                        except Exception as time_e:
                            print(f"  Warning: Could not format 'xq2_time' in {csv_file.name}: {time_e}. Keeping original format for this column.")
                            pass # Keep original format if conversion fails

                    # Rename columns
                    df_processed.rename(columns=columns_to_extract, inplace=True)

                    # --- Round numerical columns ---
                    if 'LON' in df_processed.columns:
                        df_processed['LON'] = df_processed['LON'].round(4)
                    if 'LAT' in df_processed.columns:
                        df_processed['LAT'] = df_processed['LAT'].round(4)
                    if 'AT' in df_processed.columns:
                        df_processed['AT'] = df_processed['AT'].round(2)
                    if 'ALT' in df_processed.columns:
                        df_processed['ALT'] = df_processed['ALT'].round(1)


                    # --- 3. Adapt file naming ---
                    original_filename = csv_file.name # e.g., merged_20230802-4-1-1-validation_control_tundra.csv

                    # Regex to extract components for new filename:
                    # Group 1: date (YYYYMMDD)
                    # Group 2: first number after date (e.g., '4' from '4-1-1')
                    # Group 3: third number after date (e.g., '1' from '4-2-1')
                    # Group 4: The surface type string (e.g., 'validation_control_tundra', 'tundra', 'ice')
                    match = re.search(r'merged_(\d{8})-(\d+)-\d+-(\d+)-(.+?)\.csv', original_filename)

                    if match:
                        file_date = match.group(1)
                        first_number_after_date = match.group(2)
                        third_number_after_date = match.group(3)
                        full_surface_string = match.group(4) # e.g., "validation_control_tundra"

                        # Extract the last word from the full_surface_string
                        surface_type = full_surface_string.split('_')[-1]

                        # Rename 'water' to 'sea' for file naming
                        if surface_type == 'water':
                            surface_type = 'sea'

                        # Construct the new filename
                        new_filename = f"{file_date}-{first_number_after_date}-{third_number_after_date}_{surface_type}.csv"
                    else:
                        # Fallback if filename pattern doesn't match perfectly
                        print(f"  Warning: Could not parse filename pattern for {original_filename}. Using default naming.")
                        new_filename = f"processed_{csv_file.stem}.csv" # Use original stem with "processed_" prefix

                    # Construct the full output path for the new CSV
                    output_file_path = output_date_folder / new_filename

                    # Save the processed DataFrame to CSV
                    df_processed.to_csv(output_file_path, index=False)
                    print(f"  Processed {csv_file.name} -> {new_filename}")

                except Exception as e:
                    print(f"  Error processing {csv_file.name}: {e}")

# --- Configuration ---
INPUT_DIRECTORY = r'C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\Analysis\lowtrop_pad\data\merged_interpol_profiles'

OUTPUT_DIRECTORY = r'C:\Users\jonat\OneDrive - Universität Graz\MASTERARBEIT\Analysis\lowtrop_pad\data\profiles_for_public'
# Run function to process the CSV files
process_csv_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
