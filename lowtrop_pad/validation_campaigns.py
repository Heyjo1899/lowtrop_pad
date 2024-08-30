import os
import pandas as pd
import datetime

def read_xq2_data(xq2_file):
    df = pd.read_csv(xq2_file, parse_dates=True)
    
    # Extract only the XQ data (the rest is zero)
    df = df[['XQ-iMet-XQ Pressure',
             'XQ-iMet-XQ Air Temperature',
             'XQ-iMet-XQ Humidity',
             'XQ-iMet-XQ Humidity Temp',
             'XQ-iMet-XQ Date',
             'XQ-iMet-XQ Time',
             'XQ-iMet-XQ Longitude',
             'XQ-iMet-XQ Latitude',
             'XQ-iMet-XQ Altitude',
             'XQ-iMet-XQ Sat Count']]
    
    # Rename the columns (strip 'XQ-iMet-XQ ')
    df = df.rename(columns={k: k[11:] for k in df.keys()})
    
    # Combine Date and Time into a single Datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + '-' + df['Time'])
    
    # Drop the original Date and Time columns
    df = df.drop(columns=['Date', 'Time'])
    
    # Rename columns
    df.rename(columns={
        'Longitude': 'lon',
        'Latitude': 'lat',
        'Altitude': 'alt',
        'Air Temperature': 't',
        'Sat Count': 'Sat',
        'Datetime': 'time',
        'Humidity': 'h'
    }, inplace=True)
    
    return df

def load_hl_data(directory):
    """
    Load all humilog files in the subdirectories of the given directory and return them in a dictionary.
    """
    data = {}
    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Humilog files:
            if file.endswith(".ASC"):
                # Get the folder name containing the file
                file_name = os.path.basename(file)
                
                # Extract the date part from the path
                date_part = root.split('\\')[-1]
                
                # Extract the desired substring '0817'
                hl_date = date_part[4:8]
                # Extract the last 5 numbers from the folder name
                hl_number = file_name[9:14]
                
                # Generate the individual name
                individual_name = f"hl_{hl_date}_{hl_number}"
                
                # Load the file
                file_path = os.path.join(root, file)
                #print(file_path)
                # Load the file
                hl_df = pd.read_table(file_path, encoding='latin-1', skiprows=4)
                    
                # Strip the white spaces in the headers
                hl_df.columns = hl_df.columns.str.strip()
                # Rename to simpler headers
                hl_df.rename(columns={'Datum': 'date', 'Zeit': 'time_raw', '1.Temperatur    [°C]': 't', '2.rel.Feuchte    [%]': 'h'}, inplace=True)
        
                hl_df = hl_df.iloc[:, :-1]

                hl_df['h'] = hl_df['h'].str.replace(',', '.').astype(float)
                
                # Convert 'time_raw' to datetime and reformat to 'YY-MM-DD HH:MM:SS'
                hl_df['time_raw'] = pd.to_datetime(hl_df['date'] + ' ' + hl_df['time_raw'], format='%d.%m.%y %H:%M:%S', dayfirst=True)

                # Drop the original 'date' column
                hl_df.drop(['date'], axis=1, inplace=True)
                
                # Adjust time by subtracting 2 hours
                hl_df['time'] = pd.to_datetime(hl_df['time_raw']) - datetime.timedelta(hours=2)
                
                # Store the processed DataFrame in the dictionary
                data[individual_name] = hl_df
    return data

def load_save_mast_xq2(directory, output_folder, thresh=2, start_buffer=4, end_buffer=2):   
    """
    Loops over .csv files in directory, separates them into individual ascents, and saves each ascent as a CSV file.
    
    Parameters:
        directory (str): Directory containing the .csv files.
        output_folder (str): Folder where the separated ascent CSV files will be saved.
        thresh (float): Altitude difference threshold for detecting ascents.
        start_buffer (int): Number of points before ascent detection to include.
        end_buffer (int): Number of points after ascent detection to include.
    """

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                # Extract the relevant digits from the file name for individual identification
                print('file: ', file)
                digits = file[15:21]
                individual_name = f"{digits}"
                
                # Load the CSV file
                file_path = os.path.join(root, file)
                print('file_path: ', file_path)
                df = pd.read_csv(file_path)
                
                # Rename columns for consistency
                df.rename(columns={
                    'T': 't',
                    'Sat Count': 'Sat',
                    'Datetime': 'time',
                    'Humidity': 'h'
                }, inplace=True)

                # Ensure no duplicate timestamps
                df = df.groupby('time').head(1).reset_index(drop=True)
                
                # Add potential temperature
                df['T_pot'] = (df['t'] + 273.15) * ((1000 / df['Pressure']) ** 0.286)
                
                # Detect ascents
                idx = []
                for i in range(1, len(df['alt'])):
                    if df['alt'][i] - df['alt'][i-1] >= thresh:
                        idx.append(i)
                
                # Identify boundaries of each ascent
                lower = [0]
                upper = []
                for i in range(1, len(idx)-1):
                    if idx[i+1] - idx[i] == 1 and idx[i] - idx[i-1] != 1:
                        lower.append(idx[i] - start_buffer)
                    if idx[i] - idx[i-1] == 1 and idx[i+1] - idx[i] != 1:
                        upper.append(idx[i] + end_buffer)
                upper.append(idx[-1])
                
                # Extract and store individual ascents
                for i in range(len(lower)):
                    df_single = df.iloc[lower[i]:upper[i]].reset_index(drop=True)
                    df_single['alt_ag'] = df_single['alt'] - df_single['alt'][0]
                    
                    # Construct the filename for this ascent
                    ascent_filename = f"{individual_name}_ascent_{i+1}.csv"
                    ascent_path = os.path.join(output_folder, ascent_filename)
                    
                    # Save the individual ascent to a CSV file
                    df_single.to_csv(ascent_path, index=False)
                    print(f"Saved {ascent_filename}")

