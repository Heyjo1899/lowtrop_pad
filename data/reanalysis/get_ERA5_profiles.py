import cdsapi

# Set the output directory where you want to save the file
output_directory = 'G:\\LOWTROP_VRS\\data\\reanalysis\\'


c = cdsapi.Client()
c.retrieve("reanalysis-era5-complete", {
    "class": "ea",
    "date": "2023-07-27/to/2023-08-21",
    "expver": "1",
    "levelist": "122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137",
    "levtype": "ml",
    "number": "0",
    "param": "129/130/133",
    "step": "0",
    "stream": "enda",
    "time": "09:00:00/21:00:00",
    'area'    : '82/-17/80/-15',
    'grid'    : '1.0/1.0', 
    'format'  : 'netcdf',
    "type": "4v"               
},
output_directory + 'ERA5_profiles.nc'
)    