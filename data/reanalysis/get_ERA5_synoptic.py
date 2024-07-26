import cdsapi

c = cdsapi.Client()
output_directory = "G:\\LOWTROP_VRS\\data\\reanalysis\\"
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'geopotential', 'temperature',
        ],
        'pressure_level': [
            '500', '850', '1000',
        ],
        'year': '2023',
        'month': [
            '07', '08',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '27', '28', '29',
            '30', '31',
        ],
        'time': [
            '06:00', '12:00', '18:00',
        ],
        'area': [
            85, -40, 77,
            17,
        ],
    },
        output_directory + "ERA5_synoptic.nc")