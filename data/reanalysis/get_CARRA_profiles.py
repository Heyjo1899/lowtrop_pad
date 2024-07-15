import cdsapi

output_directory = 'G:\\LOWTROP_VRS\\data\\reanalysis\\'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-carra-height-levels',
    {
        'format': 'netcdf',
        'domain': 'west_domain',
        'variable': [
            'relative_humidity', 'temperature'
        ],
        'height_level': [
            '100_m', '150_m', '15_m',
            '200_m', '250_m', '300_m',
            '30_m', '400_m', '500_m',
            '50_m', '75_m',
        ],
        'product_type': 'analysis',
        'time': [
            '00:00', '09:00', '12:00',
            '15:00', '18:00', '21:00',
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
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    },
    output_directory + 'CARRA_profiles2.nc'
    )