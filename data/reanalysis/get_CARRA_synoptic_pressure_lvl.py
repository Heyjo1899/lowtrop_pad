import cdsapi

output_directory = 'G:\\LOWTROP_VRS\\data\\reanalysis\\'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-carra-pressure-levels',
    {
        'format': 'netcdf',
        'domain': 'west_domain',
        'variable': [
            'geopotential', 'temperature', 'u_component_of_wind',
            'v_component_of_wind',
        ],
        'pressure_level': [
            '500', '850',
        ],
        'product_type': 'analysis',
        'time': '12:00',
        'year': [
            '1990', '1991', '1992',
            '1993', '1994', '1995',
            '1996', '1997', '1998',
            '1999', '2000', '2001',
            '2002', '2003', '2004',
            '2005', '2006', '2007',
            '2008', '2009', '2010',
            '2011', '2012', '2013',
            '2014', '2015', '2016',
            '2017', '2018', '2019',
            '2020', '2021', '2022',
            '2023',
        ],
        'month': [
            '05', '06', '07',
            '08', '09', '10',
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
    output_directory + 'CARRA_synoptic_pressure_lvl.nc'
    )