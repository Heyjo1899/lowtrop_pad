import cdsapi

output_directory = 'G:\\LOWTROP_VRS\\data\\reanalysis\\'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-carra-single-levels',
    {
        'format': 'netcdf',
        'domain': 'west_domain',
        'level_type': 'surface_or_atmosphere',
        'variable': [
            '10m_wind_direction', '10m_wind_speed', '2m_temperature',
            'albedo', 'fraction_of_snow_cover', 'high_cloud_cover',
            'land_sea_mask', 'low_cloud_cover', 'mean_sea_level_pressure',
            'sea_ice_area_fraction', 'total_cloud_cover', 'total_column_integrated_water_vapour',
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
    output_directory + 'CARRA_synoptic_single_lvl.nc'
    )