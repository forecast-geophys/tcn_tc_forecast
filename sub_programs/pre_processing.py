from datetime import datetime
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import shutil
import xarray as xr
import pandas as pd

"""
def calc_distance(lat_point, lon_point, lat_column, lon_column):
    lat_diff = lat_column - lat_point
    long_diff_abs = np.abs(lon_column - lon_point)
    long_diff = np.min((360 - long_diff_abs, long_diff_abs), 0)
    adjusted_long = long_diff * np.cos(np.deg2rad(lat_point))
    distance = np.sqrt(lat_diff ** 2 + adjusted_long ** 2) * 111.11
    return distance


def calc_mask_array(lats_vector, lons_vector, point_lat, point_lon, r=300):
    lat_indexes_mask = (lats_vector > point_lat - r / 111.11) & (lats_vector < point_lat + r / 111.11)
    delta_lon = r / 111.11 / np.cos(np.deg2rad(point_lat))
    if (point_lon + delta_lon) <= np.max(lons_vector):
        lon_indexes_mask = (lons_vector > point_lon - delta_lon) & (lons_vector < point_lon + delta_lon)
    else:
        lon_indexes_mask = (lons_vector > point_lon - delta_lon) | (lons_vector < point_lon + delta_lon - 360)
    np_array_points = np.zeros((np.sum(lat_indexes_mask) * sum(lon_indexes_mask), 4))
    index = 0
    for grid_lat in lats_vector[lat_indexes_mask]:
        for grid_lon in lons_vector[lon_indexes_mask]:
            np_array_points[index, 0] = grid_lat
            np_array_points[index, 1] = grid_lon
            np_array_points[index, 2] = (np.abs(lats_vector - grid_lat)).argmin()
            np_array_points[index, 3] = (np.abs(lons_vector - grid_lon)).argmin()
            index += 1
    dist_km = calc_distance(point_lat, point_lon, np_array_points[:, 0], np_array_points[:, 1])
    points_array_in_r = np_array_points[dist_km < r, :]
    mask_array = np.full((len(lats_vector), len(lons_vector)), np.nan)
    mask_array[(points_array_in_r[:, 2]).astype(int), (points_array_in_r[:, 3]).astype(int)] = 1
    return mask_array
"""
cyclones_csv_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/pre_processing/ibtracs.since1980.list.v04r01.csv'
dataset_colums = ['NAME', 'BASIN', 'USA_LAT', 'USA_LON', 'ISO_TIME', 'USA_WIND', 'USA_PRES']  # , 'DIST2LAND'
start_date = '2014-02-20'
tc_data_frame = pd.read_csv(cyclones_csv_fname, skiprows=[1], na_filter=False)
tc_data_frame = tc_data_frame[tc_data_frame['USA_LAT'] != " "]
tc_data_frame = tc_data_frame[tc_data_frame['ISO_TIME'] >= start_date]
tc_compact_data_frame = pd.DataFrame(columns=['name', 'basin', 'timestamp', 'lat', 'lon', 'wind', 'slp'])
tc_compact_data_frame['timestamp'] = pd.to_datetime(tc_data_frame['ISO_TIME']).astype(int) / 10 ** 9
tc_compact_data_frame['name'] = tc_data_frame['NAME']
tc_compact_data_frame['basin'] = tc_data_frame['BASIN']
tc_compact_data_frame['lat'] = tc_data_frame['USA_LAT'].replace({' ': np.nan}).astype('float32')
tc_compact_data_frame['lon'] = tc_data_frame['USA_LON'].replace({' ': np.nan}).astype('float32')
tc_compact_data_frame['wind'] = tc_data_frame['USA_WIND'].replace({' ': np.nan}).astype('float32')
tc_compact_data_frame['slp'] = tc_data_frame['USA_PRES'].replace({' ': np.nan}).astype('float32')


prev_name = tc_compact_data_frame[['name']].iloc[0]
unnamed_count = 0
tc_compact_data_frame = tc_compact_data_frame.reset_index()
for tc_row_index, tc_row in tc_compact_data_frame.iloc[1:].iterrows():
    if 'UNNAMED' in tc_row['name']: tc_compact_data_frame.loc[tc_row_index, 'name'] = f"UNNAMED{unnamed_count}"
    if ('UNNAMED' in prev_name) and (not 'UNNAMED' in tc_row['name']): unnamed_count += 1
    prev_name = tc_row['name']
tc_compact_data_frame.to_csv('compact_tc_dataset_2014_2024.csv', index=False)
"""
geos_fp_const_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4'
geos_fp_const_dset = Dataset(geos_fp_const_fname, 'r')
geos_fp_const_dset.set_auto_mask(False)
lats_map = geos_fp_const_dset.variables['lat'][:]
lons_map = geos_fp_const_dset.variables['lon'][:]
ocean_map_data = geos_fp_const_dset.variables['FROCEAN'][0, :, :]

radius_km = 50
land_ratio_in_radius = np.zeros(ocean_map_data.shape)
for lat_indx, point_lat in enumerate(lats_map):
    for lon_indx, point_lon in enumerate(lons_map):
        if np.abs(point_lat) > 84: land_ratio_in_radius[lat_indx, lon_indx] = ocean_map_data[lat_indx, lon_indx]; continue
        point_mask = calc_mask_array(lats_map, lons_map, point_lat, point_lon, radius_km)
        land_ratio_in_radius[lat_indx, lon_indx] = 1 - np.nansum(point_mask * ocean_map_data) / np.nansum(point_mask)
    print(point_lat)
df = xr.DataArray(land_ratio_in_radius, coords=[('lat', lats_map), ('lon', lons_map)])
df.to_netcdf('land_ratio_in_50km.nc')
shutil.move('land_ratio_in_50km.nc', "/content/drive/MyDrive/")
print('Done')


land_ratio_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/land_ratio_in_50km.nc4'
land_ratio_dset = Dataset(land_ratio_fname, 'r')
plt.imshow(land_ratio_dset['__xarray_dataarray_variable__'][:])
plt.gray()
plt.show()

dst_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/pre_processing/DST_2013-2024.txt'
ap_f107_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/pre_processing/Kp, ap, Ap, SN, F10.7_2013-2024.txt'
dst_df = pd.read_csv(dst_fname, sep='\t', header=None)
dst_df['mean_dst'] = dst_df.loc[:, 3:26].mean(axis=1)
dst_df = dst_df.rename(columns={0: "year", 1: "month", 2: "day"})
dst_df['datetime'] = pd.to_datetime(dst_df[['year', 'month', 'day']]).astype(int) / 10 ** 9
ap_f107_df = pd.read_csv(ap_f107_fname, sep='\s+', header=None)
ap_f107_df['mean_Ap'] = ap_f107_df.loc[:, 15:22].mean(axis=1)
ap_f107_df = ap_f107_df.rename(columns={0: "year", 1: "month", 2: "day", 25: 'f10.7'})
ap_f107_df['datetime'] = pd.to_datb4.csv', index=False)"""
# TODO restore calculation no, tci
