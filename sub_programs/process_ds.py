from datetime import datetime, timedelta, timezone
import netCDF4
import numpy as np
import os.path
import pandas as pd
import pickle

import config


def calc_distance(lat_point, lon_point, lat_column, lon_column):
    lat_diff = lat_column - lat_point
    long_diff_abs = np.abs(lon_column - lon_point)
    long_diff = np.min((360 - long_diff_abs, long_diff_abs), 0)
    adjusted_long = long_diff * np.cos(np.deg2rad(lat_point))
    distance = np.sqrt(lat_diff ** 2 + adjusted_long ** 2) * 111.11
    return distance


def calc_mask_array(lats_vector, lons_vector, point_lat, point_lon, r=300):
    # TODO test ring and mask
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
    mask_array = np.full((len(lats_vector), len(lons_vector)), 0)
    mask_array[(points_array_in_r[:, 2]).astype(int), (points_array_in_r[:, 3]).astype(int)] = 1
    return mask_array


def get_components(lat1, long1, lat2, long2):
    u_component = np.cos(np.radians(lat2)) * np.sin(np.radians((long2 - long1)))
    v_component = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(np.radians(lat1))
                   * np.cos(np.radians(lat2)) * np.cos(np.radians((long2 - long1))))
    return u_component, v_component


def calc_dist_azim(lat_point, lon_point, lats_vector, lons_vector, land_map, threshold=0.8):
    land_map_bin = land_map > threshold
    lons_2d, lats_2d = np.meshgrid(lons_vector, lats_vector)
    vector_dset = np.stack((lats_2d.reshape(-1), lons_2d.reshape(-1), land_map_bin.reshape(-1)), axis=1)
    land_dset = vector_dset[vector_dset[:, 2] == 1]
    dist_vector = calc_distance(lat_point, lon_point, land_dset[:, 0], land_dset[:, 1])
    min_dist_indx = np.argmin(dist_vector)
    # dist = dist_vector[min_dist_indx]
    u_dist_2_land_km = ((lon_point - land_dset[min_dist_indx, 1]) *
                        np.cos(np.radians(lat_point+land_dset[min_dist_indx, 0])/2) * 111.11)
    v_dist_2_land_km = (lat_point - land_dset[min_dist_indx, 0]) * 111.11
    return u_dist_2_land_km, v_dist_2_land_km


def get_meteo_data(lat_point, lon_point, timestamp, config):
    proc_date_str = datetime.fromtimestamp(tc_row['timestamp'], timezone.utc).strftime('%Y%m%d_%H%M')
    meteo_path = os.path.join(config.meteo_data_dir, f"{config.meteo_file_prefix}_{proc_date_str}.nc4")
    meteo_data = np.zeros((len(config.meteo_param) - 1, config.meteo_param_n_bins))
    if not os.path.isfile(meteo_path): return np.nan, np.nan, np.nan
    meteo_dataset = netCDF4.Dataset(meteo_path)
    for var in meteo_dataset.variables: meteo_dataset.variables[var].set_auto_mask(False)
    meteo_lats = meteo_dataset.variables['lat'][:]
    meteo_lons = meteo_dataset.variables['lon'][:]
    rings_radius = np.linspace(0, config.meteo_param_radius, config.meteo_param_n_bins + 1)
    ring_mask = np.zeros((len(meteo_lats), len(meteo_lons), config.meteo_param_n_bins))
    radius_mask = calc_mask_array(meteo_lats, meteo_lons, lat_point, lon_point, config.meteo_param_radius)
    for rad_indx in range(config.meteo_param_n_bins):
        perimeter_mask = calc_mask_array(meteo_lats, meteo_lons, lat_point, lon_point, rings_radius[rad_indx + 1])
        internal_mask = calc_mask_array(meteo_lats, meteo_lons, lat_point, lon_point, rings_radius[rad_indx])
        ring_mask[:, :, rad_indx] = perimeter_mask - internal_mask
    wind_speed_map = np.sqrt(meteo_dataset.variables['V10M'][:] ** 2 + meteo_dataset.variables['U10M'][:] ** 2)
    for param_index, param in enumerate(filter(lambda el: el not in ['U10M', 'V10M'], config.meteo_param + ['W10M'])):
        param_vector = np.zeros(ring_mask.shape[2])
        if param not in ['W10M']: param_map = meteo_dataset.variables[param][:]
        else: param_map = wind_speed_map
        for rad_indx in range(ring_mask.shape[2]):
            param_vector[rad_indx] = np.mean(param_map[np.nonzero(ring_mask[:, :, rad_indx])])
        meteo_data[param_index, :] = param_vector
    mean_u_wind = np.mean(meteo_dataset.variables['U10M'][:][np.nonzero(radius_mask)])
    mean_v_wind = np.mean(meteo_dataset.variables['V10M'][:][np.nonzero(radius_mask)])
    # mean_wind_speed = np.sqrt(mean_u_wind ** 2 + mean_v_wind ** 2)
    return meteo_data, mean_u_wind, mean_v_wind


def check_meteo_files():
    proc_date = datetime(2014, 2, 20)
    while proc_date < datetime(2024, 7, 5):
        proc_date_str = proc_date.strftime('%Y%m%d_%H%M')
        meteo_path = os.path.join(config.meteo_data_dir, f"{config.meteo_file_prefix}_{proc_date_str}.nc4")
        if not os.path.isfile(meteo_path): print(proc_date_str, 'missed')
        proc_date += timedelta(hours=3)

# check_meteo_files()


land_dataset = netCDF4.Dataset(config.land_ratio_map)
for var in land_dataset.variables: land_dataset.variables[var].set_auto_mask(False)
land_map = land_dataset.variables['land_ratio'][:]
lats_land = land_dataset.variables['lat'][:]
lons_land = land_dataset.variables['lon'][:]

oceans_depth_dataset = netCDF4.Dataset(config.oceans_depth_fname)
for var in oceans_depth_dataset.variables: oceans_depth_dataset.variables[var].set_auto_mask(False)
oceans_depth_map = oceans_depth_dataset.variables['depth'][:]
lat_oceans_depth = oceans_depth_dataset.variables['lat'][:]
lon_oceans_depth = oceans_depth_dataset.variables['lon'][:]
gfz_data_frame = pd.read_csv(config.gfz_indexes_fname)
tc_data_frame = pd.read_csv(config.cyclones_csv_fname, na_filter=False)
region_tc_data_frame = tc_data_frame.loc[tc_data_frame["basin"] == config.cyclone_region]
cyclenes_list = region_tc_data_frame['name'].unique()
cyclone_dict = {}
# TODO fix inf U wind and fix inf dist to land ?
# TODO check for missed observation
for cyclone_name in cyclenes_list:
    cyclone_df = region_tc_data_frame[region_tc_data_frame['name'] == cyclone_name].reset_index()
    cyclone_data_1d = np.zeros((len(cyclone_df), 13))
    cyclone_data_2d = np.zeros((len(cyclone_df), len(config.meteo_param)-1, config.meteo_param_n_bins))
    prev_lat, prev_lon, prev_timestamp = cyclone_df[['lat', 'lon', 'timestamp']].iloc[1]
    for tc_row_index, tc_row in cyclone_df.iterrows():
        meteo_data, u_wind, v_wind = get_meteo_data(tc_row['lat'], tc_row['lon'], tc_row['timestamp'], config)
        if np.isnan(u_wind): continue
        u_dist_2_land_km, v_dist_2_land_km = calc_dist_azim(tc_row['lat'], tc_row['lon'], lats_land, lons_land, land_map)
        clsst_lat_indx = np.abs(lat_oceans_depth - tc_row['lat']).argmin()
        clsst_lon_indx = np.abs(lon_oceans_depth - tc_row['lon']).argmin()
        depth_at_point = oceans_depth_map[clsst_lat_indx, clsst_lon_indx]
        clsst_gfz_indx = np.abs(gfz_data_frame['timestamp'] - tc_row['timestamp']).argmin()
        local_gfz_datetime = gfz_data_frame['timestamp'][np.max((clsst_gfz_indx-3, 0)):clsst_gfz_indx+3]
        local_tci = gfz_data_frame['tci'][np.max((clsst_gfz_indx-3, 0)):clsst_gfz_indx+3]
        tci_dated = np.interp(tc_row['timestamp'], local_gfz_datetime, local_tci)
        u_tc_speed = calc_distance((tc_row['lat']+prev_lat) / 2, tc_row['lon'], (tc_row['lat']+prev_lat) / 2, prev_lon) *\
                                    1000 / (tc_row['timestamp'] - prev_timestamp)
        v_tc_speed = (tc_row['lat'] - prev_lat) * 111130 / (tc_row['timestamp'] - prev_timestamp)  # lat ->
        if tc_row_index == 0: u_tc_speed, v_tc_speed = u_wind, v_wind
        doy = datetime.fromtimestamp(tc_row['timestamp'], timezone.utc).timetuple().tm_yday
        year = datetime.fromtimestamp(tc_row['timestamp'], timezone.utc).year
        season_cos = np.cos(doy / 366 * np.pi / 2) if year % 4 == 0 else np.cos(doy / 366 * np.pi / 2)
        prev_lat, prev_lon, prev_timestamp = (tc_row['lat'], tc_row['lon'], tc_row['timestamp'])
        cyclone_data_1d[tc_row_index, :] = (tc_row['timestamp'], tc_row['wind'] * 0.514444, tc_row['slp'] * 100, u_wind,
                                            v_wind, season_cos, np.cos(np.radians(tc_row['lat'])), tci_dated,
                                            depth_at_point, u_tc_speed, v_tc_speed, u_dist_2_land_km, v_dist_2_land_km)
        # timestamp, max wind in m/s, seal level pressure, north and west component of wind, season cos, lat cos,
        # thermosphere climate index, depth, north and west component of tc speed, north and west dist to land
        cyclone_data_2d[tc_row_index, :, :] = meteo_data
    non_zero_indxs = np.where(cyclone_data_1d[:, 1] != 0)
    cyclone_dict[cyclone_name] = (cyclone_data_1d[non_zero_indxs], cyclone_data_2d[non_zero_indxs])

with open(config.cyclones_csv_fname.replace('.csv', '.pkl'), 'wb') as f:
    pickle.dump(cyclone_dict, f)
print('done')

