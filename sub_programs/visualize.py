import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ds = xr.open_dataset('/media/xboost/PRO/ML_DS/ML_TC_FCST/GEOS_FP_FCST/inst3_2d_hrc_Nx_20190901_0000.nc4')
"""
t10m_data = ds['TS']
flipperd_map = np.flip(np.fliplr(t10m_data))
plt.imshow(np.concatenate((flipperd_map[:,630:],flipperd_map[:, :630]), axis=1), cmap='jet')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatmap of T10M')
plt.colorbar(label='T10M')
plt.show()"""
dlat = 2
dlon = 4
lat_center = 26
lon_center = -74.5
lat_indxs = (ds['lat'] > (lat_center - dlat)) & (ds['lat'] < (lat_center + dlat))
lon_indxs = (ds['lon'] > (lon_center - dlon)) & (ds['lon'] < (lon_center + dlon))
u10m_subset = ds['U10M'][lat_indxs, lon_indxs]
v10m_subset = ds['V10M'][lat_indxs, lon_indxs]
# ds = xr.open_dataset('/media/xboost/PRO/ML_DS/ML_TC_FCST/GEOS.fp.asm.inst3_3d_asm_Nv.20190901_0000.V01.nc4')

#ts_subset = ds['TS'][lat_indxs, lon_indxs]
# TODO only wind, PS, T10M, Q10M and partially TROPT have radial structure, but not TS
# 4 subplots with wind, PS, QV and EPV

ps_subset = ds['PS'][ lat_indxs, lon_indxs]
wind_speed = np.sqrt(u10m_subset**2 + v10m_subset**2)
wind_direction = np.arctan2(v10m_subset, u10m_subset) * 180 / np.pi
plt.figure(figsize=(8, 6))
plt.quiver(ds['lon'][lon_indxs], ds['lat'][lat_indxs], u10m_subset, v10m_subset, wind_speed, cmap='jet')
plt.title('Wind speed and direction map, 09/01/2019 00:00, Hurricane Dorian')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='m/s')
plt.grid(True)
plt.show()
ds.close()
extent = (ds['lon'][lon_indxs].min(), ds['lon'][lon_indxs].max(), ds['lat'][lat_indxs].min(), ds['lat'][lat_indxs].max())
plt.figure(figsize=(8, 6))
plt.imshow(ps_subset/1000, extent=extent, aspect='auto')
plt.title('Minimal pressure map, 09/01/2019 00:00, Hurricane Dorian')
speed_colorbar = plt.colorbar(label='kPa')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()