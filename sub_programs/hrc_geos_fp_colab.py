import numpy as np
import os
import requests
import netCDF4
from datetime import datetime, timedelta
import time

data_dir = '/media/xboost/PRO/ML_DS/ML_TC_FCST/GEOS_FP_FCST/'

if not os.path.isdir(data_dir): os.mkdir(data_dir)
variables = ['PS', 'QV10M', 'T10M', 'TROPT', 'TS', 'U10M', 'V10M']
# ADD ? shear of the horizontal wind through the depth of the troposphere, low-level relative
# vorticity (EPV, ertels, potential vorticity) at 200, 500 ,2000 and 5000m
# inst3_3d_asm_Nv 'EPV' 'U', 'V'
dimentions = ['lat', 'lon', 'time']
date = datetime(2014, 3, 3)
while date < datetime.now():
    url = (f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{date.year}/M{str(date.month).zfill(2)}"
            f"/D{str(date.day).zfill(2)}/GEOS.fp.asm.inst3_2d_asm_Nx.{date.year}{str(date.month).zfill(2)}"
            f"{str(date.day).zfill(2)}_{str(date.hour).zfill(2)}00.V01.nc4")
    hrc_fname = os.path.join(data_dir,
                             f"inst3_2d_hrc_Nx_{date.year}{str(date.month).zfill(2)}"
                             f"{str(date.day).zfill(2)}_{str(date.hour).zfill(2)}00.nc4")
    date += timedelta(hours=3)
    if os.path.isfile(hrc_fname): continue
    print(hrc_fname, "processing")
    geos_pf_fname = os.path.join(data_dir, url.split("/")[-1])
    for attempts in range(5):
        if os.path.isfile(geos_pf_fname): break
        try:
            response = requests.get(url)
            with open(geos_pf_fname, 'wb') as file:
                file.write(response.content)
        except requests.exceptions.RequestException:
            time.sleep(2**attempts)
            print(geos_pf_fname, 'fail to download')

    dataset = netCDF4.Dataset(geos_pf_fname)
    data = {var: dataset.variables[var][:] for var in variables + dimentions}
    var_names = {var: dataset.variables[var].standard_name for var in variables}
    dim_names = {var: dataset.variables[var].long_name for var in dimentions}
    units = {var: dataset.variables[var].units for var in variables + dimentions}
    hrc_dataset = netCDF4.Dataset(hrc_fname, 'w', format='NETCDF4')
    for var in dimentions:
        hrc_dataset.createDimension(var, len(data[var]))
        dim_var = hrc_dataset.createVariable(var,  np.float32, var)
        dim_var.units = units[var]
        dim_var.long_name = dim_names[var]
        dim_var[:] = data[var]

    for var in variables:
        new_var = hrc_dataset.createVariable(var, np.float32, ('lat', 'lon'), zlib=True, complevel=5)
        new_var[:] = data[var]
    hrc_dataset.close()
    os.remove(geos_pf_fname)

"""dst_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/DST_2013-2024.txt'
ap_f107_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/Kp, ap, Ap, SN, F10.7_2013-2024.txt'
dst_df = pd.read_csv(dst_fname, sep='\t', header=None)
dst_df['mean_dst'] = dst_df.loc[:, 3:26].mean(axis=1)
dst_df = dst_df.rename(columns={0: "year", 1: "month", 2: "day"})
dst_df['datetime'] = pd.to_datetime(dst_df[['year', 'month', 'day']])
ap_f107_df = pd.read_csv(ap_f107_fname, sep='\s+', header=None)
ap_f107_df['mean_Ap'] = ap_f107_df.loc[:, 15:22].mean(axis=1)
ap_f107_df = ap_f107_df.rename(columns={0: "year", 1: "month", 2: "day", 25: 'f10.7'})
ap_f107_df['datetime'] = pd.to_datetime(ap_f107_df[['year', 'month', 'day']])
spw_df = dst_df[['datetime', 'mean_dst']].merge(ap_f107_df[['datetime', 'mean_Ap', 'f10.7']], on='datetime', how='left')
spw_df['no'] = -1.0271 + 1.5553e-2 * spw_df['f10.7'] + 4.0665e-2 * spw_df['mean_Ap'] - 8.2360e-3 * spw_df['mean_dst']
spw_df['tci'] = spw_df['no'].rolling(60).mean()
spw_df = spw_df[~(spw_df['datetime'] < '2014-01-01')]
spw_df.to_csv('gfz_indexes_2024_2024.csv', index=False)"""