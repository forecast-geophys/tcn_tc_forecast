cyclones_csv_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/compact_tc_dataset_2014_2024.csv'
oceans_depth_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/GEBCO_2023_bath_0.1_deg.nc'
gfz_indexes_fname = '/media/xboost/PRO/ML_DS/ML_TC_FCST/gfz_indexes_2014_2024.csv'
land_ratio_map = '/media/xboost/PRO/ML_DS/ML_TC_FCST/land_ratio_in_50km.nc4'
meteo_data_dir = '/media/xboost/PRO/ML_DS/ML_TC_FCST/GEOS_FP_FCST/'
meteo_file_prefix = 'inst3_2d_hrc_Nx'

meteo_param = ['PS', 'QV10M', 'T10M', 'TROPT', 'TS', 'U10M', 'V10M']
cyclone_region = 'NA'
meteo_param_radius = 250
meteo_param_n_bins = 10
forecast_steps = 8  # 4 8
analyzing_interval = 8


# https://github.com/pavel9860/tcn_tc_forecast.git