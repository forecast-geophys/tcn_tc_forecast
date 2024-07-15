import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split

import config

# timestamp, max wind in m/s, seal level pressure, north and west component of wind, season cos, lat cos,
# thermosphere climate index, depth, north and west component of tc speed, north and west dist to land
# add center (or mean) values [ 'QV10M', 'T10M', 'TROPT', 'TS', mean_wind]


def create_ds():
    with open(config.cyclones_csv_fname.replace('.csv', '.pkl'), 'rb') as f:
        cyclone_dict = pk.load(f)
    total_state_data = np.zeros((0, config.analyzing_interval, 17))
    total_fcst_data = np.zeros((0, 3))
    ident_data_total = np.zeros((0, 2))
    step_forward = config.forecast_steps + config.analyzing_interval
    for cyclone_name, cyclone_data in cyclone_dict.items():
        if cyclone_data[0].shape[0] < (config.forecast_steps + config.analyzing_interval + 2): continue
        cyclone_state_data = np.zeros((cyclone_data[0].shape[0] - step_forward, config.analyzing_interval, 17))
        cyclone_forecast_data = np.zeros((cyclone_data[0].shape[0] - step_forward, 3))
        ident_data = np.zeros((cyclone_data[0].shape[0] - step_forward, 2))
        for time_index, time_data in enumerate(cyclone_data[0][: - step_forward, :]):
            forecast_row = cyclone_data[0][time_index + step_forward, :]
            cyclone_forecast_data[time_index, :] = (forecast_row[1], forecast_row[9], forecast_row[10])
            channel_data = cyclone_data[0][time_index: time_index + config.analyzing_interval, 1:]
            center_data = cyclone_data[1][time_index: time_index + config.analyzing_interval, 1:6, 2]
            # radial_data = cyclone_data[1][time_index: time_index + config.analyzing_interval, 2, :]
            # if channel_data[-1, 0] > channel_data[0, 0]: plt.plot(radial_data - np.min(radial_data))
            cyclone_state_data[time_index, :, :] = np.concatenate((channel_data, center_data), axis=1)
            ident_data[time_index, 0] = cyclone_data[0][time_index + step_forward, 0]
            ident_data[time_index, 1] = cyclone_data[0][time_index + step_forward, 1] / 0.54144
        if np.isnan(cyclone_state_data).any() or np.isnan(cyclone_forecast_data).any(): print(cyclone_name); continue
        total_state_data = np.concatenate((total_state_data, cyclone_state_data), axis=0)
        total_fcst_data = np.concatenate((total_fcst_data, cyclone_forecast_data), axis=0)
        ident_data_total = np.concatenate((ident_data_total, ident_data), axis=0)
        plt.show()
    total_state_data[:, :, 0] = total_state_data[:, :, 0] / 10  # max wind m/s
    total_state_data[:, :, 1] = (total_state_data[:, :, 1] - 100000) / 2000  # pressure
    total_state_data[:, :, 7] = total_state_data[:, :, 7] / 1000  # sea depth
    total_state_data[:, :, 8] = total_state_data[:, :, 8] / 5  # north component of speed
    total_state_data[:, :, 9] = total_state_data[:, :, 9] / 5  # east component of speed
    total_state_data[:, :, [10, 11]] = total_state_data[:, :, [10, 11]] / 1000  # dist to land
    total_state_data[:, :, 12] = total_state_data[:, :, 12] * 10  # Q10M
    total_state_data[:, :, 14] = (total_state_data[:, :, 14] - 190) / 40  # TROPT
    total_state_data[:, :, [13, 15]] = total_state_data[:, :, [13, 15]] - 300  # serface and 10M temperature
    total_state_data[:, :, 16] = total_state_data[:, :, 16] / 10  # mean wind speed m/s
    inf_wind_indxs = np.where(total_state_data == np.inf)
    total_state_data[np.where(total_state_data > 100000)] = 0.5
    total_state_data[inf_wind_indxs[0], inf_wind_indxs[1], inf_wind_indxs[2]] = 0
    x_train, x_test, y_train, y_test = train_test_split(total_state_data, total_fcst_data, test_size=0.2, shuffle=False)
    x_train = np.swapaxes(x_train[:, :, [0, 1, 4, 5, 8, 9, 10, 11, 15]], 2, 1)
    x_test = np.swapaxes(x_test[:, :, [0, 1, 4, 5, 8, 9, 10, 11, 15]], 2, 1)
    print('DS prepared')  #
    return x_train, x_test, y_train[:, 0]/10, y_test[:, 0]/10, ident_data_total

