import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


from utils import create_ds
from tcn_1d import TCN
import config
# TODO TCN 2D
# Create TCN for 2D dataset


def train(model,x_train, x_test, y_train, y_test, limit):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    y_train_torch = torch.Tensor(np.expand_dims(y_train, axis=1))
    y_test_torch = torch.Tensor(np.expand_dims(y_test, axis=1))
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(torch.tensor(x_train, dtype=torch.float32))
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            out = model(torch.tensor(x_test, dtype=torch.float32))
            rmse = root_mean_squared_error(out * 5, y_test * 5)
            mae = mean_absolute_error(out * 5, y_test * 5)
            print(f"Epoch {epoch + 1}: RMSE = {rmse: .2f} m/s, MAE = {mae: .2f} m/s")
            if mae <= limit: break
    return out


x_train, x_test, y_train, y_test, ident_data_total = create_ds()
model = TCN(input_size=8, output_size=1, num_channels=[16, 16, 16], kernel_size=3, dropout=0.0)

out_east = train(model, x_train,  x_test, y_train[:, 1], y_test[:, 1], 0)  # 2.65)  # 2.15) #
out_north = train(model, x_train,  x_test, y_train[:, 0], y_test[:, 0], 0)  # 2.20)  # 1.77) #
# TODO plot the track for TC
# Ident big cyclone - SAM
# Get U - north, lat  and V - wast, coord
# for cycle plot real coord and predicted
out_west = (out_east[8004-6904:8105-6904] * 5).detach().cpu().numpy()
out_north_nd = (out_north[8004-6904:8105-6904] * 5).detach().cpu().numpy()
df_cyclones = pd.read_csv(config.cyclones_csv_fname)
out_df = df_cyclones.iloc[43963-config.forecast_steps:44064]
out_df = out_df.reset_index()
coord_array = np.zeros((len(out_west), 4))

for path_indx in range(len(out_west)):
    current_lat = out_df['lat'].iloc[path_indx + config.forecast_steps]
    current_lon = out_df['lon'].iloc[path_indx + config.forecast_steps]
    time_delta = out_df['timestamp'].iloc[path_indx + config.forecast_steps] - out_df['timestamp'].iloc[path_indx]
    fcstd_lat = out_df['lat'].iloc[path_indx] + (out_north_nd[path_indx, 0] * time_delta / 1000 / 111.11)  # m -> km -> deg
    fcstd_lon = out_df['lon'].iloc[path_indx] + (out_west[path_indx, 0] * time_delta / 1000 / 111.11 / np.cos(np.radians(current_lat)))  # m -> km -> deg
    coord_array[path_indx, :] = (current_lat, current_lon, fcstd_lat, fcstd_lon)


plt.plot(coord_array[:, 1], coord_array[:, 0], 'o-', color='b')
plt.plot(coord_array[:, 3], coord_array[:, 2], 'o-', color='r')
plt.title('Track for hurricane SAM')
plt.legend(['Measured', '6H forecast'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax = plt.gca()
ax.set_xlim([out_df['lon'].min(), out_df['lon'].max()])
ax.set_ylim([out_df['lat'].min(), out_df['lat'].max()])

plt.show()


# Plot the results
plt.scatter(out * 5, y_test * 5)
plt.xlabel('Predicted cyclone intensity (m/s)')
plt.ylabel('Measured cyclone intensity (m/s)')
plt.title('TCN cyclone intensity 12H forecast for NA')
plt.show()

