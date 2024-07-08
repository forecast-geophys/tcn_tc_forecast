import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from utils import create_ds

# TODO TCN 1D -> TCN 2D
# Create TCN for 1D dataset
# Create TCN for 2D dataset


class cnn_regr(nn.Module):
    def __init__(self):
        super(cnn_regr, self).__init__()

        self.conv1 = nn.Conv1d(7, 16, 5)  # 17 input channels, 24 output channels, 10 kernel size
        # self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(16, 32, 3)
       #  self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


def train(model,x_train, x_test, y_train, y_test):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    y_train_torch = torch.Tensor(np.expand_dims(y_train, axis=1))
    y_test_torch = torch.Tensor(np.expand_dims(y_test, axis=0))
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(torch.tensor(x_train, dtype=torch.float32))
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        print(loss)
        with torch.no_grad():
            model.eval()
            out = model(torch.tensor(x_test, dtype=torch.float32))
            rmse = root_mean_squared_error(out * 10, y_test * 10)
            mae = mean_absolute_error(out * 10, y_test * 10)
            print(f"Epoch {epoch + 1}: RMSE = {rmse:.2f} m/s, MAE = {mae:.2f} m/s")
    return out


x_train, x_test, y_train, y_test = create_ds()
model = cnn_regr()
out = train(model, x_train,  x_test, y_train, y_test)


# Plot the results
plt.scatter(out * 10, y_test * 10)
plt.xlabel('Predicted cyclone intensity (m/s)')
plt.ylabel('Measured cyclone intensity (m/s)')
plt.title('CNN cyclone intensity 24H forecast for NA')
plt.show()

