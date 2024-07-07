import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from utils import create_ds

# TODO GIT CNN -> TCN 1D -> TCN 2D
# Create TCN for 1D dataset
# Create TCN for 2D dataset


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(7*8, 48)
        self.fc2 = nn.Linear(48, 1)   # Second fully connected layer (output layer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
model = GradientBoostingRegressor(n_estimators=100, loss="squared_error")
x_train_1d = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test_1d = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
model.fit(x_train_1d, y_train)
out = model.predict(x_test_1d)
"""
x_train, x_test, y_train, y_test = create_ds()
x_train_1d = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test_1d = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

y_train_torch = torch.Tensor(np.expand_dims(y_train, axis=1))
y_test_torch = torch.Tensor(np.expand_dims(y_test, axis=0))

model = FullyConnectedNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):  # Adjust the number of epochs as needed
    optimizer.zero_grad()
    predictions = model(torch.Tensor(x_train_1d))
    loss = criterion(predictions, y_train_torch)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    out = model(torch.Tensor(x_test_1d))

print(root_mean_squared_error(out*10, y_test*10))
print(mean_absolute_error(out*10, y_test*10))
plt.scatter(out*10, y_test*10)
plt.xlabel('Predicted cyclone intensity m/s')
plt.ylabel('Measured cyclone intensity m/s')
plt.title('NN cyclone intensity 24H forecast for WP')
plt.show()

