import torch
import torch.nn as nn

class TemporalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super(TemporalConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x

class TCN2D(nn.Module):
    def __init__(self, in_channels, num_channels, kernel_size, output_size, dropout=0.2):
        super(TCN2D, self).__init__()
        layers = []
        for i, num_out_channels in enumerate(num_channels):
            layers.append(TemporalConv2D(in_channels, num_out_channels, kernel_size, dropout))
            in_channels = num_out_channels

        self.network = nn.Sequential(*layers)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        features = self.network(x)
        pooled_features = self.global_pooling(features).squeeze(-1).squeeze(-1)
        output = self.fc(pooled_features)
        return output


# TODO norm 2D dataset -> test TCN 2D -> create TCNs pictures -> create track images -> TCN 1D +2D
input_channels = 5  # RGB channels
output_size = 10  # Number of classes
num_channels = [64, 128, 256]  # Number of channels in each layer
kernel_size = 3
dropout = 0.2

model = TCN2D(input_channels, num_channels, kernel_size, output_size, dropout)
print(model)
