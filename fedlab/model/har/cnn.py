import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Input: (Batch, 9 channels, 128 sequence length)

        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv1d(
            in_channels=9,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )
        # Output size after conv1 with padding=2 and kernel_size=5 is (Batch, 64, 128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output size after pool1 is (Batch, 64, 64)

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2,
        )
        # Output size after conv2 is (Batch, 128, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output size after pool2 is (Batch, 128, 32)

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )
        # Output size after conv3 is (Batch, 256, 32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Output size after pool3 is (Batch, 256, 16)

        # Calculate the size of the flattened layer
        # Size = channels * sequence_length = 256 * 16 = 4096

        # Fully Connected Layer
        self.fc1 = nn.Linear(256 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x is (Batch, 9, 128)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)  # Flattens all dimensions except batch dimension

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
