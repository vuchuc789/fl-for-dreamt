import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size=8, n_classes=6):
        super().__init__()

        self.bidirectional = True
        num_directions = 2 if self.bidirectional else 1
        hidden_size = 128

        # Expect input: (batch, seq_len=1920, input_size)
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=0.3,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * num_directions, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, feat_num) (32, 1920, 8)
        x = x.permute(0, 2, 1)  # (batch, feat_num, seq_len) (32, 8, 1920)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)  # (batch, reduced_seq_len, channels) (32, 120, 64)

        _, h_n = self.gru(x)  # (num_layers, batch, hidden_size) (2, 32, 128)

        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]

        out = self.dropout(last_hidden)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)  # (batch, n_classes) (32, 6)

        return out
