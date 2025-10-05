import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 128, n_classes: int = 6):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, win_size, feat_num)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size) – last layer's hidden state
        out = self.fc1(self.dropout(last_hidden))
        out = self.fc2(self.dropout(out))
        return out
