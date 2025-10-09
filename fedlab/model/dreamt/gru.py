import torch.nn as nn

# import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size: int = 8, n_classes: int = 6):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            # dropout=0.3,  # 2 layers
        )

        # 1 layer
        self.fc = nn.Linear(128, n_classes)

        # 2 layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, win_size, feat_num)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size) – last layer's hidden state

        # 1 layer
        out = self.fc(last_hidden)  # (batch, n_classes)

        # 2 layers
        # out = self.dropout(last_hidden)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)  # (batch, n_classes)

        return out
