import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size: int = 8, n_classes: int = 6):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            # dropout=0.3,
        )

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (batch, win_size, feat_num)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size) – last layer's hidden state
        out = self.fc(last_hidden)
        return out
