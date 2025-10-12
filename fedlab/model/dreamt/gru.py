import torch
import torch.nn as nn

# import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        n_classes: int = 6,
    ):
        super().__init__()

        self.bidirectional = False
        num_directions = 2 if self.bidirectional else 1
        hidden_size = 128

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
            # dropout=0.3,
        )

        self.fc = nn.Linear(hidden_size * num_directions, n_classes)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * num_directions, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, win_size, feat_num)
        _, h_n = self.gru(x)  # h_n: (num_layers * num_directions, batch, hidden_size)

        if self.bidirectional:
            last_hidden = torch.cat(
                (h_n[-2], h_n[-1]), dim=1
            )  # (batch, hidden_size * 2)
        else:
            last_hidden = h_n[-1]  # (batch, hidden_size)

        out = self.fc(last_hidden)

        # out = self.dropout(last_hidden)
        # out = F.relu(self.fc1(out))
        # out = self.fc2(out)  # (batch, n_classes)

        return out
