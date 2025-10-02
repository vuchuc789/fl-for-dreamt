import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, num_classes=6):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # GRU expects (seq_len, batch, input_size)
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, channels=9, timesteps=128)
        # Rearrange to (seq_len=128, batch, input_size=9)
        x = x.permute(2, 0, 1)

        # GRU forward pass
        output, h_n = self.gru(x)
        # h_n: (num_layers, batch, hidden_size)

        # Take the last hidden state from the last layer
        last_hidden = h_n[-1]

        # Final classification
        out = self.fc(last_hidden)  # (batch, num_classes)

        return out
