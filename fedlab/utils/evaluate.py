import os

import pandas as pd
import torch
from torch import from_numpy, nn
from torch.types import Device
from torch.utils.data import ConcatDataset, DataLoader

from fedlab.data.dreamt import DREAMTDataset
from fedlab.task import Net, test
from fedlab.utils.plot import LivePlot


def evaluate(
    model: nn.Module,
    model_file: str,
    eval_dataloader: DataLoader,
    device: Device = "cpu",
    history_file: str = None,
    dir_path="model",
):
    model_path = os.path.join(dir_path, model_file)

    if not os.path.exists(model_path):
        print("Model not found!!")
        return

    checkpoint = torch.load(model_path, weights_only=False)

    if history_file:
        history_path = os.path.join(dir_path, history_file)

        if not os.path.exists(history_path):
            print("History not found!!")
            return

        history = pd.read_csv(history_path)
        history = history[history["round"] <= checkpoint["round"]]
        plotter = LivePlot(
            epochs=history["round"],
            train_losses=history["train_loss"],
            test_losses=history["test_loss"],
            accuracies=history["accuracy"],
        )
        plotter.show()

    model.load_state_dict(checkpoint["model_state"])
    metrics = test(model, eval_dataloader, device, True)

    print("\nResults:")
    key_width = max(len(k) for k in metrics) + 1
    for k, v in metrics.items():
        print(f" - {k:<{key_width}}: {v:.4f}")


if __name__ == "__main__":
    # participants = [0]
    participants = [i for i in range(20)]

    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    net = Net()
    datasets = ConcatDataset(
        [
            DREAMTDataset(participant, test=True, transform=from_numpy)
            for participant in participants
        ]
    )
    dataloader = DataLoader(
        datasets,
        batch_size=32,
        shuffle=False,
    )

    evaluate(
        net,
        model_file="model_030.pth",
        history_file="model_history.csv",
        eval_dataloader=dataloader,
        device=device,
    )
