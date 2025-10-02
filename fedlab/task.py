import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.types import Device
from torch.utils.data import DataLoader

from fedlab.data.har import load_data
from fedlab.model.har.gru import Net
from fedlab.utils.plot import LivePlot


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    rr: float,
    device: Device,
    testloader: DataLoader = None,
    plot: bool = False,
):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=rr)

    if plot:
        plotter = LivePlot()

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(trainloader)

        if testloader is not None:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}", end="")

            metrics = test(net, testloader, device)
            print(
                f" | Test Loss: {metrics['loss']:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"Prec: {metrics['precision']:.4f}, "
                f"Rec: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}"
            )
            if plot:
                plotter.update(
                    epoch + 1,
                    train_loss=train_loss,
                    test_loss=metrics["loss"],
                    accuracy=metrics["accuracy"],
                )
        else:
            if plot:
                plotter.update(
                    epoch + 1,
                    train_loss=train_loss,
                )

    if plot:
        plotter.show()

    return {
        "loss": train_loss,
    }


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: Device,
):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    all_labels, all_preds, all_probs = [], [], []
    loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            probs = F.softmax(outputs, dim=1)  # class probabilities
            preds = torch.argmax(probs, 1)

            loss += criterion(outputs, y).item()
            correct += (preds == y).sum().item()

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Metrics
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    # AUC (multi-class one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        auc = 0.0  # if not computable

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


if __name__ == "__main__":
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = Net()
    model.to(device)

    trainloader, valloader = load_data(batch_size=32)

    train(
        net=model,
        trainloader=trainloader,
        epochs=10,
        lr=1e-3,
        rr=1e-5,
        device=device,
        testloader=valloader,
        plot=True,
    )
