import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.types import Device, Tensor
from torch.utils.data import DataLoader

from fedlab.data.dreamt import SLEEP_STAGES, load_data
from fedlab.model.dreamt.gru import Net
from fedlab.utils.model import load_checkpoint, save_history, save_model
from fedlab.utils.plot import LivePlot


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: Device,
    class_weights: Tensor = None,
    proximal_mu: float = 0.0,
    testloader: DataLoader = None,
    checkpoint: int = 0,
):
    net.to(device)
    class_weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if proximal_mu > 0:
        global_params = [p.detach().clone() for p in net.parameters()]

    if testloader:
        checkpoint, history = load_checkpoint(
            net,
            optimizer,
            checkpoint,
        )
        plotter = (
            LivePlot(
                epochs=history["round"],
                train_losses=history["train_loss"],
                test_losses=history["test_loss"],
                accuracies=history["accuracy"],
            )
            if history
            else LivePlot()
        )

    epochs += checkpoint

    for epoch in range(checkpoint + 1, epochs + 1):
        net.train()
        train_loss = 0.0
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(X), y)

            if proximal_mu > 0:
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                    net.parameters(), global_params
                ):
                    proximal_term += (
                        torch.linalg.vector_norm(
                            local_weights - global_weights,
                            ord=2,
                        )
                        ** 2
                    )

                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(trainloader)

        if testloader:
            save_model(epoch, model=model, optimizer=optimizer)

            metrics = test(net, testloader, device)
            save_history(epoch, metrics={"train_loss": train_loss} | metrics)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"
                f" | Test Loss: {metrics['test_loss']:.4f}, "
                f"Acc: {metrics['accuracy']:.4f}, "
                f"Prec: {metrics['precision']:.4f}, "
                f"Rec: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, "
                f"AUC: {metrics['auc']:.4f}"
            )
            plotter.update(
                epoch,
                train_loss=train_loss,
                test_loss=metrics["test_loss"],
                accuracy=metrics["accuracy"],
            )

    if testloader:
        plotter.show()

    return {
        "train_loss": train_loss,
    }


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: Device,
    show_confusion_matrix: bool = False,
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

    if show_confusion_matrix:
        le = LabelEncoder().fit(SLEEP_STAGES)

        raw_labels = le.inverse_transform(all_labels)
        raw_preds = le.inverse_transform(all_preds)

        print(f"Total : {len(raw_labels):4d} samples")
        values, counts = np.unique(raw_labels, return_counts=True)
        zipped_dict = dict(zip(values, counts))
        label_width = max(len(label) for label in SLEEP_STAGES) + 1
        for label in SLEEP_STAGES:
            print(
                f" - {label:<{label_width}}:"
                f" {zipped_dict[label] if label in zipped_dict else 0:4d}"
            )

        cm = confusion_matrix(raw_labels, raw_preds, labels=SLEEP_STAGES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SLEEP_STAGES)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

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
        "test_loss": loss,
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

    trainloader, valloader, class_weights = load_data(
        0,
        batch_size=32,
        # alpha_s=0.0,
        # alpha_l=0.0,
    )

    train(
        net=model,
        trainloader=trainloader,
        epochs=30,
        lr=1e-4,
        weight_decay=1e-6,
        class_weights=class_weights,
        device=device,
        testloader=valloader,
        # checkpoint=30,
    )
