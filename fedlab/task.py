from typing import Literal

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
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.types import Device, Tensor
from torch.utils.data import DataLoader

from fedlab.data.dreamt import SLEEP_STAGES, load_data
from fedlab.model.dreamt.gru import Net
from fedlab.utils.model import load_checkpoint, save_history, save_model
from fedlab.utils.plot import LivePlot

MODE: Literal["binary", "multiclass"] = "binary"


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", mode="multiclass"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mode = mode

    def forward(self, logits, targets):
        if self.mode == "binary":
            targets = targets.view(-1, 1).float()
            logits = logits.view(-1, 1)

            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

            probs = torch.sigmoid(logits)
            pt = probs * targets + (1 - probs) * (1 - targets)

            loss = ((1 - pt) ** self.gamma) * bce_loss

            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_t * loss

        elif self.mode == "multiclass":
            logpt = F.log_softmax(logits, dim=1)
            pt = torch.exp(logpt)

            logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

            loss = -((1 - pt) ** self.gamma) * logpt

            if self.alpha is not None:
                loss = self.alpha[targets] * loss

            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha_t = torch.tensor(
                        self.alpha,
                        dtype=torch.float32,
                        device=logits.device,
                    )
                else:
                    alpha_t = self.alpha

                loss = alpha_t[targets] * loss

        else:
            raise Exception(f"Mode {self.mode} is not supported!!")

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: Device,
    class_weights: Tensor = None,
    gamma: float = 0.0,
    proximal_mu: float = 0.0,
    testloader: DataLoader = None,
    checkpoint: int = 0,
    mode: str = "multiclass",
):
    net.to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma, mode=mode)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if proximal_mu > 0.0:
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
                metrics=history["accuracy"] if mode == "multiclass" else history["auc"],
                metric_name="Accuracy" if mode == "multiclass" else "AUC",
            )
            if history
            else LivePlot(
                metric_name="Accuracy" if mode == "multiclass" else "AUC",
            )
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

            if proximal_mu > 0.0:
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

            metrics = test(net, testloader, device, mode=mode)
            save_history(epoch, metrics={"train_loss": train_loss} | metrics)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"
                f" | Test Loss: {metrics['test_loss']:.4f}, "
                f"Threshold: {metrics['threshold']:.4f}, "
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
                metric=metrics["accuracy"] if mode == "multiclass" else metrics["auc"],
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
    plot: bool = False,
    mode: str = "multiclass",
):
    net.to(device)

    if mode == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif mode == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Mode {mode} is not supported!!")

    all_labels, all_probs = [], []
    test_loss = 0.0

    net.eval()
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)

            if mode == "binary":
                outputs = outputs.squeeze(1)  # (batch,)
                probs = torch.sigmoid(outputs)
                loss = criterion(outputs, y.float())
            else:
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, y)

            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            test_loss += loss.item()

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    test_loss = test_loss / len(testloader)

    if mode == "binary":
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        J_scores = tpr - fpr
        best_idx = np.argmax(J_scores)
        best_threshold = thresholds[best_idx].item()
        all_preds = (all_probs >= best_threshold).astype(int)
    elif mode == "multiclass":
        best_threshold = 0.0  # no threshold for multi-class classification
        all_preds = np.argmax(all_probs, 1)
    else:
        raise Exception(f"Mode {mode} is not supported!!")

    accuracy = (all_preds == all_labels).mean().item()
    precision = precision_score(
        all_labels,
        all_preds,
        average="binary" if mode == "binary" else "weighted",
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average="binary" if mode == "binary" else "weighted",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        average="binary" if mode == "binary" else "weighted",
        zero_division=0,
    )

    try:
        if mode == "binary":
            auc = roc_auc_score(
                all_labels,
                all_probs,
            )
        else:
            auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="weighted",
            )
    except ValueError:
        auc = 0.0  # if not computable

    if plot:
        if mode == "multiclass":
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
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=SLEEP_STAGES
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()
        elif mode == "binary":
            stage_mapping = {0: "Sleep", 1: "Wake"}
            raw_labels = np.vectorize(stage_mapping.get)(all_labels)
            raw_preds = np.vectorize(stage_mapping.get)(all_preds)

            print(f"Total : {len(raw_labels):4d} samples")
            values, counts = np.unique(raw_labels, return_counts=True)
            zipped_dict = dict(zip(values, counts))
            label_width = max(len(label) for _, label in stage_mapping.items()) + 1
            for _, label in stage_mapping.items():
                print(
                    f" - {label:<{label_width}}:"
                    f" {zipped_dict[label] if label in zipped_dict else 0:4d}"
                )
            cm = confusion_matrix(
                raw_labels, raw_preds, labels=list(stage_mapping.values())
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=list(stage_mapping.values())
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
            plt.scatter(
                fpr[best_idx],
                tpr[best_idx],
                color="red",
                label=f"Threshold ({best_threshold:.4f})",
                zorder=5,
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
        else:
            raise Exception(f"Mode {mode} is not supported!!")

    return {
        "test_loss": test_loss,
        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


if __name__ == "__main__":
    mode = MODE
    print(f"Mode: {mode}")

    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if mode == "binary":
        model = Net(n_classes=1)
    else:
        model = Net()

    model.to(device)

    trainloader, valloader, class_weights = load_data(
        0,
        batch_size=32,
        alpha_s=0.0,
        alpha_l=0.0,
        mode=mode,
    )

    train(
        net=model,
        trainloader=trainloader,
        epochs=30,
        lr=1e-4,
        weight_decay=1e-6,
        class_weights=class_weights,
        gamma=0.0,
        device=device,
        mode=mode,
        testloader=valloader,
        # checkpoint=30,
    )
