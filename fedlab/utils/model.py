import os
from typing import OrderedDict

import pandas as pd
import torch
from torch.optim import Optimizer


def save_model(
    round_num: int,
    model: torch.nn.Module | OrderedDict,
    optimizer: Optimizer | OrderedDict = None,
    dir_path="model",
):
    os.makedirs(dir_path, exist_ok=True)
    model_path = os.path.join(dir_path, f"model_{round_num:03d}.pth")

    obj = {
        "round": round_num,
        "model_state": model.state_dict()
        if isinstance(model, torch.nn.Module)
        else model,
    }

    if optimizer:
        obj |= {
            "optimizer_state": optimizer.state_dict()
            if isinstance(optimizer, Optimizer)
            else optimizer
        }

    torch.save(obj, model_path)


def save_history(
    round_num: int,
    metrics: dict[str, float],
    dir_path="model",
):
    os.makedirs(dir_path, exist_ok=True)
    history_path = os.path.join(dir_path, "model_history.csv")

    history = (
        pd.read_csv(history_path)
        if os.path.exists(history_path)
        else pd.DataFrame(columns=["round"])
    )

    old_metrics = history[history["round"] == round_num]
    if not old_metrics.empty:
        metrics = old_metrics.to_dict("records")[-1] | metrics
    history = history[history["round"] < round_num]

    metrics = {"round": round_num} | metrics
    metrics = pd.DataFrame([metrics])
    history = pd.concat([history, metrics], ignore_index=True)
    history.to_csv(history_path, index=False)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer = None,
    round_num: int = 0,
    dir_path="model",
) -> tuple[int, dict[str, list[int] | list[float]]]:
    history_path = os.path.join(dir_path, "model_history.csv")
    if not os.path.exists(history_path):
        return 0, {}

    history = pd.read_csv(history_path)
    if round_num == 0 or round_num > history["round"].iloc[-1]:
        round_num = history["round"].iloc[-1]

    history = history[history["round"] <= round_num]
    history = {col: history[col].to_list() for col in history.columns}

    model_path = os.path.join(dir_path, f"model_{round_num:03d}.pth")
    if not os.path.exists(model_path):
        return 0, {}

    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return round_num, history
