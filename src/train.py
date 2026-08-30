import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

try:
    from .dataset import get_dataloaders
    from .model import get_model
except ImportError:  # pragma: no cover
    from dataset import get_dataloaders
    from model import get_model


def log_json(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("model", {})
    config["model"].setdefault("architecture", config.get("model_architecture", "resnet18"))
    config["model"].setdefault("num_classes", config.get("num_classes", 10))

    config.setdefault("data", {})
    config["data"].setdefault("data_dir", config.get("data_dir", "./data"))

    config.setdefault("training", {})
    config["training"].setdefault("batch_size", config.get("batch_size", 64))
    config["training"].setdefault("epochs", config.get("epochs", 10))
    config["training"].setdefault("learning_rate", config.get("learning_rate", 1e-3))
    config["training"].setdefault("early_stopping_patience", config.get("early_stopping_patience", 3))
    config["training"].setdefault("num_workers", config.get("num_workers", 2))

    config.setdefault("output", {})
    config["output"].setdefault("checkpoint_dir", config.get("output_dir", "artifacts/checkpoints"))
    config["output"].setdefault("model_name", config.get("model_name", "model.pt"))

    return config


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc="Train", leave=False, total=len(loader))
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        current_loss = total_loss / total if total else 0.0
        current_acc = correct / total if total else 0.0
        progress.set_postfix(loss=current_loss, acc=current_acc)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc="Validation", leave=False, total=len(loader))
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        current_loss = total_loss / total if total else 0.0
        current_acc = correct / total if total else 0.0
        progress.set_postfix(loss=current_loss, acc=current_acc)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    config_path = Path(os.getenv("TRAINING_CONFIG_PATH", "/app/configs/training_config.yaml"))
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    config = load_config(str(config_path))
    if os.getenv("DATA_DIR"):
        config["data"]["data_dir"] = os.getenv("DATA_DIR")
    if os.getenv("OUTPUT_DIR"):
        config["output"]["checkpoint_dir"] = os.getenv("OUTPUT_DIR")
    if os.getenv("MODEL_NAME"):
        config["output"]["model_name"] = os.getenv("MODEL_NAME")

    log_json(
        {
            "event": "training_config_loaded",
            "config": {
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "data_dir": config["data"]["data_dir"],
                "checkpoint_dir": config["output"]["checkpoint_dir"],
                "model_name": config["output"]["model_name"],
            },
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )
    log_json(
        {
            "event": "dataset_loaded",
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "train_examples": len(train_loader.dataset),
            "val_examples": len(val_loader.dataset),
            "batch_size": config["training"]["batch_size"],
            "device": str(device),
        }
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_json({"event": "checkpoint_dir_ready", "path": str(checkpoint_dir)})
    for epoch in range(config["training"]["epochs"]):
        log_json({"event": "epoch_start", "epoch": epoch + 1, "total_epochs": config["training"]["epochs"]})
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        log_json({"event": "epoch_metrics", **log_entry})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            log_json({"event": "checkpoint_saved", "path": str(save_path), "val_loss": round(val_loss, 4)})
        else:
            patience_counter += 1
        if patience_counter >= patience:
            log_json({"event": "early_stopping", "epoch": epoch + 1, "patience": patience})
            break
    log_json({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
