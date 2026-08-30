import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.dataset import get_dataloaders, get_transforms
from src.model import get_model
from src.serve import app, load_model
from src.train import evaluate, load_config, train_one_epoch


def test_model_output_shape():
    model = get_model(architecture="cnn", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    assert y.shape == (2, 10)


def test_get_model_resnet18_shape():
    model = get_model(architecture="resnet18", num_classes=8)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    assert y.shape == (2, 8)


def test_get_model_unsupported_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="unknown", num_classes=10)


def test_get_transforms_train_and_eval():
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)

    image = Image.new("RGB", (32, 32), color=(255, 0, 0))

    train_out = train_transform(image)
    eval_out = eval_transform(image)

    assert train_out.shape == (3, 32, 32)
    assert eval_out.shape == (3, 32, 32)
    assert torch.is_tensor(train_out)
    assert torch.is_tensor(eval_out)


def test_get_dataloaders_uses_custom_dataset(monkeypatch):
    class FakeDataset(list):
        def __len__(self):
            return len(self)

    def fake_cifar10(*args, **kwargs):
        return [
            (torch.randn(3, 32, 32), 1),
            (torch.randn(3, 32, 32), 0),
            (torch.randn(3, 32, 32), 1),
            (torch.randn(3, 32, 32), 0),
        ]

    monkeypatch.setattr("src.dataset.datasets.CIFAR10", fake_cifar10)

    train_loader, val_loader = get_dataloaders(data_dir="dummy", batch_size=2, num_workers=0)

    assert len(train_loader) == 2
    assert len(val_loader) == 2
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2


def test_load_config_reads_flat_yaml(tmp_path):
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(
        "batch_size: 8\n"
        "epochs: 4\n"
        "learning_rate: 0.01\n"
        "num_workers: 1\n"
        "num_classes: 7\n"
        "data_dir: ./data\n"
        "output_dir: tmp_artifacts\n"
        "model_name: test_model.pt\n"
        "early_stopping_patience: 2\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["training"]["batch_size"] == 8
    assert config["training"]["epochs"] == 4
    assert config["training"]["learning_rate"] == 0.01
    assert config["model"]["num_classes"] == 7
    assert config["output"]["model_name"] == "test_model.pt"


def test_train_one_epoch_and_evaluate():
    model = get_model(architecture="cnn", num_classes=2)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    eval_loss, eval_acc = evaluate(model, loader, criterion, torch.device("cpu"))

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert isinstance(eval_loss, float)
    assert isinstance(eval_acc, float)
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= eval_acc <= 1.0


def test_serve_health_and_predict(tmp_path):
    model_path = tmp_path / "model.pt"
    model = get_model(architecture="resnet18", num_classes=10)
    torch.save({"model_state_dict": model.state_dict()}, model_path)

    import src.serve as serve_module

    serve_module.MODEL_PATH = model_path
    serve_module.model = None

    client = TestClient(serve_module.app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    predict_response = client.post(
        "/predict",
        files={"file": ("image.png", buffer.getvalue(), "image/png")},
    )

    assert predict_response.status_code == 200
    probabilities = predict_response.json()["probabilities"]
    assert len(probabilities) == 10
    assert all(isinstance(p, float) for p in probabilities)
    assert abs(sum(probabilities) - 1.0) < 1e-6
