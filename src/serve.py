import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

try:
    from .model import get_model
except ImportError:  # pragma: no cover
    from model import get_model


MODEL_PATH = Path("artifacts/checkpoints/model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()
model = None


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ]
)


def load_model():
    global model
    if model is None:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model_obj = get_model(architecture="resnet18", num_classes=10)
        model_obj.load_state_dict(checkpoint["model_state_dict"])
        model_obj.to(DEVICE)
        model_obj.eval()
        model = model_obj


@app.get("/health")
def health() -> dict:
    if model is None:
        try:
            load_model()
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    try:
        load_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return {"probabilities": probs}
