from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    import timm
except ImportError:
    timm = None


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_MODELS_JSON = Path(__file__).resolve().parent / "models.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(name: str) -> nn.Module:
    """Build supported baseline architectures with single-logit output."""
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 1),
        )
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 1),
        )
    elif name == "efficientnet":
        if timm is None:
            raise ImportError("timm is required for architecture='efficientnet'.")
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=1,
            drop_rate=0.3,
        )
    else:
        raise ValueError(f"Unsupported architecture: {name}")
    return model


def get_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def normalize_checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Accept either full training checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint format is invalid.")


def resolve_checkpoint(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate

    # By convention, checkpoints live under repo-level `models/`.
    model_relative = MODEL_DIR / candidate
    if model_relative.exists():
        return model_relative
    return ROOT_DIR / candidate


def load_registry_from_json(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("models.json must contain a JSON list.")

    registry: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("name", "architecture", "checkpoint")):
            continue
        entry = {
            "name": str(item["name"]),
            "architecture": str(item["architecture"]),
            "checkpoint": str(item["checkpoint"]),
            "img_size": int(item.get("img_size", 224)),
            "best": bool(item.get("best", False)),
        }
        registry.append(entry)
    return registry


def auto_discover_checkpoints() -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    if not MODEL_DIR.exists():
        return discovered

    checkpoint_paths = sorted(
        list(MODEL_DIR.rglob("*.pt")) + list(MODEL_DIR.rglob("*.pth"))
    )
    for ckpt in checkpoint_paths:
        rel = ckpt.relative_to(MODEL_DIR)
        name = rel.stem.replace("_", " ").replace("-", " ").title()
        discovered.append(
            {
                "name": f"Auto: {name}",
                "architecture": "resnet18",
                "checkpoint": str(rel).replace("\\", "/"),
                "img_size": 224,
                "best": False,
            }
        )
    return discovered


def load_registry() -> list[dict[str, Any]]:
    from_json = load_registry_from_json(DEFAULT_MODELS_JSON)
    if from_json:
        return from_json
    return auto_discover_checkpoints()


def load_model_bundle(spec: dict[str, Any]) -> tuple[nn.Module, transforms.Compose]:
    checkpoint_path = resolve_checkpoint(spec["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(spec["architecture"]).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    tf = get_eval_transform(spec["img_size"])
    return model, tf


class InferenceService:
    def __init__(self, registry: list[dict[str, Any]]) -> None:
        self.registry = registry
        self.label_to_spec = {self._display_name(spec): spec for spec in registry}
        self.loaded_models: dict[str, tuple[nn.Module, transforms.Compose]] = {}

    @staticmethod
    def _display_name(spec: dict[str, Any]) -> str:
        return f"{spec['name']} ⭐ Best" if spec.get("best") else spec["name"]

    def labels(self) -> list[str]:
        return list(self.label_to_spec.keys())

    def best_label(self) -> str | None:
        for label, spec in self.label_to_spec.items():
            if spec.get("best"):
                return label
        return None

    def _get_loaded(self, label: str) -> tuple[nn.Module, transforms.Compose]:
        if label not in self.label_to_spec:
            raise ValueError("Unknown model selection.")
        if label not in self.loaded_models:
            self.loaded_models[label] = load_model_bundle(self.label_to_spec[label])
        return self.loaded_models[label]

    def predict(self, image: Image.Image, selected_label: str) -> tuple[dict[str, float], str]:
        if image is None:
            raise gr.Error("Please upload an image before running prediction.")
        model, tf = self._get_loaded(selected_label)

        x = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(x).squeeze()
            p_fake = torch.sigmoid(logit).item()
        p_real = 1.0 - p_fake

        label = "FAKE" if p_fake >= 0.5 else "REAL"
        confidence = p_fake if label == "FAKE" else p_real
        details = (
            f"Prediction: {label}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Model: {selected_label}\n"
            f"Device: {DEVICE.type.upper()}"
        )
        return {"REAL": p_real, "FAKE": p_fake}, details


def build_app() -> gr.Blocks:
    registry = load_registry()
    if not registry:
        raise RuntimeError(
            "No models found. Add Gradio/models.json or place .pt/.pth checkpoints under models/."
        )

    service = InferenceService(registry)
    labels = service.labels()
    best_label = service.best_label()
    best_note = f"Best model: `{best_label}`" if best_label else "Best model not marked yet."

    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("# Deepfake Detector")
        gr.Markdown(
            "Upload an image to classify it as **REAL** or **FAKE**.\n"
            "Select any trained checkpoint from the dropdown."
        )
        gr.Markdown(best_note)

        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            output_label = gr.Label(label="Class Probabilities")

        model_dropdown = gr.Dropdown(
            choices=labels,
            value=best_label or labels[0],
            label="Model Checkpoint",
        )
        details_box = gr.Textbox(label="Prediction Details", lines=4)
        run_btn = gr.Button("Run Inference", variant="primary")

        run_btn.click(
            fn=service.predict,
            inputs=[image_input, model_dropdown],
            outputs=[output_label, details_box],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
