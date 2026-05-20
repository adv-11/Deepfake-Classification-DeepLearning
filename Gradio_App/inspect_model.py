"""
Run this locally BEFORE uploading to HuggingFace to verify
each model loads and produces the correct output shape.

Usage:
    python inspect_model.py
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
import numpy as np

DEVICE = torch.device("cpu")


# ── ResNet-18 ────────────────────────────────────────────────────────────────
# Checkpoint stores weights under model_state_dict with standard ResNet key
# names (conv1.weight, layer1.0.conv1.weight, fc.1.weight, fc.1.bias).
# The fc was replaced with nn.Sequential(Dropout, Linear(512→2)) during training.
# NO wrapper class needed — just build torchvision ResNet18 + same fc and load directly.

def test_resnet(path="resnet18.pt"):
    print("\n" + "="*60)
    print("ResNet-18")
    print("="*60)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # fc.1.weight is [1, 512] → single sigmoid output, NOT 2-class softmax
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)          # shape (1, 1)
        fake_prob = torch.sigmoid(out).item()
    print(f"  Output shape : {out.shape}  ✓  (sigmoid, 1-class)")
    print(f"  Fake prob    : {fake_prob:.4f}")
    print(f"  Val AUC      : {ckpt['val_auc']:.4f}")
    print("  PASS ✓")


# ── MaxViT variants ──────────────────────────────────────────────────────────

def test_maxvit(path, label):
    print("\n" + "="*60)
    print(label)
    print("="*60)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"]
    print(f"  Keys in state_dict : {len(sd)}")
    print(f"  Config             : {ckpt.get('config', {})}")
    print(f"  Val metrics        : {ckpt.get('val_metrics', {})}")

    # Check head output size
    head_w = sd.get("head.5.weight")
    if head_w is not None:
        print(f"  head.5 (classifier): {head_w.shape}  → {head_w.shape[0]} classes ✓")
    else:
        print("  WARNING: head.5.weight not found")
    print("  State dict loaded ✓  (full model test requires architecture match)")


if __name__ == "__main__":
    import os

    files = {
        "resnet18.pt": ("resnet", None),
        "final_dual_maxvit.pt": ("maxvit", "Dual MaxViT"),
        "final_dual_maxvit_distilled_MobileNetv3_student.pt": ("maxvit", "MaxViT → MobileNetV3 Student"),
    }

    for fname, (kind, label) in files.items():
        if not os.path.exists(fname):
            print(f"\n⚠  {fname} not found — skipping")
            continue
        if kind == "resnet":
            test_resnet(fname)
        else:
            test_maxvit(fname, label)

    print("\n" + "="*60)
    print("All available models inspected.")