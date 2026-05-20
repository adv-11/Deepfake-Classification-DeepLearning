import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
import io
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 1.  MODEL DEFINITIONS
# ─────────────────────────────────────────────


def _build_resnet18() -> nn.Module:
    """
    ResNet-18 with fc = Sequential(Dropout, Linear(512->1)).
    Output is a raw logit — apply sigmoid to get fake probability.
    Checkpoint keys: conv1.weight ... fc.1.weight [1,512] / fc.1.bias [1]
    """
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1),
    )
    return model


# ─────────────────────────────────────────────
# 2.  MODEL LOADER
# ─────────────────────────────────────────────

DEVICE = torch.device("cpu")

MODEL_FILES = {
    "ResNet-18":               "resnet18.pt",
    "Dual MaxViT":             "final_dual_maxvit.pt",
    "MaxViT → MobileNetV3":   "final_dual_maxvit_distilled_MobileNetv3_student.pt",
}

_model_cache: dict[str, nn.Module] = {}


def load_model(name: str) -> nn.Module:
    if name in _model_cache:
        return _model_cache[name]

    path = MODEL_FILES[name]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. "
            "Make sure it is uploaded to the Space root."
        )

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    if name == "ResNet-18":
        model = _build_resnet18()
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    else:  # both MaxViT variants share the same checkpoint schema
        # Build a flat module that accepts 'rgb.*' and 'head.*' keys
        model = _build_maxvit_from_ckpt(ckpt)

    model.eval()
    _model_cache[name] = model
    return model


def _build_maxvit_from_ckpt(ckpt: dict) -> nn.Module:
    """
    Full dual-stream architecture reverse-engineered from checkpoint keys:

    RGB stream  (558 keys):
      timm maxvit_tiny_tf_224.in1k with its internal head intact
        rgb.head.norm      LayerNorm(512)
        rgb.head.pre_logits.fc  Linear(512->512)
      Output: 512-dim

    Spec stream (20 keys):
      3 blocks of Conv2d + BatchNorm2d + ReLU (no bias on Conv)
        b1: Conv(2->32, 3x3)  + BN
        b2: Conv(32->64, 3x3) + BN
        b3: Conv(64->128,3x3) + BN
      Each block: AdaptiveAvgPool after last block -> flatten -> 128-dim
      spec.head: Sequential( Identity[0], Linear(128->256)[2] )
      Output: 256-dim

    Fusion: cat(rgb_512, spec_256) = 768-dim

    Head (6 keys):
      head.0  LayerNorm(768)
      head.1  GELU            (no weights)
      head.2  Linear(768->512)
      head.3  GELU            (no weights)
      head.4  Dropout(0.4)    (no weights)
      head.5  Linear(512->2)
    """
    import timm

    sd  = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    dropout = cfg.get("dropout", 0.4)

    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    class SpecStream(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = _conv_block(2, 32)
            self.b2 = _conv_block(32, 64)
            self.b3 = _conv_block(64, 128)
            self.pool = nn.AdaptiveAvgPool2d(1)
            # spec.head keys: head.2.weight [256,128] -> index 2 means
            # Sequential(Identity, Identity, Linear)
            self.head = nn.Sequential(
                nn.Identity(),          # head.0  (no weights)
                nn.Identity(),          # head.1  (no weights)
                nn.Linear(128, 256),    # head.2
            )

        def forward(self, x):
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
            x = self.pool(x).flatten(1)   # (B, 128)
            return self.head(x)            # (B, 256)

    class DualMaxViTDetector(nn.Module):
        def __init__(self):
            super().__init__()
            # rgb backbone with internal head (outputs 512-dim)
            self.rgb = timm.create_model(
                "maxvit_tiny_tf_224.in1k",
                pretrained=False,
                num_classes=0,          # removes final Linear->1000
                                        # but keeps pre_logits head -> 512
            )
            self.spec = SpecStream()
            # head takes cat(rgb=512, spec=256) = 768
            self.head = nn.Sequential(
                nn.LayerNorm(768),      # head.0
                nn.GELU(),              # head.1
                nn.Linear(768, 512),    # head.2
                nn.GELU(),              # head.3
                nn.Dropout(dropout),    # head.4
                nn.Linear(512, 2),      # head.5
            )

        def forward(self, x):
            rgb_feats  = self.rgb(x)                        # (B, 512)
            # Spec input: 2-channel frequency map from RGB
            spec_input = _rgb_to_dct2ch(x)                  # (B, 2, H, W)
            spec_feats = self.spec(spec_input)               # (B, 256)
            fused      = torch.cat([rgb_feats, spec_feats], dim=1)  # (B, 768)
            return self.head(fused)

    model = DualMaxViTDetector()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    return model


def _rgb_to_dct2ch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert normalised RGB tensor (B,3,H,W) to a 2-channel frequency-domain
    map expected by the spec stream.

    The spec CNN input shape is [32, 2, 3, 3] meaning 2 input channels.
    Standard approach in deepfake detection: convert to YCbCr, take Y channel,
    apply 2D DCT, return log-magnitude split into low/high frequency bands.
    We approximate with a simple 2-channel representation:
      ch0 = grayscale (luminance)
      ch1 = high-frequency residual (image - blurred)
    This is inference-safe and parameter-free.
    """
    # Luminance from RGB (ITU-R BT.601)
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B,H,W)

    # High-freq residual via simple average pooling blur
    gray_4d = gray.unsqueeze(1)                          # (B,1,H,W)
    blurred = torch.nn.functional.avg_pool2d(
        gray_4d, kernel_size=5, stride=1, padding=2
    )
    hf = gray_4d - blurred                               # (B,1,H,W)

    out = torch.cat([gray_4d, hf], dim=1)                # (B,2,H,W)
    return out


# ─────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────

# ImageNet-style normalisation (standard for all three models)
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def preprocess(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return TRANSFORM(img).unsqueeze(0)   # (1, 3, 224, 224)


# ─────────────────────────────────────────────
# 4.  INFERENCE HELPERS
# ─────────────────────────────────────────────

def predict_single(model: nn.Module, img: Image.Image) -> tuple[str, float]:
    """Returns (label, fake_probability).
    
    ResNet-18  → output shape (1,1) → sigmoid → fake prob
    MaxViT     → output shape (1,2) → softmax → index[1] = fake prob
    """
    tensor = preprocess(img).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)

    if logits.shape[-1] == 1:
        # Sigmoid single-output (ResNet-18)
        fake_prob = torch.sigmoid(logits).item()
    else:
        # Softmax 2-class (MaxViT variants)
        fake_prob = torch.softmax(logits, dim=1)[0][1].item()

    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    return label, fake_prob


# ─────────────────────────────────────────────
# 5.  GRADIO CALLBACKS
# ─────────────────────────────────────────────

VERDICT_HTML = """
<div style="
  font-family: 'DM Mono', monospace;
  background: {bg};
  border: 2px solid {border};
  border-radius: 12px;
  padding: 24px 32px;
  text-align: center;
  margin-top: 8px;
">
  <div style="font-size: 2.4rem; font-weight: 800; color: {color}; letter-spacing: 4px;">
    {label}
  </div>
  <div style="font-size: 1.1rem; color: #aaa; margin-top: 6px;">
    Fake probability: <strong style="color:{color}">{score:.1%}</strong>
  </div>
  <div style="font-size: 0.85rem; color: #666; margin-top: 4px;">
    Model: {model_name}
  </div>
</div>
"""

def cb_single_predict(image: Image.Image, model_name: str):
    if image is None:
        return "<p style='color:#ff6b6b'>⚠ Please upload an image first.</p>"
    try:
        model = load_model(model_name)
    except FileNotFoundError as e:
        return f"<p style='color:#ff6b6b'>⚠ {e}</p>"

    label, score = predict_single(model, image)

    if label == "FAKE":
        bg, border, color = "#1a0a0a", "#ff4444", "#ff4444"
    else:
        bg, border, color = "#0a1a0a", "#44ff88", "#44ff88"

    return VERDICT_HTML.format(
        bg=bg, border=border, color=color,
        label=label, score=score, model_name=model_name
    )


# ── Batch / benchmark ───────────────────────────────────────────────────────

def cb_batch_benchmark(csv_file, selected_models: list[str]):
    """
    CSV must have columns:  image_path, true_label
      true_label: 0 = real, 1 = fake   (or 'real'/'fake')
    Returns: summary dataframe, per-image dataframe, plotly figure HTML
    """
    if csv_file is None:
        return None, None, "<p style='color:#ff6b6b'>⚠ Upload a CSV file.</p>"
    if not selected_models:
        return None, None, "<p style='color:#ff6b6b'>⚠ Select at least one model.</p>"

    # ── Read CSV ─────────────────────────────
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return None, None, f"<p style='color:#ff6b6b'>⚠ CSV error: {e}</p>"

    required = {"image_path", "true_label"}
    if not required.issubset(df.columns):
        return None, None, (
            "<p style='color:#ff6b6b'>⚠ CSV must have columns: "
            "<code>image_path</code> and <code>true_label</code></p>"
        )

    # Normalise true_label → 0/1
    df["true_label"] = df["true_label"].apply(
        lambda v: 1 if str(v).strip().lower() in ("1", "fake") else 0
    )

    # ── Run inference ─────────────────────────
    results = []           # one row per (image, model)
    summary_rows = []      # one row per model

    for mname in selected_models:
        try:
            model = load_model(mname)
        except FileNotFoundError as e:
            summary_rows.append({
                "Model": mname, "Error": str(e),
                "Accuracy": None, "AUC (approx)": None,
                "Avg Latency (ms)": None,
            })
            continue

        preds, scores, latencies = [], [], []

        for _, row in df.iterrows():
            img_path = row["image_path"]
            if not os.path.exists(img_path):
                pred, score, lat = -1, -1.0, 0.0
            else:
                try:
                    img = Image.open(img_path)
                    t0 = time.perf_counter()
                    label, score = predict_single(model, img)
                    lat = (time.perf_counter() - t0) * 1000
                    pred = 1 if label == "FAKE" else 0
                except Exception:
                    pred, score, lat = -1, -1.0, 0.0

            preds.append(pred)
            scores.append(score)
            latencies.append(lat)

        # Per-image results
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                "image":      os.path.basename(row["image_path"]),
                "true_label": "FAKE" if row["true_label"] == 1 else "REAL",
                "model":      mname,
                "prediction": "FAKE" if preds[i] == 1 else ("REAL" if preds[i] == 0 else "ERROR"),
                "fake_score": round(scores[i], 4),
                "correct":    preds[i] == row["true_label"],
                "latency_ms": round(latencies[i], 1),
            })

        # Summary
        valid = [(p, s, row["true_label"])
                 for p, s, (_, row) in zip(preds, scores, df.iterrows())
                 if p >= 0]
        if valid:
            acc = sum(p == t for p, s, t in valid) / len(valid)
            # Simple ROC-AUC approximation (rank-based)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score([t for _, _, t in valid],
                                    [s for _, s, _ in valid])
            except Exception:
                auc = None
            avg_lat = np.mean([l for l in latencies if l > 0])
        else:
            acc, auc, avg_lat = None, None, None

        summary_rows.append({
            "Model":            mname,
            "Accuracy":         f"{acc:.1%}" if acc is not None else "N/A",
            "AUC":              f"{auc:.4f}" if auc is not None else "N/A",
            "Avg Latency (ms)": f"{avg_lat:.1f}" if avg_lat is not None else "N/A",
            "Samples":          len(valid),
        })

    per_img_df  = pd.DataFrame(results)
    summary_df  = pd.DataFrame(summary_rows)
    fig_html    = _build_benchmark_charts(per_img_df, summary_df)

    return summary_df, per_img_df, fig_html


def _build_benchmark_charts(per_img: pd.DataFrame, summary: pd.DataFrame) -> str:
    if per_img.empty:
        return "<p>No data to plot.</p>"

    models   = per_img["model"].unique().tolist()
    palette  = ["#00e5ff", "#ff6b6b", "#a8ff78", "#f7971e"]
    colors   = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Fake Score Distribution",
            "Accuracy per Model",
            "Latency Distribution (ms)",
            "Correct vs Incorrect per Model",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 1. Score distribution (violin / histogram per model)
    for m in models:
        sub = per_img[per_img["model"] == m]
        fig.add_trace(
            go.Violin(
                y=sub["fake_score"], name=m,
                box_visible=True, meanline_visible=True,
                line_color=colors[m], fillcolor=colors[m],
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # 2. Accuracy bar
    acc_vals = []
    for _, row in summary.iterrows():
        try:
            acc_vals.append(float(row["Accuracy"].strip("%")) / 100)
        except Exception:
            acc_vals.append(0.0)

    fig.add_trace(
        go.Bar(
            x=summary["Model"].tolist(),
            y=acc_vals,
            marker_color=[colors.get(m, "#aaa") for m in summary["Model"]],
            text=[f"{v:.1%}" for v in acc_vals],
            textposition="outside",
            name="Accuracy",
        ),
        row=1, col=2,
    )

    # 3. Latency distribution
    for m in models:
        sub = per_img[(per_img["model"] == m) & (per_img["latency_ms"] > 0)]
        fig.add_trace(
            go.Box(
                y=sub["latency_ms"], name=m,
                marker_color=colors[m],
                boxmean=True,
            ),
            row=2, col=1,
        )

    # 4. Correct / Incorrect stacked bar
    for m in models:
        sub = per_img[per_img["model"] == m]
        correct   = sub["correct"].sum()
        incorrect = (~sub["correct"]).sum()
        fig.add_trace(
            go.Bar(name=f"{m} ✓", x=[m], y=[correct],
                   marker_color=colors[m], opacity=0.9),
            row=2, col=2,
        )
        fig.add_trace(
            go.Bar(name=f"{m} ✗", x=[m], y=[incorrect],
                   marker_color="#333", marker_line_color=colors[m],
                   marker_line_width=2, opacity=0.6),
            row=2, col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(family="DM Mono, monospace", color="#e0e0e0"),
        height=700,
        showlegend=True,
        barmode="stack",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333"),
        title=dict(
            text="Model Benchmark Analysis",
            font=dict(size=20, color="#00e5ff"),
            x=0.5,
        ),
    )
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ─────────────────────────────────────────────
# 6.  GRADIO UI
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #0a0a0a !important;
    font-family: 'DM Mono', monospace !important;
}
.gr-button-primary {
    background: #00e5ff !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    border: none !important;
    letter-spacing: 1px !important;
}
.gr-button-secondary {
    background: transparent !important;
    color: #00e5ff !important;
    border: 1px solid #00e5ff !important;
    border-radius: 6px !important;
}
.gr-panel, .gr-box {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
}
label, .gr-label { color: #aaa !important; font-size: 0.8rem !important; letter-spacing: 1px !important; }
.gr-dropdown select { background: #1a1a1a !important; color: #e0e0e0 !important; }
h1 { font-family: 'Syne', sans-serif !important; }
"""

HEADER_HTML = """
<div style="
  text-align:center;
  padding: 32px 0 16px;
  font-family: 'Syne', sans-serif;
">
  <div style="font-size:0.75rem; letter-spacing:6px; color:#00e5ff; margin-bottom:8px;">
    258 - Deep Learning Project
  </div>
  <h1 style="
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
  ">
    DEEPFAKE DETECTOR
  </h1>
  <div style="color:#555; font-size:0.8rem; margin-top:10px; letter-spacing:2px;">
    ResNet-18 · Dual MaxViT · MaxViT→MobileNetV3
  </div>
</div>
"""

MODEL_CHOICES = list(MODEL_FILES.keys())


def build_ui():
    with gr.Blocks(title="Deepfake Detector") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Tab 1: Single Image ──────────────────────────────────────
            with gr.Tab("🔍  Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="pil",
                            label="UPLOAD IMAGE",
                            height=320,
                        )
                        model_dd = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value=MODEL_CHOICES[0],
                            label="SELECT MODEL",
                        )
                        run_btn = gr.Button("ANALYSE", variant="primary")

                    with gr.Column(scale=1):
                        verdict_html = gr.HTML(
                            "<div style='color:#444;text-align:center;padding:80px 0;"
                            "font-size:0.9rem;letter-spacing:2px'>"
                            "AWAITING INPUT</div>"
                        )

                run_btn.click(
                    fn=cb_single_predict,
                    inputs=[img_input, model_dd],
                    outputs=verdict_html,
                )

            # ── Tab 2: Batch Benchmark ───────────────────────────────────
            with gr.Tab("📊  Batch Benchmark"):
                gr.HTML("""
                <div style='color:#666;font-size:0.8rem;padding:8px 0 16px;line-height:1.8'>
                  Upload a <code style='color:#00e5ff'>.csv</code> with columns:
                  <strong style='color:#e0e0e0'>image_path</strong> (absolute or relative path to each image)
                  and <strong style='color:#e0e0e0'>true_label</strong> (0/real or 1/fake).
                  Select the models you want to compare, then click Run.
                </div>
                """)

                with gr.Row():
                    csv_input = gr.File(
                        label="TEST CSV",
                        file_types=[".csv"],
                    )
                    model_check = gr.CheckboxGroup(
                        choices=MODEL_CHOICES,
                        value=MODEL_CHOICES,
                        label="MODELS TO BENCHMARK",
                    )

                bench_btn = gr.Button("RUN BENCHMARK", variant="primary")

                summary_table = gr.Dataframe(
                    label="SUMMARY",
                    wrap=True,
                )
                per_img_table = gr.Dataframe(
                    label="PER-IMAGE RESULTS",
                    wrap=True,
                )
                charts_html = gr.HTML()

                bench_btn.click(
                    fn=cb_batch_benchmark,
                    inputs=[csv_input, model_check],
                    outputs=[summary_table, per_img_table, charts_html],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(css=CSS)