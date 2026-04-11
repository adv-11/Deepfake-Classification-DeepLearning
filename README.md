# Deepfake Detector
**CMPE 258-01 | Deep Learning Final Project**

**Team:** Advait Shinde · Nickzad Bayati · Toney Zhen

---

## What This Project Does

We are building a deep learning model that can look at an image and determine whether it is **real or AI-generated**. The final product is a web interface where a user uploads an image and gets a prediction with a confidence score.

---

## Datasets

| Dataset | Description |
|---|---|
| **CIFAKE** | 120,000 images (32×32) — real images from CIFAR-10 + AI-generated images from Stable Diffusion |
| **DeepDetect-2025** | Higher-resolution images (256×256+) — real vs. AI-generated using recent diffusion models |

---

## Plan of Action

### Phase 1 — Exploratory Data Analysis (EDA)
- Understand the distribution, quality, and characteristics of both datasets
- Visualize sample images from each class (real vs. fake)
- Check for class imbalance, image resolution differences, and artifacts
- Document key findings to guide model design

### Phase 2 — Baseline Model
- Train a standard CNN architecture (e.g., **ResNet** or **EfficientNet**) as a baseline
- Evaluate using accuracy and AUROC
- Measure inference latency on CPU/GPU
- Establishes a performance benchmark for comparison

### Phase 3 — Advanced Models & Comparative Analysis
- Train and experiment with multiple model architectures
- Compare performance across models on both datasets
- Explore techniques like knowledge distillation for efficient inference
- Select the best-performing model for the final demo

---

## Demo
A lightweight **Gradio web interface** that lets users:
1. Upload an image
2. Get a **Real / Fake prediction** with a confidence score

---

## Project Notebooks

The EDA, model training work is done in Google Colab | The final models are loaded as checkpoint and loaded into the Gradio UI  organized as follows:

```
 EDA/
   ├── 01_EDA.ipynb
 Baseline/
   ├── baseline_model.ipynb
 Final Approach/
   ├── models.ipynb
   └── model_1.py
   └── model_2.py
        .....
   └── model.pt

 Gradio
   ├── app.py

```

---

## References
1. J. J. Bird, "CIFAKE: Real and AI-Generated Synthetic Images," GitHub, 2023. https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
2. A. Datta, "DeepDetect-2025," Kaggle, 2025. https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025
