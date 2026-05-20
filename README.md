# Deepfake Detector


Team: Advait Shinde · Nickzad Bayati · Toney Zhen

Overview
- Deep learning models to classify images as REAL vs FAKE (AI-generated).
- Includes notebooks for EDA and training, trained checkpoints, a Gradio demo app, and evaluation results.

Datasets ( Kaggle )
- CIFAKE — synthetic + CIFAR-10 real images (32×32)
- DeepDetect-2025 — higher-resolution real vs AI-generated images (256×)

Directory structure (key files)
- Gradio app and models: [Gradio_App/deepfake_detection/](Gradio_App/)
- EDA notebooks: [EDA/01_EDA.ipynb](EDA/01_EDA.ipynb)
- Baseline training: [Baseline Approach/Deepfake_Detector_Baseline.ipynb](Baseline%20Approach/Deepfake_Detector_Baseline.ipynb)
- Final experiments: [Final Approach/](Final%20Approach/)
- Saved results: [results/](results/)

Model Files uploaded to: 
1. https://www.kaggle.com/models/srgmanatee/deepfakedetector/ ( Distilled Model: Maxvit -> Mobile Net)
2. https://huggingface.co/toney02/cmpe258-deepfake-detector-models/tree/main ( Baseline: Resnet + Final: Dual Maxvit)


Quick start — run the Gradio demo (Windows)
1. Open a terminal and create / activate a virtual env (recommended):

```powershell
cd Gradio_App\deepfake_detection
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# or .venv\Scripts\activate   # cmd.exe
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the demo:

```powershell
python app.py
```

Notes:
- The UI uses `Gradio` and runs locally by default. Open the provided local URL in your browser.
- The `requirements.txt` pins CPU-only PyTorch builds to match the included checkpoints.

Models included in the demo
- ResNet-18 — [resnet18.pt](Gradio_App/deepfake_detection/resnet18.pt)
- Dual MaxViT — [final_dual_maxvit.pt](Gradio_App/deepfake_detection/final_dual_maxvit.pt)
- MaxViT → MobileNetV3 (distilled student) — [final_dual_maxvit_distilled_MobileNetv3_student.pt](Gradio_App/deepfake_detection/final_dual_maxvit_distilled_MobileNetv3_student.pt)

How the demo works (summary)
- `Gradio_App/deepfake_detection/app.py` exposes two tabs:
  - Single Image: upload an image and pick a model to get REAL/FAKE + confidence.
  - Batch Benchmark: upload a CSV with `image_path` and `true_label` to run per-model inference and get summary metrics (accuracy, approx AUC, latency) and plots.
- Model loading and preprocessing are implemented in `app.py`. MaxViT variants require `timm` at runtime.

Notebooks and training
- EDA: [EDA/01_EDA.ipynb](EDA/01_EDA.ipynb) and [EDA/02_EDA_Colab.ipynb](EDA/02_EDA_Colab.ipynb)
- Baseline training & evaluation: [Baseline Approach/Deepfake_Detector_Baseline.ipynb](Baseline%20Approach/Deepfake_Detector_Baseline.ipynb)
- Final approach notebooks: files in [Final Approach/](Final%20Approach/)

Results
- JSON outputs from evaluation runs are in the `results/` folder (e.g., dual_maxvit_cifake_test_20260516_194144.json).

Deployment Notes
- To add a new model to the demo, place the checkpoint file in `Gradio_App/deepfake_detection/` and add an entry to `MODEL_FILES` in `app.py`.
- The demo expects model checkpoints to follow the conventions used by the training scripts (see notebook/train files in `Final Approach/` if available).
