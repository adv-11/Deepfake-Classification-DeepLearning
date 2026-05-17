### Gradio App

The app is available at `Gradio/app.py`.

Setup and run:
1. Install dependencies:
   - `pip install -r Gradio/requirements.txt`
2. Configure model checkpoints:
   - For each checkpoint in `/models/` that should be included in the Gradio app, add an entry to `Gradio/models.json`
   - Set architecture (`resnet18`, `resnet50`, `efficientnet`) and `best` flag
3. Start the app:
   - `python Gradio/app.py`

Notes:
- The app runs inference only (no training).
- If `Gradio/models.json` is missing, the app auto-discovers `.pt` / `.pth` checkpoints from `/models/` (default architecture assumption: `resnet18`).
