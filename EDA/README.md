## 1. Structural Baseline (Data Integrity)

- **What to do:** Plot distributions of resolution, aspect ratios, and file formats. Check for "black frames" or data corruption.
- **The Knowledge:** If all `Fake` images are $256 \times 256$ and `Real` are varying high-res, your model will learn to detect **interpolation/resizing artifacts** rather than "fakeness."
- **Model Impact:** Tells you if you need a specific **Image Resize** strategy (e.g., Letterboxing) or if you must pad images to prevent stretching features.

## 2. Color Space & Intensity Profiling (Intermediate)

- **What to do:** Analyze **RGB**, **YCbCr**, and **LAB** color spaces. Compare the **Mean Pixel Intensity** histograms for both classes.
- **The Knowledge:** Generators often struggle with "Global Illumination." If the `Fake` class has a narrower color gamut or a shift in the Chrominance ($Cb/Cr$) channels, the model might over-rely on color biases.
- **Model Impact:** Decides if you should train on **Grayscale** to force the model to learn structure, or use **Color Jitter** augmentation to prevent the model from "cheating" via color shifts.

## 3. Frequency Domain Analysis (PhD Forensic Layer)

- **What to do:** Perform **Fast Fourier Transforms (FFT)** or **Discrete Cosine Transforms (DCT)** on both classes. Calculate the **Power Spectrum Density (PSD)**.
- **The Knowledge:** GANs often leave "checkerboard" artifacts or high-frequency "spectral peaks" due to upsampling layers. Real images have a natural decay in high frequencies; fakes often have artificial spikes.
- **Model Impact:** If spectral peaks are prominent, a **CNN** (strong at local textures) is preferred. You may also decide to input the **FFT Magnitude** as an extra channel to the model.

## 4. Biological & Geometric Consistency (Domain Specific)

- **What to do:** Use a facial landmark detector (MediaPipe/Dlib) to measure **Inter-ocular distance** or **Symmetry**. Analyze **Error Level Analysis (ELA)** to find compression inconsistencies.
- **The Knowledge:** Deepfakes often exhibit "jitter" in landmarks or mismatched eye reflections (specular highlights) that violate biological physics.
- **Model Impact:** If geometric errors are high, a **Vision Transformer (ViT)** is superior because its **Global Attention** can "see" the relationship between distant features (e.g., left eye vs. right ear) better than a CNN’s local filters.

## 5. Error Surface & Noise Residuals (Forensic Fingerprinting)

- **What to do:** Apply a **High-Pass Filter** (like SRM - Steganalysis Rich Model) to extract the noise component, stripping away the semantic content.
- **The Knowledge:** This reveals the "camera fingerprint" (PRNU). If `Real` images come from diverse sensors but `Fake` images share the same "algorithmic noise," the model will overfit to the noise pattern.
- **Model Impact:** Dictates **Hyperparameters**. If the signal is subtle (noise-based), you need a **smaller Learning Rate** and higher **Weight Decay** to prevent the model from converging on background noise
