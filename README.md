# Skin Lesion Segmentation: Deep Learning with UNet

![Skin Lesion Segmentation Sample](https://raw.githubusercontent.com/mastermind-zed/Skin-Lesion-Segmentation/main/visualization_sample.png)

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Resources & Requirements](#-resources--requirements)
3. [Project Setup](#-project-setup)
4. [Model Implementation](#-model-implementation)
5. [Training Workflow](#-training-workflow)
6. [Results & Evaluation](#-results--evaluation)
7. [Visualizations](#-visualizations)

---

## 🔬 Project Overview
This project focuses on the automated segmentation of skin lesions from dermatoscopic images using a **UNet Architecture**. High-accuracy segmentation is critical for early detection of melanoma and other skin cancers.

**Key Achievements:**
- **Dice Coefficient:** 86.02%
- **IoU (Jaccard Index):** 77.79%
- **Environment:** Seamless integration with Google Colab (T4 GPU).

---

## 🛠 Resources & Requirements

### Dataset: ISIC 2016
We use the Challenge Part 1 (Segmentation) data from the [International Skin Imaging Collaboration (ISIC)](https://challenge.isic-archive.com/landing/2016/).
- **Train:** 900 images (80/20 split used for Val).
- **Test:** 379 official challenge test images.

### Software Stack
- **Framework:** PyTorch & TorchVision
- **Augmentation:** Albumentations
- **Visualization:** Matplotlib, OpenCV
- **Metrics:** Dice, IoU, Pixel Accuracy, Precision, Recall

---

## 🚀 Project Setup

### 1. Google Colab (Recommended)
1.  **Clone & CD:**
    ```python
    !git clone https://github.com/mastermind-zed/Skin-Lesion-Segmentation.git
    %cd Skin-Lesion-Segmentation
    ```
2.  **Dataset Upload:**
    Create a folder `Colab_Notebooks_Data` in your Google Drive and upload the ISIC 2016 ZIP files.
3.  **Run Notebook:** Open `Skin Lesion Project.ipynb` in Colab, mount Drive, and execute cells.

### 2. Local Setup
```bash
git clone https://github.com/mastermind-zed/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation
pip install -r requirements.txt
```
*Note: Requires a CUDA-enabled GPU (8GB+ VRAM recommended).*

---

## 🏗 Model Implementation: UNet Deep Dive

The model is a standard **UNet (Encoder-Decoder)** with skip connections, featuring approximately 97.9 Million parameters.

### Chronological Processing Steps:

1.  **Encoder (Downsampling):**
    - The input image (512x512x3) passes through 4 levels of "Double Convolution" blocks.
    - Each block: `Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU`.
    - `MaxPool2d (2x2)` reduces spatial resolution while increasing feature depth (64, 128, 256, 512).
2.  **Bottleneck:**
    - The deepest level processes highly abstract features at a resolution of 32x32 with 1024 channels.
3.  **Decoder (Upsampling):**
    - `ConvTranspose2d` doubles spatial resolution.
    - **Skip Connections:** Concatenates feature maps from the corresponding encoder level to recover spatial information lost during pooling.
    - Successive Double Convolutions refine the segmentation mask.
4.  **Final Head:**
    - A `1x1 Convolution` reduces depth to 1 channel, followed by a `Sigmoid` activation (handled by `BCEWithLogitsLoss` during training) to produce a probability map.

---

## 🔄 Training Workflow

1.  **Data Preprocessing:**
    - Images resized to 512x512.
    - Normalization using ImageNet statistics.
2.  **Augmentation:**
    - Training: Horizontal/Vertical Flips, Random Rotate, Color Jitter, Elastic Transforms.
    - Validation: Strictly Resize and Normalize.
3.  **Loss Function:**
    - **Combined Loss:** `0.5 * BCEWithLogits + 0.5 * DiceLoss`. This balances pixel-wise accuracy with overall overlap.
4.  **Learning Strategy:**
    - **Optimizer:** AdamW.
    - **Scheduler:** Cosine Annealing with Warm Restarts.
    - **Mixed Precision:** AMP (Automatic Mixed Precision) for faster training on T4 GPUs.

---

## 📊 Results & Evaluation

Evaluation on the **379 Test Images**:

| Metric | Score | 
| :--- | :--- |
| **Dice Coefficient** | **86.02%** |
| **IoU (Jaccard Index)** | **77.79%** |
| **Pixel Accuracy** | **94.50%** |

### Test-Time Augmentation (TTA)
We experimented with 8-way TTA (Flips + Rotations).
- **Inference with TTA:** 85.95%
- **Standard Inference:** 86.02%
*Observation: The model exhibits high spatial robustness, as TTA yields nearly identical (slightly lower) scores compared to standard prediction.*

---

## 🖼 Visualizations
The model successfully identifies irregular boundaries and varied lesion textures.

- **Original Image:** Raw dermatoscopic data.
- **Probability Heatmap:** Visualizes model confidence (Jet colormap).
- **Binary Mask:** Thresholded at 0.5 for final segmentation.

---
*Developed by mastermind-zed as part of the Skin Lesion Research.*
