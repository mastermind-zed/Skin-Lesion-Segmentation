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
- **Environment:** Seamless integration with Google Colab (High-Performance A100 GPU).

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

## 🏗 Model Architecture: UNet Deep Dive

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
    - **Skip Connections:** Concatenates feature maps from the corresponding encoder level to recover spatial information lost during pooling. This is key for pixel-perfect segmentation.
    - Successive Double Convolutions refine the segmentation mask.
4.  **Final Head:**
    - A `1x1 Convolution` reduces depth to 1 channel, followed by a `Sigmoid` activation (handled by `BCEWithLogitsLoss` during training) to produce a probability map.

---

## 🔄 The Training Regime: Step-by-Step

The training process is optimized for robustness and speed using state-of-the-art PyTorch features.

### 1. Data Pipeline & Augmentation
- **Input Scaling:** Images are resized to 512x512 with bicubic interpolation.
- **Normalization:** Zero-centering using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Albumentations Library:** Used for high-speed augmentations.
    - `ShiftScaleRotate`: Simulates different camera angles and distances.
    - `ElasticTransform`: Handles varied lesion shapes.
    - `ColorJitter`: Accounts for different lighting conditions in dermatoscopy.

### 2. The Loss Function: Combined Loss
We use a weighted sum of two losses to ensure the model learns both local pixel accuracy and global region overlap:
- **Binary Cross Entropy (BCE):** Penalizes pixel-level misclassifications.
- **Dice Loss:** Directly optimizes for the Dice Coefficient, making the model robust to class imbalance (where the lesion is small compared to healthy skin).
- **Formula:** `Total Loss = (1.0 * BCE) + (1.0 * DiceLoss)`

### 3. Optimization Strategy
- **Optimizer:** `AdamW` (Adam with Decoupled Weight Decay). Weight decay is set to `1e-4` to prevent overfitting.
- **Learning Rate Scheduler:** `CosineAnnealingLR`. It starts at `1e-4` and smoothly decays following a cosine curve, helping the model settle into narrow optima.

### 4. Advanced Training Features
- **Mixed Precision (AMP):** We use `torch.cuda.amp` to perform calculations in `float16`. This halves memory usage and significantly speeds up training on high-end hardware like the NVIDIA A100 without loss of precision.
- **Checkpointing:** The model automatically saves the `best_model.pth` based on the lowest validation loss recorded.

---

## 📊 Results & Evaluation

Evaluation on the **379 Test Images**:

| Metric | Score | 
| :--- | :--- |
| **Dice Coefficient** | **86.02%** |
| **IoU (Jaccard Index)** | **77.79%** |
| **Pixel Accuracy** | **94.50%** |

### Test-Time Augmentation (TTA)
We compared standard inference against an 8-way TTA (averaging predictions from 8 different flips/rotations).
- **Inference with TTA:** 85.95%
- **Standard Inference:** 86.02%
*The model's base robustness is high enough that standard inference remains the most efficient high-accuracy path.*

---

## 🖼 Visualizations

### 🔬 Test Set Predictions
The following image shows how the model handles the test set data, accurately identifying lesion boundaries.

![Test Set Sample](skin_lesion_segmentation_result_png_1772238009593.png)

### 📤 Custom Inference (Upload a New Image)
The project includes a dedicated pipeline for user-uploaded images. This simulates a real-world clinical application.
- **Step 1:** Upload any raw dermatoscopic image.
- **Step 2:** The system automatically resizes, normalizes, and runs the image through the UNet.
- **Step 3:** The output includes the probability map (confidence) and the final thresholded mask.

![Inference Sample](inference_sample_result_png_1772238564559.png)

---
*Developed by mastermind-zed as part of the Skin Lesion Research.*
