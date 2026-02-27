# Skin Lesion Segmentation Project

This project implements a **UNet architecture** in PyTorch for automatic skin lesion segmentation using the **ISIC 2016** dataset. The model is designed to be trained on Google Colab leveraging GPU acceleration.

## 🚀 Getting Started

### 1. Prerequisites
You need to download the **ISIC 2016** dataset ZIP files from the [official challenge page](https://challenge.isic-archive.com/landing/2016/):
- `ISBI2016_ISIC_Part1_Training_Data.zip`
- `ISBI2016_ISIC_Part1_Training_GroundTruth.zip`
- `ISBI2016_ISIC_Part1_Test_Data.zip`
- `ISBI2016_ISIC_Part1_Test_GroundTruth.zip`

### 2. Running on Google Colab (Recommended)
1.  **Clone the Repository**:
    ```python
    !git clone https://github.com/mastermind-zed/Skin-Lesion-Segmentation.git
    %cd Skin-Lesion-Segmentation
    ```
2.  **Install Dependencies**:
    ```python
    !pip install -r requirements.txt
    ```
3.  **Dataset Setup**:
    - Upload the ZIP files to your Google Drive in a folder named `Colab_Notebooks_Data`.
    - Run the notebook cells to mount Drive and extract the data.

### 3. Running Locally
1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Update Paths**:
    - Modify the `drive_data_path` in the notebook to point to your local folder containing the ZIP files.
    - Ensure you have a CUDA-enabled GPU for training.

## ⚠️ Important Notes
- **GitHub Files**: Do NOT push the dataset ZIP files to GitHub (they exceed the 100MB limit). Keep them in Google Drive or local storage.
- **Hardware**: Training is computationally intensive; a GPU (Tesla T4 or better) is highly recommended.

## 📋 Project Status / Roadmap
- [x] Analyze project requirements and dependencies
- [x] Create a `requirements.txt` file
- [x] Setup `.gitignore` for large data files
- [x] Create comprehensive `README.md`
- [/] Push project to GitHub repository
- [ ] Verify execution on Google Colab

---
*Created as part of the Skin Lesion Segmentation Research.*
