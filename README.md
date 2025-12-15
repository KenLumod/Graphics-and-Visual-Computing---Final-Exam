# CNN + CBAM: Architecture Fusion for Fine-Grained Classification

## Overview
This project evaluates the impact of integrating the **Convolutional Block Attention Module (CBAM)** into a standard Convolutional Neural Network (CNN) for fine-grained plant image classification. 

By adding **Channel Attention** ("What to look for") and **Spatial Attention** ("Where to look"), the model significantly outperforms a standard CNN baseline by filtering out background noise and focusing on discriminative flower features.

## Models Compared
1.  **Standard CNN (Baseline):** A custom 3-layer convolutional network.
2.  **CNN + CBAM (Ours):** The same architecture fused with attention modules after the feature extraction blocks.

## Dataset
* **Dataset:** [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* **Source:** Visual Geometry Group, University of Oxford
* **Classes:** 102 flower species
* **Preprocessing:** Resized to 224x224, normalized using ImageNet statistics.

## Experimental Results
The fused architecture demonstrates superior performance in both accuracy and error reduction.

| Metric | Standard CNN | CNN + CBAM | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 56.84% | **63.92%** | **+7.08%** |
| **Test Loss** | 1.6018 | **1.4225** | **-11.2%** |
| **False Positives**| 707 | **591** | **-16.4%** |

> **Key Finding:** The reduction in False Positives (from 707 to 591) indicates that the **Spatial Attention** mechanism successfully prevented the model from mistaking background foliage for flower parts.

## 👁️ Visualization (Grad-CAM)
We employed **Grad-CAM** to interpret the model's focus:
* **Standard CNN:** Attention is often scattered, focusing on background leaves or grass.
* **CNN + CBAM:** Attention is tightly clustered around the flower petals and reproductive structures.
* *(Visual results can be viewed in the `fusion_CNN+CBAM.ipynb` notebook)*

## Requirements
* Python 3.x
* PyTorch & Torchvision
* NumPy & Scikit-learn
* Matplotlib (for plotting results)
* OpenCV (for Grad-CAM overlay)

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision matplotlib opencv-python scikit-learn
    ```

3.  **Run the Notebook:**
    Open `fusion_CNN+CBAM.ipynb` in Jupyter Lab or VS Code.
    * *Note: The script is configured to automatically download the dataset via torchvision, so no manual dataset setup is required.*
