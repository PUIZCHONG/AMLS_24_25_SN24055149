# AMLS ELEC0134 (24/25) — Biomedical Image Classification (MedMNIST)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)

This repository contains coursework for **UCL AMLS (Applied Machine Learning Systems)**. The project investigates **automated biomedical image classification** on the **MedMNIST** benchmark using both **deep learning** and **classical machine learning** methods.

---

## Project Summary

The project consists of two core tasks that cover both **binary** and **multi-class** classification scenarios:

- **Task A (BreastMNIST)**: Breast ultrasound image classification  
  **Binary**: malignant vs. (benign + normal)

- **Task B (BloodMNIST)**: Peripheral blood cell image classification  
  **8-class**: eight different blood cell categories

We explore multiple architectures and paradigms, including **CNNs**, **SVM**, **Random Forests**, and **ensemble / hybrid** systems that combine deep and classical models.

---

## Tech Stack

- **Deep Learning**: TensorFlow / Keras (CNN architectures, augmentation, regularization)
- **Machine Learning**: scikit-learn (SVM, Random Forest, Decision Tree, PCA, GridSearchCV)
- **Data & Visualization**: NumPy, Pandas, Matplotlib, Seaborn
- **Experimental Methods**: cross-validation, learning curves, confusion matrices, decision boundary visualization

---

## Experimental Design & Key Results

### Task A — BreastMNIST (Binary Classification)

**Dataset characteristics**
- 28×28 grayscale images
- relatively small sample size

**Models**
- **CNN feature extractor + SVM classifier (best)**  
  A CNN is used to extract high-dimensional feature representations, followed by an SVM for classification.  
  **Best accuracy: 85.3%**
- Baselines: **Decision Tree** and **Random Forest**, with hyperparameter tuning via **GridSearchCV**

**Key techniques**
- **Data augmentation**: random flips, rotations, and scaling to reduce overfitting under limited data
- **PCA visualization**: CNN features reduced to 2D for interpretability; Random Forest decision boundaries plotted in 2D space

---

### Task B — BloodMNIST (8-Class Classification)

**Dataset characteristics**
- 28×28 RGB images
- 8 blood cell classes

**Models**
- **Hybrid ensemble system (best)**  
  Combines:
  - CNN probability outputs  
  - SVM probability outputs built on CNN-extracted features  
  Final prediction uses a **weighted fusion**:  
  **0.3 × CNN + 0.7 × SVM**  
  **Best test accuracy: 95.3%**
- Additional branches: ResNet-28 experiments, a custom 5-layer CNN, and sensitivity analysis on training set size (e.g., 10,000-sample runs)

**Key techniques**
- **Regularization**: Batch Normalization, Dropout (0.3–0.5), and L2 weight decay for better generalization
- **Learning-rate scheduling**: ReduceLROnPlateau to adapt LR during training

---

## Repository Structure

```text
.
├── TaskA.py                 # Task A main script: CNN feature extraction + classifier comparisons
├── Print_data_A.py          # BreastMNIST dataset inspection/analysis utilities
├── Folder B/
│   ├── TaskB.py             # Task B main script: CNN + SVM ensemble system
│   ├── B_10000.py           # Specialized experiment with 10,000 samples
│   ├── B_resnet28.py        # ResNet-28 experimental code
│   ├── B_5layer.py          # 5-layer CNN experimental architecture
│   └── Prin_data_B.py       # BloodMNIST dataset inspection/analysis utilities
└── README.md
