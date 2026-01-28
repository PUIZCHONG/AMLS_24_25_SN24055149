# 👋 Hi, I'm Pui Yuen Zhuang （庄培源）

🎓 MSc in Integrated Machine Learning Systems, University College London (2024 – 2025)  
🎓 BEng in Electronic and Communications Engineering, University of Leeds (2021 – 2024)  

💡 Interests: Artificial Intelligence · Machine Learning · Deep Learning · Embedded Systems  

---

# 🚀 Projects Overview

| Project                     | Domain            | Tech Stack              | Core Contribution                                    |
|-----------------------------|-------------------|-------------------------|-----------------------------------------------------|
| [**MedMNIST Classification**](https://github.com/PUIZCHONG/AMLS_24_25_SN24055149) | Medical CV        | CNN, SVM, PCA           | Hybrid CNN-SVM architecture for robust diagnostic imaging. |
| [**NBME Clinical Extraction**](https://github.com/PUIZCHONG/DLNLP_assignment_25_24055149) | Medical NLP       | DeBERTa-v3, FGM, NER    | Complex offset alignment and adversarial training for clinical NER. |
| [**GENIE Skill Training**](https://github.com/PUIZCHONG/ELEC0054_final) | RL / Generative   | JAX, World Models, KL-Distill | 3-stage policy learning (KL → WBC → RL) in latent action spaces. |
| [**Cassava Disease Diagnosis**](https://github.com/PUIZCHONG/AMLS_II_assignment24_25) | Agricultural CV | EfficientNet, CropNet, Grad-CAM | Ensemble learning with model interpretability (Explainable AI). |

---

## 🔍 Detailed Project Showcases

### 🧬 [1. Biomedical Image Classification (MedMNIST)](https://github.com/PUIZCHONG/AMLS_24_25_SN24055149)
A comprehensive study on medical image diagnosis using the MedMNIST suite, covering both binary (BreastMNIST) and multi-class (BloodMNIST) scenarios.  
**Key Innovation:**  
- Implemented a Hybrid Architecture where a Deep CNN acts as a feature extractor for a downstream SVM, outperforming end-to-end deep learning on small medical datasets.  
**Performance:**  
- Achieved 95.3% accuracy on BloodMNIST using an ensemble of CNN-extracted features and probability weighting.  
**Visualization:**  
- Used PCA (Principal Component Analysis) to visualize decision boundaries in high-dimensional feature spaces.

---

### ✍️ [2. NBME - Score Clinical Patient Notes (NLP)](https://github.com/PUIZCHONG/DLNLP_assignment_25_24055149)
A Kaggle-based competition project focused on identifying clinical concepts within patient history notes.  
**Key Innovation:**  
- Developed a robust character-level span detection pipeline.  
- Solved the critical challenge of Offset Correction—maintaining label integrity after text standardization (e.g., expanding medical abbreviations like "htn" to "hypertension").  
**Advanced Techniques:**  
- Utilized Adversarial Training (FGM) and Focal Loss to handle class imbalance and improve model generalization on unstructured text.  
**Backbone:**  
- Fine-tuned DeBERTa-v3-large for Named Entity Recognition (NER).

---

### 🤖 [3. GENIE Skill Policy Training (RL & World Models)](https://github.com/PUIZCHONG/ELEC0054_final)
This project extends JAFAR (an open-source reimplementation of Google's Genie) to train skill policies (
9
θ
) within generative interactive environments.  
**Key Innovation:**  
- Implemented a Three-Stage Training Pipeline:  
  1. **KL Distillation** for initial policy alignment.  
  2. **Weighted Behavioral Cloning (WBC)** for high-quality trajectory imitation.  
  3. **RL Fine-tuning** for end-to-end reward optimization.  
**Framework:**  
- Entirely built on JAX for high-performance hardware acceleration.  
**Metrics:**  
- Evaluated using perceptual metrics (SSIM, PSNR) and video quality metrics (FVD).

---

### 🍃 [4. Cassava Leaf Disease Classification](https://github.com/PUIZCHONG/AMLS_II_assignment24_25)
An automated system for diagnosing cassava plant diseases to support global food security.  
**Key Innovation:**  
- Developed a Dual-Model Ensemble combining EfficientNetB4 (global context) and CropNet (local texture details).  
**Explainable AI (XAI):**  
- Integrated Grad-CAM visualizations to highlight the specific regions of a leaf that triggered a "diseased" classification, providing transparency for agricultural experts.  
**Pipeline:**  
- Includes a full EDA suite for RGB distribution analysis and data augmentation impact studies.

---

## 🛠️ Technical Skillset

- **Deep Learning:** PyTorch, TensorFlow/Keras, JAX, Flax  
- **NLP:** HuggingFace Transformers, Tokenization Alignment, NER, DeBERTa  
- **Computer Vision:** Image Segmentation, Classification, Grad-CAM, Perceptual Metrics  
- **Reinforcement Learning:** World Models, Policy Gradient, Latent Space Dynamics  
- **Data Science:** Scikit-Learn, Pandas, NumPy, OpenCV, PCA, GridSearchCV  
- **Tools:** Git, Kaggle P100/T4, Weights & Biases (WandB), Linux/Bash