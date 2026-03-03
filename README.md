# 🧬 GAN-Based Medical Image Augmentation for Skin Lesion Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-79%25-brightgreen?style=flat-square)

**A research-grade comparative study of DCGAN, StyleGAN2, and class-specific GAN 
architectures for synthetic medical image generation and skin lesion classification.**

</div>

---

## 📌 Overview

Medical image datasets suffer from **severe class imbalance** — rare skin lesion types 
have very few training samples, causing CNN models to underperform on minority classes.

This project systematically compares **5 augmentation strategies** across **6 experimental 
phases** to identify the optimal GAN-based approach for improving CNN classification 
performance on the HAM10000 dermoscopic dataset (10,015 images, 7 classes).

> 💡 **Key Finding:** Class-aware synthetic augmentation — not just quantity — is the 
> critical driver of performance. Training 7 class-specific DCGANs and generating 
> 300 balanced samples per class improved CNN accuracy from 76.5% to **79%** and 
> Macro F1 from 0.476 to **0.740**.

---

## 📊 Results Summary

| Method | Accuracy | Macro F1 | Minority Class Behavior |
|--------|----------|----------|------------------------|
| Baseline CNN | 76.5% | 0.476 | Weak minority recall |
| DCGAN (full dataset) + CNN | 74.2% | 0.430 | Worsened imbalance |
| StyleGAN2 + CNN | 74.3% | 0.430 | Still unbalanced |
| **Class-Specific DCGANs + CNN** | **79.0%** | **0.740** | Strong minority gains |
| Class Weights on Augmented Data | 77.3% | 0.720 | Improved balance |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x, Keras |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib |
| Models | DCGAN, StyleGAN2, CNN |
| Dataset | HAM10000 (Kaggle) |
| Environment | Google Colab, Jupyter Notebook |

---

## 🏗️ Architecture

### Generator — DCGAN
```
Input: Random Noise Vector (100-dim)
       ↓
Dense(8×8×256) + BatchNorm + ReLU
       ↓  reshape to (8, 8, 256)
Conv2DTranspose(128, 5×5, stride=2) → (16, 16, 128) + BatchNorm + ReLU
       ↓
Conv2DTranspose(64, 5×5, stride=2)  → (32, 32, 64)  + BatchNorm + ReLU
       ↓
Conv2DTranspose(3, 5×5, stride=2)   → (64, 64, 3)   + tanh
       ↓
Output: Synthetic RGB Image (64×64×3)
```

### Discriminator — DCGAN
```
Input: RGB Image (64×64×3)
       ↓
Conv2D(64,  5×5, stride=2) → (32, 32, 64)  + LeakyReLU(0.2) + Dropout(0.3)
       ↓
Conv2D(128, 5×5, stride=2) → (16, 16, 128) + LeakyReLU(0.2) + Dropout(0.3)
       ↓
Conv2D(256, 5×5, stride=2) → (8,  8,  256) + LeakyReLU(0.2) + Dropout(0.3)
       ↓
Flatten → Dense(1)  [no activation — from_logits=True]
       ↓
Output: Real / Fake logit
```

### Generator — StyleGAN2
```
Input: Latent Vector z (100-dim)
       ↓
Mapping Network: z → w (intermediate latent space)
  • Latent normalization
  • 4× Dense + LeakyReLU(0.2) → w (256-dim)
       ↓
Learned constant 4×4 feature map
       ↓
Style Block ×4 (progressive upsampling with AdaIN):
  4×4  → 8×8   + AdaIN(w) + LeakyReLU
  8×8  → 16×16 + AdaIN(w) + LeakyReLU
  16×16→ 32×32 + AdaIN(w) + LeakyReLU
  32×32→ 64×64 + AdaIN(w) + LeakyReLU
       ↓
toRGB Conv2D(3, 1×1) + tanh
       ↓
Output: Synthetic RGB Image (64×64×3)
```

### CNN Classifier
```
Input: RGB Image (64×64×3)
       ↓
Conv Block 1: Conv2D(32, 3×3, same) + ReLU → MaxPooling → (32×32×32)
       ↓
Conv Block 2: Conv2D(64, 3×3, same) + ReLU → MaxPooling → (16×16×64)
       ↓
Conv Block 3: Conv2D(128, 3×3, same) + ReLU → MaxPooling → (8×8×128)
       ↓
Flatten → Dense(128) + ReLU → Dropout(0.5)
       ↓
Dense(7) + Softmax
       ↓
Output: Class Probabilities (7 skin lesion types)
```

---

## 🔬 Dataset — HAM10000

| Class | Label | Description | Samples |
|-------|-------|-------------|---------|
| Melanocytic Nevi | nv | Benign mole (majority class) | ~6,705 |
| Melanoma | mel | Malignant tumor | ~1,113 |
| Benign Keratosis | bkl | Non-cancerous growth | ~1,099 |
| Basal Cell Carcinoma | bcc | Common skin cancer | ~514 |
| Actinic Keratoses | akiec | Pre-cancerous lesion | ~327 |
| Vascular Lesions | vasc | Blood vessel lesion | ~142 |
| Dermatofibroma | df | Benign fibrous growth | ~115 |

> ⚠️ NV class represents ~67% of all samples. DF and VASC together < 3%.  
> This extreme imbalance is the core problem this project addresses.

---

## 🔁 Experimental Phases

### Phase 1 — Preprocessing
- Extracted HAM10000 images from ZIP files via Google Drive
- Built `image_id → file_path` mapping from metadata CSV
- Resized all images to **64×64×3**
- Normalized to **[-1, 1]** for GAN training, **[0, 1]** for CNN
- Created TensorFlow Dataset with shuffle, batch (64), prefetch

### Phase 2 — DCGAN on Full Dataset
- Trained DCGAN on all 10,015 images (no class conditioning)
- 100 epochs, batch size 64, Adam optimizer (lr=1e-4, β1=0.5)
- Binary cross-entropy loss for both Generator and Discriminator
- Fixed seed used to save 4×4 image grids at epochs 1, 25, 50, 75, 100
- Generated **2,000 synthetic images** — all assigned majority class (NV)

### Phase 3 — CNN Baseline vs DCGAN-Augmented
- Stratified train/val/test split: **70% / 15% / 15%**
- One-hot encoded labels, Adam optimizer (lr=1e-4), 50 epochs, batch 32
- Baseline CNN: **76.5% accuracy, Macro F1: 0.476**
- DCGAN-augmented CNN: **74.2% accuracy** — worse due to reinforced imbalance

### Phase 4 — StyleGAN2 Synthetic Generation
- Implemented lightweight StyleGAN2 with mapping network (z→w) and AdaIN
- Trained 100 epochs on full dataset, generated 2,000 majority-class samples
- StyleGAN2-augmented CNN: **74.3% accuracy** — no improvement over DCGAN
- Confirmed: architecture quality alone cannot fix class imbalance

### Phase 5 — Class-Specific DCGANs ✅ Best Approach
- Trained **7 separate DCGANs** — one per lesion class
- Each DCGAN trained 60 epochs on class-filtered images
- Generated **300 synthetic images per class = 2,100 total**
- Created class-balanced augmented dataset: **12,115 samples**
- 80/20 stratified split, 60 epochs, batch 64
- **79% accuracy, Macro F1: 0.740** — best across all experiments

### Phase 6 — Class-Weighted Training
- Computed balanced class weights via `sklearn.utils.compute_class_weight`
- Applied weights to categorical cross-entropy via Keras `class_weight` argument
- 90/10 train/val split, 60 epochs on augmented dataset
- **77.3% accuracy, Macro F1: 0.720** — improved balance, slightly below Phase 5

---

## 📈 Best Model — Per-Class Results (Phase 5)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NV (majority) | 0.83 | 0.93 | 0.87 | 1,401 |
| DF (minority) | **0.96** | **0.78** | **0.86** | 83 |
| VASC (minority) | **0.97** | **0.88** | **0.92** | 88 |
| BCC | 0.75 | 0.59 | 0.66 | 163 |
| AKIEC | 0.83 | 0.64 | 0.72 | 125 |
| BKL | 0.62 | 0.57 | 0.59 | 280 |
| MEL | 0.58 | 0.46 | 0.51 | 283 |
| **Overall** | **0.79** | **0.69** | **0.74** | **2,423** |

---

## 📁 Repository Structure

```
gan-medical-image-augmentation/
│
├── 📓 ANN_Project2_DCGAN.ipynb
│   │
│   ├── 🔧 SETUP & DATA LOADING
│   │   ├── Google Drive mount & HAM10000 ZIP extraction
│   │   ├── Image directory validation & sample visualization
│   │   ├── Metadata CSV loading (image_id → file_path mapping)
│   │   └── Library imports: TensorFlow, Keras, NumPy, Pandas, Matplotlib
│   │
│   ├── 🗃️ PHASE 1 — PREPROCESSING PIPELINE
│   │   ├── IMG_SIZE = 64×64×3, normalize to [-1, 1] for GAN
│   │   ├── Custom preprocess_image() function
│   │   ├── Load all 10,015 images → NumPy array (float32)
│   │   ├── TF Dataset: shuffle(10000) → batch(64) → prefetch(AUTOTUNE)
│   │   └── Visual verification of preprocessed sample image
│   │
│   ├── 🤖 PHASE 2 — DCGAN ON FULL DATASET
│   │   ├── Generator: Dense(8×8×256) → 3×Conv2DTranspose → tanh output (64×64×3)
│   │   ├── Discriminator: 3×Conv2D + LeakyReLU(0.2) + Dropout(0.3) → Dense(1)
│   │   ├── Loss: BinaryCrossentropy(from_logits=True) for both networks
│   │   ├── Optimizer: Adam(lr=1e-4, β1=0.5) for Generator & Discriminator
│   │   ├── Training: 100 epochs, batch=64, fixed seed for progress grids
│   │   ├── Image grids saved at epochs 1, 5, 10, 15, 20, 25, 50, 70, 85, 100
│   │   ├── generate_synthetic_images(2000) → majority class (NV) label assigned
│   │   └── Synthetic image grid visualization (4×4 grid display)
│   │
│   ├── 🧠 PHASE 3A — BASELINE CNN CLASSIFIER
│   │   ├── prepare_classifier_images(): normalize to [0, 1] for CNN
│   │   ├── Stratified split: 70% train / 15% val / 15% test
│   │   ├── One-hot encoding (7 classes), Adam(lr=1e-4), 50 epochs, batch=32
│   │   ├── CNN: Conv(32)→Pool → Conv(64)→Pool → Conv(128)→Pool → Dense(128)+Dropout(0.5) → Softmax(7)
│   │   ├── Training & validation loss/accuracy curves plotted
│   │   ├── Confusion matrix (Seaborn heatmap)
│   │   ├── Classification report: precision, recall, F1 per class
│   │   └── Result: Baseline Accuracy = 76.5%, Macro F1 = 0.476
│   │
│   ├── 📊 PHASE 3B — DCGAN-AUGMENTED CNN
│   │   ├── Concatenate X_train + 2,000 synthetic images (majority class)
│   │   ├── Same CNN architecture retrained from scratch on augmented data
│   │   ├── Confusion matrix & classification report comparison
│   │   └── Result: Augmented Accuracy = 74.2%, Macro F1 = 0.430 (worse — imbalance reinforced)
│   │
│   ├── 🎯 PHASE 5 — CLASS-SPECIFIC DCGANs  ✅ BEST APPROACH
│   │   ├── load_images_for_label(df, label): filter images per class
│   │   ├── make_tf_dataset(): TF pipeline per class
│   │   ├── train_dcgan_for_label(): trains separate DCGAN for each of 7 classes
│   │   ├── Loop: 7 classes × 60 epochs each → trained_models dict
│   │   ├── Generate 300 synthetic images per class = 2,100 total
│   │   ├── Merge real (10,015) + synthetic (2,100) → 12,115 sample dataset
│   │   ├── 80/20 stratified split, 60 epochs, batch=64
│   │   ├── Classification report: per-class precision, recall, F1
│   │   └── Result: Best Accuracy = 79%, Macro F1 = 0.740
│   │
│   └── ⚖️ PHASE 6 — CLASS-WEIGHTED TRAINING
│       ├── compute_class_weight(balanced, classes, y_train_labels)
│       ├── class_weight dict passed to Keras model.fit()
│       ├── 90/10 train/val split, 60 epochs on augmented dataset
│       ├── Confusion matrix + classification report
│       └── Result: Accuracy = 77.3%, Macro F1 = 0.720
│
├── 📓 ANN_Project2_StyleGAN2.ipynb
│   │
│   ├── 🔧 SETUP & DATA LOADING
│   │   └── [Same as DCGAN notebook — Drive mount, extraction, preprocessing]
│   │
│   ├── 🎨 PHASE 4A — StyleGAN2 ARCHITECTURE & TRAINING
│   │   ├── build_mapping_network(): z (100-dim) → w (256-dim)
│   │   │   ├── Latent normalization layer
│   │   │   └── 4× Dense + LeakyReLU(0.2) layers
│   │   ├── adain(): Adaptive Instance Normalization (style modulation)
│   │   │   ├── gamma & beta projected from Dense(channels) on w vector
│   │   │   └── feat_norm = (feat - mean) / std × gamma + beta
│   │   ├── StyleGAN Generator: learned 4×4 → 8×8 → 16×16 → 32×32 → 64×64
│   │   │   └── Each block: Conv2D + AdaIN(w) + LeakyReLU
│   │   ├── Discriminator: same as DCGAN (3×Conv2D + LeakyReLU + Dropout → Dense(1))
│   │   ├── Loss: BinaryCrossentropy(from_logits=True)
│   │   ├── Optimizer: Adam(lr=1e-4, β1=0.5)
│   │   ├── Training: 100 epochs, batch=64, fixed seed image grids
│   │   └── Generated 2,000 synthetic images → assigned majority class (NV)
│   │
│   ├── 🧠 PHASE 4B — BASELINE CNN ON RAW DATA
│   │   ├── Stratified split: 70% / 15% / 15%
│   │   ├── Same CNN: Conv(32→64→128) + Dense(128) + Dropout(0.5) + Softmax(7)
│   │   ├── Adam(lr=1e-4), 75 epochs, batch=32
│   │   ├── Loss/accuracy curves, confusion matrix, classification report
│   │   └── Result: Baseline Accuracy = 76.5%
│   │
│   └── 📊 PHASE 4C — StyleGAN2-AUGMENTED CNN
│       ├── Concatenate X_train + 2,000 StyleGAN2 synthetic majority-class images
│       ├── Retrained CNN from scratch, 60 epochs, batch=32
│       ├── Confusion matrix + full classification report
│       └── Result: Accuracy = 74.3%, Macro F1 = 0.430
│           (No improvement — architecture quality alone cannot fix imbalance)
│
└── 📄 ANN_Project2_Report.pdf
    ├── Abstract & Problem Statement
    ├── Dataset Analysis — HAM10000 class distribution & imbalance
    ├── Methodology — All 6 experimental phases documented
    ├── Results — Classification reports & 5-model comparison table
    ├── Discussion — Why class-specific GANs outperform all others
    ├── Conclusion & Future Work
    └── References (9 academic citations)
```

---

## 🚀 How To Run

### 1. Clone the Repository
```bash
git clone https://github.com/saiteja-pegallapati/gan-medical-image-augmentation
cd gan-medical-image-augmentation
```

### 2. Install Dependencies
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn jupyter
```

### 3. Download the HAM10000 Dataset
Download from [Kaggle — Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/kmader/skin-lesion-analysis-toward-melanoma-detection)

Place the files in your Google Drive:
```
MyDrive/
├── HAM10000_images_part_1.zip
├── HAM10000_images_part_2.zip
└── HAM10000_metadata.csv
```

### 4. Run on Google Colab *(Recommended)*
Upload notebooks to Colab, mount Google Drive, and run cells in order:
```
1. ANN_Project2_DCGAN.ipynb       → Phases 1, 2, 3
2. ANN_Project2_StyleGAN2.ipynb   → Phases 4, 5, 6
```

> ⚠️ **GPU Required** — Enable in Colab: Runtime → Change Runtime Type → T4 GPU  
> Training all 7 class-specific DCGANs takes ~2–3 hours on Colab free tier.

---

## 💡 Key Takeaways

**1. Single GAN ≠ Solution**
Training one GAN on the full dataset without class conditioning reinforces the 
majority class (NV) and actively worsens imbalance during CNN training.

**2. Architecture Quality Doesn't Fix Imbalance**
StyleGAN2 produced visually superior images compared to DCGAN but achieved 
identical classification results (74.3% vs 74.2%) — because both were trained 
without class conditioning.

**3. Class-Aware Strategy Is the Key**
Training 7 separate DCGANs and generating 300 samples per class produced 
a balanced dataset that significantly improved all minority-class metrics — 
DF: 0.86 F1, VASC: 0.92 F1.

**4. Data Quality > Data Quantity**
2,100 class-balanced synthetic images outperformed 2,000 majority-class 
synthetic images — demonstrating that balance matters more than volume.

**5. Class Weighting Is Complementary, Not Superior**
Class weighting improved minority-class consistency but did not surpass 
the accuracy of class-specific GAN augmentation alone.

---

## 🔮 Future Work

- Explore **Conditional GANs (cGAN)** for more direct class-controlled synthesis
- Scale to higher resolution (**128×128, 256×256**) with larger compute
- Integrate **clinical metadata** (age, sex, localization) for richer augmentation
- Apply **Diffusion Models** as an alternative to GAN-based synthesis
- Evaluate using **FID Score** (Fréchet Inception Distance) to quantify 
  synthetic image quality objectively

---

## 👥 Authors

**Sai Teja Pegallapati**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/sai-teja-pegallapati)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/saiteja-pegallapati)

---

<div align="center">

*University of Houston — MS Engineering Data Science*  

</div>
