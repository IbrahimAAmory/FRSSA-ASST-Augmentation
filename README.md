# FRSSA-ASST-Augmentation
Implementations of FRSSA augmentation (Pre-Splitting and Post-Splitting) with ASST for Sepsis Mortality

This repository provides code and custom augmentation modules developed to enhance sepsis mortality prediction using ICU datasets. It introduces **FRSSA** (Full Region Synthetic Sampling Approach), a novel technique combining elements of SMOTE and ADASYN while applying dynamic hardness-based interpolation and expansion dispersion by adding Gaussian distribution to create non-linear equation for augmentation.

The repository supports both:
- **Pre-splitting augmentation** (augmentation applied to full data before train-test split)
- **Post-splitting augmentation** (augmentation applied only to training data)

---

## 📊 Datasets Used

1. **MIMIC-IV (v2.2)**  
   Public ICU dataset provided by PhysioNet.  
   🔗 https://physionet.org/content/mimiciv/2.2/

2. **Processed MIMIC-IV CSV**  
   Pre-cleaned and pre-processed MIMIC-IV dataset from:  
   🔗 https://github.com/yuyinglu2000/Sepsis-Mortality

3. **eICU Collaborative Research Database (eICU-CRD v2.0)**  
   Multi-center ICU dataset used for external validation.  
   🔗 https://physionet.org/content/eicu-crd/2.0/

> ⚠ Access to full MIMIC-IV and eICU-CRD requires credentialed PhysioNet approval.

---

## 📂 File Overview

| File | Description |
|------|-------------|
| `frssa_module_pre.py` | Core FRSSA module for **pre-splitting** strategy |
| `frssa_module_post.py` | Enhanced FRSSA module for **post-splitting**, with expansion control |
| `Pre_Splitting_FRSSA_MIMIC_Models.py` | Comparison of 7 classifiers (e.g., RF, XGB, SVM) under pre-splitting |
| `Pre_Splitting_FRSSA_MIMIC_RF.py` | Random Forest evaluation with confidence intervals (pre-splitting) |
| `Post_Splitting_FRSSA_MIMIC_Models.py` | Classifier performance across multiple expansion weights (post-splitting) |
| `Post_Splitting_FRSSA_MIMIC_RF.py` | Random Forest with weight optimization (post-splitting) |

---

## 💻 Requirements

Install the following Python packages:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm

📈 Method Summary (FRSSA)
FRSSA identifies borderline minority samples using K-nearest neighbors, calculates their hardness level, and then:

Allocates more synthetic samples to harder regions.

Generates synthetic points through controlled interpolation.

Applies optional Gaussian dispersion (in post-splitting) to increase diversity.

This leads to better representation of complex minority class boundaries, especially useful in ICU mortality prediction tasks.

📄 License
This code is licensed under a custom academic-use-only MIT-style license.
See the LICENSE file for full terms.

⚠ Usage Terms:
✅ Free for academic, research, and educational purposes.

❌ Commercial use is strictly prohibited without prior written permission.

📚 Citation is required for any publications, derivatives, or academic work using this repository.

Citation format:
Ibrahim A. Amory , Parviz Rashidi-Khazaee , Saleh Yousefi (2025). "Improving Sepsis Mortality Prediction with Machine Learning Using Full Region Synthetic Sampling Approach"
For commercial licensing requests, please contact: [ibrahim.a.amory@gmail.com]

🙏 Acknowledgments: PhysioNet and contributors for MIMIC-IV and eICU datasets and Yuying Lu et al. for the pre-cleaned MIMIC-IV CSV dataset
