# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm  # For confidence interval calculation
from frssa_module_post import frssa_borderline_sampler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_table("Data_after_Cleaning.csv", sep=",")

# Drop 'ID' and 'subject_id' columns
data.drop(columns=['ID', 'subject_id'], inplace=True)

# Selected features and target variable
selected_features = [
    'max_age', 'los_icu', 'sofa_score', 'avg_urineoutput', 'glucose_min', 'glucose_max', 'glucose_average',
    'sodium_max', 'sodium_min', 'sodium_average', 'diabetes_without_cc', 'diabetes_with_cc', 'severe_liver_disease', 
    'aids', 'renal_disease', 'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'sbp_min', 'sbp_max', 'sbp_mean', 
    'dbp_min', 'dbp_max', 'dbp_mean', 'resp_rate_min', 'resp_rate_max', 'resp_rate_mean', 'spo2_min', 'spo2_max', 
    'spo2_mean', 'coma', 'albumin', 'race_Black or African American', 'race_Hispanic or Latin', 'race_Others race', 
    'race_White', 'antibiotic_Vancomycin', 'antibiotic_Vancomycin Antibiotic Lock', 'antibiotic_Vancomycin Enema', 
    'antibiotic_Vancomycin Intrathecal', 'antibiotic_Vancomycin Oral Liquid', 'gender_F', 'gender_M'
]
X = data[selected_features]
y = data['hospital_expire_flag']  # Target variable

# Handle missing values for numerical features
X.fillna(X.median(), inplace=True)

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Preprocessing and scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data **before augmentation** (Post-Splitting Strategy)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === Baseline Evaluation (No Augmentation) ===
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

baseline_acc = accuracy_score(y_test, baseline_pred)
baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
baseline_rec = recall_score(y_test, baseline_pred)

## 2. Determine How Many Synthetic Samples to Generate
n_min = sum(y_train == 1)
n_maj = sum(y_train == 0)
n_needed = n_maj - n_min
if n_needed <= 0:
    print("Classes already balanced.")

rf_results = []

for w in np.arange(0.0, 1.05, 0.05):  # ASST 0.00 to 1.00 step 0.05
    # ASST - Generate synthetic samples using FRSSA
    X_syn, y_syn = frssa_borderline_sampler(X_train, y_train, n_samples=n_needed, expansion_weight=w)

    # Combine real + synthetic training data
    X_train_aug = np.vstack((X_train, X_syn))
    y_train_aug = np.hstack((y_train, y_syn))

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_aug, y_train_aug)

    # Predict on real test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute all metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    rf_results.append({
        'Expansion Weight': w,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'ROC AUC': auc
    })

# Print results for all weights
for row in rf_results:
    print(f"\nExpansion Weight: {row['Expansion Weight']:.2f}")
    print(f"Accuracy : {row['Accuracy']:.4f}")
    print(f"Precision: {row['Precision']:.4f}")
    print(f"Recall   : {row['Recall']:.4f}")
    print(f"F1 Score : {row['F1 Score']:.4f}")
    print(f"ROC AUC  : {row['ROC AUC']:.4f}")

# Prepare metrics
weights = [d['Expansion Weight'] for d in rf_results]
accuracy = [d['Accuracy'] for d in rf_results]
precision = [d['Precision'] for d in rf_results]
recall = [d['Recall'] for d in rf_results]

# === Plot ===
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Accuracy and Baseline Accuracy
plt.plot(weights, accuracy, 'ro-', label='Accuracy')
plt.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline Accuracy')

# Precision and Baseline Precision
plt.plot(weights, precision, 'gs-', label='Precision')
plt.axhline(y=baseline_prec, color='g', linestyle='--', label='Baseline Precision')

# Recall and Baseline Recall
plt.plot(weights, recall, 'b^-', label='Recall')
plt.axhline(y=baseline_rec, color='b', linestyle='--', label='Baseline Recall')

# Axis and layout
plt.xlabel('ASST interpolation weight (1 - expansion weight)')
plt.ylabel('Score')
plt.title('Effectiveness of ASST Weight Optimization (Random Forest)')
plt.grid(True)

# Cleaner legend (center)
plt.legend(loc='center', bbox_to_anchor=(0.30, 0.2), ncol=1)

plt.tight_layout()
plt.show()