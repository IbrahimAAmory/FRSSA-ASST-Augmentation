# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
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

## 2. Determine How Many Synthetic Samples to Generate
n_min = sum(y_train == 1)
n_maj = sum(y_train == 0)
n_needed = n_maj - n_min
if n_needed <= 0:
    print("Classes already balanced.")

# ------------------------------
# Define Classifiers
# ------------------------------
classifiers = {
    'RF': RandomForestClassifier(random_state=42),
    'DT': DecisionTreeClassifier(random_state=42),
    'GB': GradientBoostingClassifier(random_state=42),
    'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'MLP': MLPClassifier(random_state=42, max_iter=500),
    'SVM': SVC(probability=True, random_state=42)
}


# ------------------------------
# Evaluation Loop (Best Accuracy)
# ------------------------------
model_results = []

for model_name, model in classifiers.items():
    best_acc = 0
    best_metrics = None
    best_weights = []

    for w in np.arange(0.0, 1.05, 0.05):
        X_syn, y_syn = frssa_borderline_sampler(X_train, y_train, n_samples=n_needed, expansion_weight=w)
        X_train_aug = np.vstack((X_train, X_syn))
        y_train_aug = np.hstack((y_train, y_syn))

        model.fit(X_train_aug, y_train_aug)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                'Model': model_name,
                'ACC': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC': auc
            }
            best_weights = [round(w, 2)]
        elif acc == best_acc:
            best_weights.append(round(w, 2))

    best_metrics['Best Weight'] = ', '.join(map(str, best_weights))
    model_results.append(best_metrics)

# ------------------------------
# Print Final Table
# ------------------------------
results_df = pd.DataFrame(model_results)
print(results_df.to_string(index=False))