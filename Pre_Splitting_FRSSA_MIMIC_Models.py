# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from frssa_module_pre import frssa_borderline_sampler

# Load and preprocess dataset
data = pd.read_csv("Data_after_Cleaning.csv")
data.drop(columns=['ID', 'subject_id'], inplace=True)

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
y = data['hospital_expire_flag']
X.fillna(X.median(), inplace=True)
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define all 7 models
models = {
    "RF": RandomForestClassifier(random_state=42),
    "DT": DecisionTreeClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LGBM": LGBMClassifier(random_state=42),
    "MLP": MLPClassifier(max_iter=300, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Hyperparameter grids
param_grids = {
    "RF": {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    },
    "DT": {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "GB": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "XGB": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "LGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    },
    "MLP": {
        'hidden_layer_sizes': [(100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

# Function to train, tune, and evaluate models
def evaluate_models(X_data, y_data, description):
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    for name, model in models.items():
        param_grid = param_grids[name]
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                    n_iter=10, cv=3, verbose=0, random_state=42, n_jobs=-1, scoring='roc_auc')
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        results.append({
            "Models": name,
            f"ACC_{description}": accuracy_score(y_test, y_pred),
            f"Precision_{description}": precision_score(y_test, y_pred),
            f"Recall_{description}": recall_score(y_test, y_pred),
            f"F1_{description}": f1_score(y_test, y_pred),
            f"ROC_{description}": roc_auc_score(y_test, y_proba)
        })
    return results

# Evaluate on original data
no_aug_results = evaluate_models(X_scaled, y, "NoAug")

# Evaluate on FRSSA-augmented data
X_aug, y_aug = frssa_borderline_sampler(X_scaled, y)
frssa_results = evaluate_models(X_aug, y_aug, "FRSSA")

# Merge results into one DataFrame
final_results = []
for no_aug, frssa in zip(no_aug_results, frssa_results):
    combined = {**no_aug, **{k: v for k, v in frssa.items() if k != "Models"}}
    final_results.append(combined)

# Output final comparison
results_df = pd.DataFrame(final_results)
print("\nModel Comparison (Tuned): No Augmentation vs FRSSA Augmentation")
print(results_df.to_string(index=False))
