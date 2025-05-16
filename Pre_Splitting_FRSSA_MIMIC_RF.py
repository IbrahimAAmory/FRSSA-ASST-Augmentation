# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
from frssa_module_pre import frssa_borderline_sampler


# Load the dataset
data = pd.read_table("Data_after_Cleaning.csv", sep=",")

# Drop 'ID' and 'subject_id' columns
data.drop(columns=['ID', 'subject_id'], inplace=True)

# Identify features and target variable
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

# Handle missing values
X.fillna(X.median(), inplace=True)

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Preprocessing and scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply to your data
X_aug, y_aug = frssa_borderline_sampler(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)


# Function to compute bootstrapped confidence intervals
def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05):
    y_true = np.array(y_true)  # Convert to NumPy array
    y_pred = np.array(y_pred)  # Convert to NumPy array
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return lower, upper

# Initialize and tune the Random Forest model
rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=param_dist, n_iter=100,
    cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='roc_auc'
)

rf_random.fit(X_train, y_train)

# Evaluate model performance on test data
y_pred = rf_random.predict(X_test)
y_pred_proba = rf_random.predict_proba(X_test)[:, 1]

# Compute confidence intervals
accuracy_ci = bootstrap_ci(accuracy_score, y_test, y_pred)
roc_auc_ci = bootstrap_ci(roc_auc_score, y_test, y_pred_proba)

# Compute final metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print final evaluation results
print("\nFinal Test Results:")
print(f"Accuracy:  {accuracy:.4f} (95% CI: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}])")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f} (95% CI: [{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}])")
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: 0', 'Predicted: 1'],
            yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()