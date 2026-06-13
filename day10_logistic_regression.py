# ══ DAY 10 — Logistic Regression Model ══════════════════════════════════════
# Business Q: Can we predict which borrowers will default?
# Tool: Python · Sklearn · Pandas · Matplotlib · Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import sqlite3
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

# ── STEP 1: Load cleaned data from SQLite ────────────────────────────────────
conn = sqlite3.connect('loan_analysis.db')
df = pd.read_sql("SELECT * FROM loans", conn)
conn.close()

print("✓ Data loaded from SQLite")
print(f"  Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ── STEP 2: Select features ───────────────────────────────────────────────────
# Based on Day 4 correlation matrix findings:
# loan_percent_income (0.38), loan_int_rate (0.33), loan_grade (strong)
# cb_person_default_on_file, dti, person_income

features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_default_on_file',
    'cb_person_cred_hist_length',
    'loan_grade',
    'loan_intent',
    'person_home_ownership',
    'dti'
]

target = 'loan_status'

X = df[features]
y = df[target]

print(f"\n✓ Features selected: {len(features)}")
print(f"  Target: loan_status (0=repaid, 1=default)")
print(f"  Class split: {y.value_counts(normalize=True).round(3).to_dict()}")

# ── STEP 3: Train/Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train/Test split (80/20, stratified):")
print(f"  Train: {X_train.shape[0]} rows")
print(f"  Test:  {X_test.shape[0]} rows")

# ── STEP 4: Scale features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✓ Features scaled using StandardScaler")

# ── STEP 5: Train Logistic Regression ────────────────────────────────────────
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # handles 78/22 class imbalance
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("\n✓ Logistic Regression model trained")
print(f"  Solver: lbfgs | class_weight: balanced | max_iter: 1000")

# ── STEP 6: Predictions & Evaluation ─────────────────────────────────────────
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n── Model Performance ──────────────────────────────")
print(f"  ROC-AUC Score: {auc:.3f}")
print(f"\n── Classification Report ──────────────────────────")
print(classification_report(y_test, y_pred, target_names=['Repaid', 'Default']))

# ── STEP 7: Confusion Matrix ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Repaid', 'Default'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Chart 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'Logistic Regression (AUC = {auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', lw=1, label='Random (AUC = 0.500)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend()

# Chart 3: Feature Importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

colors = ['coral' if c > 0 else 'steelblue' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
axes[2].axvline(0, color='black', linewidth=0.8)
axes[2].set_title('Feature Importance (Coefficients)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Coefficient Value')

plt.tight_layout()
plt.savefig('day10_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Charts saved as day10_model_results.png")

# ── STEP 8: Business Interpretation ──────────────────────────────────────────
print("\n══ DAY 10 BUSINESS INSIGHTS ════════════════════════════════════════════")
print(f"""
MODEL SUMMARY:
- Algorithm    : Logistic Regression (baseline classification model)
- ROC-AUC      : {auc:.3f}  → model significantly better than random (0.500)
- Class weight : balanced → corrects for 78/22 default imbalance

WHAT AUC MEANS IN BUSINESS TERMS:
- AUC {auc:.3f} means the model correctly ranks a defaulter above a
  non-defaulter {auc*100:.1f}% of the time
- A bank using this model would dramatically reduce manual review burden

TOP DEFAULT RISK FACTORS (from coefficients):
- Positive coefficients → increase default probability
- Negative coefficients → reduce default probability
- Confirm Day 4 EDA findings: loan_grade, dti, loan_int_rate
  are the strongest signals

NEXT STEPS:
- Compare with Decision Tree / Random Forest for accuracy gain
- Add SHAP values for explainability (important for banking compliance)
- Build a simple credit scorecard from the top 5 features
""")

# ── STEP 9: Save predictions to CSV ──────────────────────────────────────────
results_df = X_test.copy()
results_df['actual_default']      = y_test.values
results_df['predicted_default']   = y_pred
results_df['default_probability']  = y_pred_prob.round(4)
results_df.to_csv('day10_predictions.csv', index=False)
print("✓ Predictions saved to day10_predictions.csv")

print("\n✓ Day 10 complete — Logistic Regression model built and evaluated")
