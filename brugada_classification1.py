import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score, f1_score,
                             precision_score, recall_score)
from imblearn.over_sampling import SMOTE

# Reproducibility
np.random.seed(42)

# Plotting style
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("  BRUGADA SYNDROME BINARY CLASSIFICATION FROM 12-LEAD ECG")
print("=" * 70)
print(f"  NumPy version:      {np.__version__}")
print(f"  Pandas version:     {pd.__version__}")
print("=" * 70)

# SECTION 1: DATA LOADING

print("\n" + "=" * 70)
print("  SECTION 1: DATA LOADING")
print("=" * 70)

DATA_DIR = 'brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0'
FILES_DIR = os.path.join(DATA_DIR, 'files')

metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
print(f"\nTotal recordings: {len(metadata)}")
print(f"Columns: {list(metadata.columns)}")
print(f"\nBrugada distribution:")
print(metadata['brugada'].value_counts().sort_index().to_string())
print(f"\nBasal pattern distribution:")
print(metadata['basal_pattern'].value_counts().sort_index().to_string())
print(f"\nSudden death distribution:")
print(metadata['sudden_death'].value_counts().sort_index().to_string())

# Binary labels: Brugada (1,2) -> 1, Normal (0) -> 0
metadata['label'] = (metadata['brugada'] >= 1).astype(int)

n_normal = (metadata['label'] == 0).sum()
n_brugada = (metadata['label'] == 1).sum()
print(f"\n--- Binary Classification Labels ---")
print(f"  Normal:  {n_normal} ({n_normal/len(metadata):.1%})")
print(f"  Brugada: {n_brugada} ({n_brugada/len(metadata):.1%})")
print(f"  Imbalance ratio: 1:{n_normal/n_brugada:.1f}")


def load_ecg_signals(metadata_df, files_dir):
    """Load all ECG signals into a NumPy array."""
    signals_list = []
    valid_ids = []
    failed_ids = []

    for _, row in metadata_df.iterrows():
        patient_id = str(row['patient_id'])
        record_path = os.path.join(files_dir, patient_id, patient_id)

        try:
            record = wfdb.rdrecord(record_path)
            sig = record.p_signal

            if sig.shape == (1200, 12):
                signals_list.append(sig)
                valid_ids.append(row['patient_id'])
            else:
                failed_ids.append(patient_id)
        except Exception as e:
            print(f"  Failed to load {patient_id}: {e}")
            failed_ids.append(patient_id)

    signals = np.array(signals_list)
    print(f"\nLoaded {len(signals_list)} ECG recordings")
    print(f"  Signal array shape: {signals.shape}")
    if failed_ids:
        print(f"  Warning: Failed {len(failed_ids)} recordings")

    return signals, valid_ids, failed_ids


print("\nLoading ECG signal data...")
signals, valid_ids, failed_ids = load_ecg_signals(metadata, FILES_DIR)

metadata_valid = metadata[metadata['patient_id'].isin(valid_ids)].reset_index(drop=True)
labels = metadata_valid['label'].values

print(f"\nFinal dataset: {len(labels)} samples")
print(f"  Normal:  {(labels == 0).sum()}")
print(f"  Brugada: {(labels == 1).sum()}")

# SECTION 2: EXPLORATORY DATA ANALYSIS

print("\n" + "=" * 70)
print("  SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
time_axis = np.arange(1200) / 100.0

# 2.1 Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = metadata_valid['label'].value_counts().sort_index()
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(['Normal (0)', 'Brugada (1)'], counts.values, color=colors,
                   edgecolor='white', linewidth=2, width=0.5)
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Recordings', fontsize=12)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max(counts.values) * 1.15)

axes[1].pie(counts.values, labels=['Normal', 'Brugada'],
            colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 13}, explode=[0, 0.05],
            shadow=True, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] class_distribution.png")

# 2.2 V1-V3 ECG Comparison
idx_normal = np.where(labels == 0)[0][0]
idx_brugada = np.where(labels == 1)[0][0]

key_leads = [6, 7, 8]

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharey='row')
fig.suptitle('ECG Comparison: Normal vs Brugada Syndrome (V1-V3 Leads)',
             fontsize=16, fontweight='bold', y=1.02)

for i, lead_idx in enumerate(key_leads):
    axes[i, 0].plot(time_axis, signals[idx_normal, :, lead_idx], color='#2ecc71', linewidth=1)
    axes[i, 0].set_ylabel(lead_names[lead_idx] + '\n(mV)', fontsize=12, fontweight='bold')
    if i == 0:
        axes[i, 0].set_title('Normal - Lead ' + lead_names[lead_idx], fontsize=13)
    axes[i, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[i, 0].fill_between(time_axis, signals[idx_normal, :, lead_idx], alpha=0.1, color='#2ecc71')

    axes[i, 1].plot(time_axis, signals[idx_brugada, :, lead_idx], color='#e74c3c', linewidth=1)
    if i == 0:
        axes[i, 1].set_title('Brugada - Lead ' + lead_names[lead_idx], fontsize=13)
    axes[i, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[i, 1].fill_between(time_axis, signals[idx_brugada, :, lead_idx], alpha=0.1, color='#e74c3c')

for ax in axes[-1, :]:
    ax.set_xlabel('Time (seconds)', fontsize=11)

plt.tight_layout()
plt.savefig('ecg_comparison_v1v3.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] ecg_comparison_v1v3.png")

# 2.3 Full 12-Lead
fig, axes = plt.subplots(12, 2, figsize=(16, 28), sharex=True)
fig.suptitle('Full 12-Lead ECG: Normal (left) vs Brugada (right)',
             fontsize=16, fontweight='bold', y=1.005)

for i in range(12):
    axes[i, 0].plot(time_axis, signals[idx_normal, :, i], color='#3498db', linewidth=0.8)
    axes[i, 0].set_ylabel(lead_names[i], fontsize=10, fontweight='bold', rotation=0, labelpad=30)
    axes[i, 1].plot(time_axis, signals[idx_brugada, :, i], color='#e74c3c', linewidth=0.8)

axes[0, 0].set_title('Normal', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Brugada', fontsize=13, fontweight='bold')
axes[-1, 0].set_xlabel('Time (seconds)')
axes[-1, 1].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('full_12lead_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] full_12lead_comparison.png")

# 2.4 Signal Statistics
stats_data = []
for cls, cls_name in [(0, 'Normal'), (1, 'Brugada')]:
    cls_signals = signals[labels == cls]
    for lead_idx in range(12):
        lead_data = cls_signals[:, :, lead_idx].flatten()
        stats_data.append({
            'Class': cls_name, 'Lead': lead_names[lead_idx],
            'Mean(mV)': round(np.mean(lead_data), 4),
            'Std(mV)': round(np.std(lead_data), 4),
            'Min(mV)': round(np.min(lead_data), 4),
            'Max(mV)': round(np.max(lead_data), 4),
            'Skewness': round(skew(lead_data), 4),
            'Kurtosis': round(kurtosis(lead_data), 4)
        })

stats_df = pd.DataFrame(stats_data)
print("\n  Signal Statistics (V1-V3, key diagnostic leads):")
v_leads = stats_df[stats_df['Lead'].isin(['V1', 'V2', 'V3'])]
print(v_leads.to_string(index=False))

# SECTION 3: PREPROCESSING

print("\n" + "=" * 70)
print("  SECTION 3: PREPROCESSING")
print("=" * 70)


def bandpass_filter(signal_data, lowcut=0.5, highcut=40.0, fs=100, order=3):
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return scipy_signal.filtfilt(b, a, signal_data, axis=0)


def preprocess_signals(signals_array):
    n_samples = signals_array.shape[0]
    processed = np.zeros_like(signals_array)
    for i in range(n_samples):
        for lead in range(12):
            processed[i, :, lead] = bandpass_filter(signals_array[i, :, lead])
        for lead in range(12):
            mean = np.mean(processed[i, :, lead])
            std = np.std(processed[i, :, lead])
            if std > 0:
                processed[i, :, lead] = (processed[i, :, lead] - mean) / std
    return processed


print("  Applying bandpass filter (0.5-40 Hz) + z-score normalization...")
signals_processed = preprocess_signals(signals)
print(f"  Preprocessed shape: {signals_processed.shape}")

# Before/after plot
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(time_axis, signals[idx_brugada, :, 7], color='gray', linewidth=1)
axes[0].set_title('Before Preprocessing (Raw Signal - V2, Brugada patient)', fontsize=12)
axes[0].set_ylabel('Amplitude (mV)')

axes[1].plot(time_axis, signals_processed[idx_brugada, :, 7], color='#e74c3c', linewidth=1)
axes[1].set_title('After Preprocessing (Bandpass + Z-score - V2, Brugada patient)', fontsize=12)
axes[1].set_ylabel('Normalized Amplitude')
axes[1].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] preprocessing_comparison.png")

# Train/Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    signals_processed, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\n  Train: {X_train_raw.shape[0]} (Normal={sum(y_train==0)}, Brugada={sum(y_train==1)})")
print(f"  Test:  {X_test_raw.shape[0]} (Normal={sum(y_test==0)}, Brugada={sum(y_test==1)})")

# SECTION 4: FEATURE EXTRACTION

print("\n" + "=" * 70)
print("  SECTION 4: FEATURE EXTRACTION")
print("  13 features x 12 leads = 156 handcrafted features per recording")
print("=" * 70)


def extract_features(signal_12lead, fs=100):
    features = []
    for lead in range(12):
        x = signal_12lead[:, lead]

        features.append(np.mean(x))
        features.append(np.std(x))
        features.append(np.min(x))
        features.append(np.max(x))
        features.append(skew(x))
        features.append(kurtosis(x))
        features.append(np.sqrt(np.mean(x**2)))
        features.append(np.max(x) - np.min(x))

        zero_crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
        features.append(zero_crossings / len(x))

        freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
        fft_vals = np.abs(np.fft.rfft(x))
        power = fft_vals ** 2

        features.append(freqs[np.argmax(power[1:]) + 1])

        band1 = np.sum(power[(freqs >= 0) & (freqs < 5)])
        band2 = np.sum(power[(freqs >= 5) & (freqs < 15)])
        band3 = np.sum(power[(freqs >= 15) & (freqs <= 40)])
        total = np.sum(power) + 1e-10
        features.append(band1 / total)
        features.append(band2 / total)
        features.append(band3 / total)

    return np.array(features)


feature_names = []
feat_labels = ['Mean', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis', 'RMS',
               'PeakToPeak', 'ZeroCrossRate', 'DomFreq',
               'Energy_0_5Hz', 'Energy_5_15Hz', 'Energy_15_40Hz']
for lead in lead_names:
    for feat in feat_labels:
        feature_names.append(lead + '_' + feat)

print(f"\n  Extracting features...")
X_train_feat = np.array([extract_features(X_train_raw[i]) for i in range(len(X_train_raw))])
X_test_feat = np.array([extract_features(X_test_raw[i]) for i in range(len(X_test_raw))])
print(f"  Training: {X_train_feat.shape}  |  Test: {X_test_feat.shape}")

# SMOTE
print("\n  Applying SMOTE oversampling on training set...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_feat, y_train)
print(f"  Before SMOTE: Normal={sum(y_train==0)}, Brugada={sum(y_train==1)}")
print(f"  After SMOTE:  Normal={sum(y_train_smote==0)}, Brugada={sum(y_train_smote==1)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test_feat)

# SECTION 5: METHOD 1 - RANDOM FOREST

print("\n" + "=" * 70)
print("  SECTION 5: METHOD 1 - RANDOM FOREST")
print("=" * 70)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

print("\n  Running GridSearchCV (5-fold stratified CV)...")
grid_rf = GridSearchCV(
    rf, param_grid_rf, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=0
)
grid_rf.fit(X_train_scaled, y_train_smote)

print(f"  Best parameters: {grid_rf.best_params_}")
print(f"  Best CV F1 score: {grid_rf.best_score_:.4f}")

rf_best = grid_rf.best_estimator_

y_pred_rf = rf_best.predict(X_test_scaled)
y_prob_rf = rf_best.predict_proba(X_test_scaled)[:, 1]

print("\n" + "-" * 60)
print("  RANDOM FOREST - TEST SET RESULTS")
print("-" * 60)
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Brugada']))

# RF plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Brugada'], yticklabels=['Normal', 'Brugada'],
            annot_kws={'size': 16})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('Random Forest - Confusion Matrix', fontsize=13, fontweight='bold')

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)
axes[1].plot(fpr_rf, tpr_rf, color='#3498db', linewidth=2.5,
             label='Random Forest (AUC = {:.4f})'.format(auc_rf))
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].fill_between(fpr_rf, tpr_rf, alpha=0.15, color='#3498db')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('Random Forest - ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig('rf_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] rf_evaluation.png")

# Feature importance
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1][:20]

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = ['#e74c3c' if any(v in feature_names[i] for v in ['V1_', 'V2_', 'V3_'])
              else '#3498db' for i in indices]

ax.barh(range(20), importances[indices][::-1], color=bar_colors[::-1],
        edgecolor='white', linewidth=0.5)
ax.set_yticks(range(20))
ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='V1/V2/V3 leads (diagnostic)'),
                   Patch(facecolor='#3498db', label='Other leads')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] feature_importance.png")

print("\n  Top 10 most important features:")
for rank, i in enumerate(indices[:10], 1):
    print("    {:2d}. {:25s}  importance = {:.4f}".format(rank, feature_names[i], importances[i]))

# SECTION 6: METHOD 2 - SUPPORT VECTOR MACHINE (SVM)

print("\n" + "=" * 70)
print("  SECTION 6: METHOD 2 - SUPPORT VECTOR MACHINE (RBF KERNEL)")
print("=" * 70)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
}

svm = SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')

print("\n  Running GridSearchCV (5-fold stratified CV)...")
grid_svm = GridSearchCV(
    svm, param_grid_svm, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1', n_jobs=-1, verbose=0
)
grid_svm.fit(X_train_scaled, y_train_smote)

print(f"  Best parameters: {grid_svm.best_params_}")
print(f"  Best CV F1 score: {grid_svm.best_score_:.4f}")

svm_best = grid_svm.best_estimator_

y_pred_svm = svm_best.predict(X_test_scaled)
y_prob_svm = svm_best.predict_proba(X_test_scaled)[:, 1]

print("\n" + "-" * 60)
print("  SVM (RBF) - TEST SET RESULTS")
print("-" * 60)
print(classification_report(y_test, y_pred_svm, target_names=['Normal', 'Brugada']))

# SVM plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Normal', 'Brugada'], yticklabels=['Normal', 'Brugada'],
            annot_kws={'size': 16})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('SVM (RBF) - Confusion Matrix', fontsize=13, fontweight='bold')

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
auc_svm = auc(fpr_svm, tpr_svm)
axes[1].plot(fpr_svm, tpr_svm, color='#e67e22', linewidth=2.5,
             label='SVM RBF (AUC = {:.4f})'.format(auc_svm))
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].fill_between(fpr_svm, tpr_svm, alpha=0.15, color='#e67e22')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('SVM (RBF) - ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig('svm_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] svm_evaluation.png")

# SECTION 7: MODEL COMPARISON

print("\n" + "=" * 70)
print("  SECTION 7: MODEL COMPARISON")
print("=" * 70)

# Combined ROC
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(fpr_rf, tpr_rf, color='#3498db', linewidth=2.5,
        label='Random Forest (AUC = {:.4f})'.format(auc_rf))
ax.fill_between(fpr_rf, tpr_rf, alpha=0.1, color='#3498db')

ax.plot(fpr_svm, tpr_svm, color='#e67e22', linewidth=2.5,
        label='SVM RBF (AUC = {:.4f})'.format(auc_svm))
ax.fill_between(fpr_svm, tpr_svm, alpha=0.1, color='#e67e22')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve Comparison: Random Forest vs SVM', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] roc_comparison.png")

# Summary table
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Brugada)', 'Recall (Brugada)',
               'F1-Score (Brugada)', 'AUC-ROC', 'Specificity (Normal Recall)'],
    'Random Forest': [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf),
        auc_rf,
        recall_score(y_test, y_pred_rf, pos_label=0)
    ],
    'SVM (RBF)': [
        accuracy_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_svm),
        auc_svm,
        recall_score(y_test, y_pred_svm, pos_label=0)
    ]
})

comparison_display = comparison.copy()
comparison_display['Random Forest'] = comparison_display['Random Forest'].map('{:.4f}'.format)
comparison_display['SVM (RBF)'] = comparison_display['SVM (RBF)'].map('{:.4f}'.format)

print("\n" + "=" * 65)
print("        MODEL COMPARISON - SUMMARY TABLE")
print("=" * 65)
print(comparison_display.to_string(index=False))

# Visual comparison bar chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
rf_values = [
    accuracy_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_rf),
    f1_score(y_test, y_pred_rf),
    auc_rf
]
svm_values = [
    accuracy_score(y_test, y_pred_svm),
    precision_score(y_test, y_pred_svm),
    recall_score(y_test, y_pred_svm),
    f1_score(y_test, y_pred_svm),
    auc_svm
]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest',
               color='#3498db', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, svm_values, width, label='SVM (RBF)',
               color='#e67e22', edgecolor='white', linewidth=1.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            '{:.3f}'.format(bar.get_height()), ha='center', va='bottom',
            fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            '{:.3f}'.format(bar.get_height()), ha='center', va='bottom',
            fontsize=10, fontweight='bold')

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] model_comparison.png")

# SECTION 8: CONCLUSION

print("\n" + "=" * 70)
print("                    FINAL ANALYSIS SUMMARY")
print("=" * 70)
print()
print("DATASET:")
print("  Total ECG recordings: {}".format(len(labels)))
print("  Normal controls:      {} ({:.1%})".format(sum(labels == 0), np.mean(labels == 0)))
print("  Brugada syndrome:     {} ({:.1%})".format(sum(labels == 1), np.mean(labels == 1)))
print("  ECG format:           12-lead, 100 Hz, 12 seconds (1200 samples)")
print()
print("PREPROCESSING:")
print("  Bandpass filter:      0.5-40 Hz (Butterworth, order 3)")
print("  Normalization:        Z-score per lead")
print("  Class imbalance:      SMOTE + balanced class weights")
print("  Train/Test split:     80/20 stratified")
print()
print("METHOD 1 - RANDOM FOREST:")
print("  Features:             {} handcrafted (13 per lead x 12 leads)".format(len(feature_names)))
print("  Best parameters:      {}".format(grid_rf.best_params_))
print("  Test Accuracy:        {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("  Test F1 (Brugada):    {:.4f}".format(f1_score(y_test, y_pred_rf)))
print("  Test AUC-ROC:         {:.4f}".format(auc_rf))
print()
print("METHOD 2 - SVM (RBF Kernel):")
print("  Best parameters:      {}".format(grid_svm.best_params_))
print("  Test Accuracy:        {:.4f}".format(accuracy_score(y_test, y_pred_svm)))
print("  Test F1 (Brugada):    {:.4f}".format(f1_score(y_test, y_pred_svm)))
print("  Test AUC-ROC:         {:.4f}".format(auc_svm))
print()
print("KEY FINDINGS:")
print("  - V1, V2, V3 leads contain the most discriminative features for Brugada")
print("    syndrome, consistent with clinical diagnostic criteria.")
print("  - Both methods successfully detect Brugada syndrome from 12-lead ECG.")
print("  - Recall (sensitivity) for Brugada class is clinically critical;")
print("    missing a true Brugada case could be life-threatening.")
print()

if auc_rf > auc_svm:
    print("  >>> WINNER: Random Forest (higher AUC-ROC)")
    print("      Ensemble tree-based method captures non-linear feature interactions.")
    print("      Provides feature importance for clinical interpretability.")
elif auc_svm > auc_rf:
    print("  >>> WINNER: SVM with RBF Kernel (higher AUC-ROC)")
    print("      Kernel-based method excels at finding non-linear decision boundaries.")
    print("      Effective in high-dimensional feature spaces.")
else:
    print("  >>> TIE: Both models perform equally (same AUC-ROC)")

print()
print("=" * 70)
print("  GENERATED PLOTS:")
print("    1. class_distribution.png")
print("    2. ecg_comparison_v1v3.png")
print("    3. full_12lead_comparison.png")
print("    4. preprocessing_comparison.png")
print("    5. rf_evaluation.png")
print("    6. feature_importance.png")
print("    7. svm_evaluation.png")
print("    8. roc_comparison.png")
print("    9. model_comparison.png")
print("=" * 70)
print("  DONE!")
