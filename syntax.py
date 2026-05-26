# ================================================================
# CardioAI — TRAINING SCRIPT FINAL
# Metode : Random Forest + GridSearchCV + SHAP TreeExplainer
# Dataset: UCI Heart Disease (303 baris, 13 fitur)
# Output : folder output_model/  (dipakai oleh app.py)
# ================================================================
# Install:
#   pip install pandas numpy scikit-learn imbalanced-learn shap
#               joblib matplotlib seaborn
# Jalankan:
#   python syntax.py
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')               # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import shap

# ── Folder output ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(BASE_DIR, 'output_model')
VIZ_DIR  = os.path.join(OUT, 'visualisasi')
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Warna & style global ─────────────────────────────────────────

PLT_STYLE = {
    'figure.facecolor':  '#0d1526',
    'axes.facecolor':    '#111d30',
    'axes.edgecolor':    '#1a2840',
    'axes.labelcolor':   '#e2eaf5',
    'xtick.color':       '#6b82a0',
    'ytick.color':       '#6b82a0',
    'text.color':        '#e2eaf5',
    'grid.color':        '#1a2840',
    'grid.alpha':        0.6,
}
plt.rcParams.update(PLT_STYLE)

COL_BLUE  = '#38bdf8'
COL_GREEN = '#34d399'
COL_PINK  = '#f472b6'
COL_WARN  = '#fbbf24'
COL_RED   = '#fb7185'

def hdr(n, j):
    print(f"\n{'='*65}\n  LANGKAH {n}: {j}\n{'='*65}")

def sub(j):
    print(f"\n  >> {j}\n  {'-'*50}")

def save_fig(name, dpi=150):
    path = os.path.join(VIZ_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f"  ✓ Visualisasi disimpan: visualisasi/{name}")
    return path


# ================================================================
# LANGKAH 1 — BACA DATASET
# ================================================================
hdr(1, "MEMBACA DATASET")

FILE_PATH = os.path.join(BASE_DIR, 'heart_disease.csv')
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {FILE_PATH}\nJalankan data.py dulu.")

df = pd.read_csv(FILE_PATH)
print(f"  ✓ {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"  ✓ Kolom: {list(df.columns)}")


# ================================================================
# LANGKAH 2 — PREPROCESSING
# ================================================================
hdr(2, "PREPROCESSING DATA")

df_clean = df.copy()

# Binarisasi target
df_clean['target'] = (df_clean['num'] > 0).astype(int)
df_clean = df_clean.drop(columns=['num'])
sehat = (df_clean['target'] == 0).sum()
sakit = (df_clean['target'] == 1).sum()
print(f"  ✓ Sehat (0): {sehat}  |  Sakit (1): {sakit}")

# Encoding ordinal — tetap gunakan 13 fitur asli (tanpa OHE)
# cp   : 1-4  → 0-3
if df_clean['cp'].max() == 4:
    df_clean['cp'] = df_clean['cp'] - 1
# thal : 3/6/7 → 0/1/2
thal_map = {3.0: 0, 6.0: 1, 7.0: 2}
df_clean['thal'] = df_clean['thal'].map(thal_map)
# slope: 1-3  → 0-2
if df_clean['slope'].min() == 1:
    df_clean['slope'] = df_clean['slope'] - 1

# Gunakan 13 fitur asli — TANPA one-hot encoding
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X_raw = df_clean[feature_cols].astype(float)
y     = df_clean['target']
print(f"  ✓ Jumlah fitur  : {len(feature_cols)} (13 fitur asli — tanpa OHE)")
print(f"  ✓ Daftar fitur  : {feature_cols}")

# Imputasi median
imputer   = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)
print(f"  ✓ Missing setelah imputasi: {X_imputed.isnull().sum().sum()}")

# Train/test split stratified 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y)
print(f"  ✓ Train: {len(X_train)}  |  Test: {len(X_test)}")

# SMOTE hanya pada data training
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"  ✓ Setelah SMOTE — Train: {len(X_res)} "
      f"(Sehat:{(y_res==0).sum()} Sakit:{(y_res==1).sum()})")

# StandardScaler — fit HANYA pada train
scaler     = StandardScaler()
X_res_sc   = scaler.fit_transform(X_res)
X_test_sc  = scaler.transform(X_test)
X_train_sc = scaler.transform(X_train)   # utk SHAP partial


# ================================================================
# LANGKAH 3 — VIZ: DISTRIBUSI DATA
# ================================================================
hdr(3, "VISUALISASI DISTRIBUSI DATA")

# ── 3.1 Target & distribusi fitur utama ─────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Distribusi Fitur UCI Heart Disease', fontsize=14,
             color=COL_BLUE, fontweight='bold', y=1.01)

# Target
ax = axes[0, 0]
bars = ax.bar(['Sehat (0)', 'Sakit (1)'], [sehat, sakit],
              color=[COL_GREEN, COL_RED], width=0.5, edgecolor='none')
ax.set_title('Distribusi Target', color=COL_BLUE, fontsize=11)
ax.bar_label(bars, fmt='%d', color='white', fontsize=10)
ax.set_ylim(0, max(sehat, sakit) * 1.2)

# Fitur numerik
num_feats = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
colors    = [COL_BLUE, COL_PINK, COL_WARN, COL_GREEN, COL_RED, COL_BLUE]
for i, (feat, col) in enumerate(zip(num_feats, colors)):
    row, c = divmod(i + 1, 4)
    ax = axes[row, c]
    df_clean[df_clean['target']==0][feat].dropna().hist(
        ax=ax, bins=15, alpha=0.7, color=COL_GREEN, label='Sehat', density=True)
    df_clean[df_clean['target']==1][feat].dropna().hist(
        ax=ax, bins=15, alpha=0.7, color=COL_RED, label='Sakit', density=True)
    ax.set_title(feat, color=COL_BLUE, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig('01_distribusi_data.png')

# ── 3.2 Correlation heatmap — semua 13 fitur ────────────────────
num_df = df_clean[feature_cols + ['target']].copy()
corr   = num_df.corr()

fig, ax = plt.subplots(figsize=(13, 11))
mask    = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap_custom = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap_custom,
            center=0, square=True, linewidths=0.5,
            annot_kws={'size': 9, 'color': 'white'},
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Korelasi Antar Fitur Numerik', color=COL_BLUE,
             fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
save_fig('02_correlation_heatmap.png')


# ================================================================
# LANGKAH 4 — TUNING RANDOM FOREST
# ================================================================
hdr(4, "HYPERPARAMETER TUNING — GridSearchCV")

param_grid = {
    'n_estimators'    : [300, 500],
    'max_depth'       : [6, 8, None],
    'min_samples_leaf': [1, 2],
    'max_features'    : ['sqrt', 'log2'],
    'class_weight'    : ['balanced'],
}

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=cv_inner,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0,
    refit=True
)
grid_search.fit(X_res_sc, y_res)
best_params = grid_search.best_params_
best_cv_auc = grid_search.best_score_

print(f"\n  Best CV AUC  : {best_cv_auc:.4f}")
print(f"  Best Params  :")
for k, v in best_params.items():
    print(f"    {k:20s}: {v}")


# ================================================================
# LANGKAH 5 — LATIH MODEL FINAL
# ================================================================
hdr(5, "LATIH RANDOM FOREST FINAL")

rf_model = grid_search.best_estimator_
print(f"  ✓ Model dilatih dengan best params dari GridSearch")

# Cross-validation 5-fold pada seluruh data (hanya untuk info)
X_all_imp = pd.DataFrame(imputer.transform(X_raw), columns=feature_cols)
X_all_sc  = scaler.fit_transform(X_all_imp)
rf_cv    = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf_cv, X_all_sc, y, cv=5, scoring='roc_auc')
print(f"  ✓ CV 5-fold AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ================================================================
# LANGKAH 6 — EVALUASI
# ================================================================
hdr(6, "EVALUASI MODEL")

# Cari threshold optimal (Youden's J)
y_prob = rf_model.predict_proba(X_test_sc)[:, 1]
fpr_a, tpr_a, thresholds_a = roc_curve(y_test, y_prob)
j_scores = tpr_a - fpr_a
best_idx = np.argmax(j_scores)
best_thr = float(thresholds_a[best_idx])

y_pred = (y_prob >= best_thr).astype(int)
cm_arr = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm_arr[0][0], cm_arr[0][1], cm_arr[1][0], cm_arr[1][1]

acc_v  = accuracy_score(y_test, y_pred)
prec_v = precision_score(y_test, y_pred, zero_division=0)
rec_v  = recall_score(y_test, y_pred, zero_division=0)
f1_v   = f1_score(y_test, y_pred, zero_division=0)
auc_v  = roc_auc_score(y_test, y_prob)

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Accuracy  : {acc_v:.4f}  ({acc_v*100:.2f}%)                  │
  │  Precision : {prec_v:.4f}                              │
  │  Recall    : {rec_v:.4f}                              │
  │  F1-Score  : {f1_v:.4f}                              │
  │  AUC-ROC   : {auc_v:.4f}                              │
  │  Threshold : {best_thr:.4f}  (Youden's J)              │
  │  TN={tn:3d}  FP={fp:3d}  FN={fn:3d}  TP={tp:3d}             │
  └─────────────────────────────────────────────────────┘
""")
print(classification_report(y_test, y_pred, target_names=['Sehat', 'Sakit']))


# ================================================================
# LANGKAH 7 — VISUALISASI RANDOM FOREST
# ================================================================
hdr(7, "VISUALISASI RANDOM FOREST")

# ── 7.1 Confusion Matrix heatmap ────────────────────────────────
sub("7.1 Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 5))
cm_disp = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_disp, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Prediksi Sehat', 'Prediksi Sakit'], fontsize=11)
ax.set_yticklabels(['Aktual Sehat', 'Aktual Sakit'], fontsize=11)
ax.set_title(f'Confusion Matrix — RF\nAcc={acc_v*100:.1f}%  AUC={auc_v:.4f}',
             color=COL_BLUE, fontsize=13, fontweight='bold', pad=12)
labels = [['TN\n'+str(tn), 'FP\n'+str(fp)],
          ['FN\n'+str(fn), 'TP\n'+str(tp)]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i][j], ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if cm_disp[i,j] > cm_disp.max()/2 else COL_BLUE)
plt.tight_layout()
save_fig('03_confusion_matrix.png')

# ── 7.2 ROC Curve ───────────────────────────────────────────────
sub("7.2 ROC Curve")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_a, tpr_a, color=COL_BLUE, lw=2.5,
        label=f'Random Forest (AUC = {auc_v:.4f})')
ax.fill_between(fpr_a, tpr_a, alpha=0.1, color=COL_BLUE)
ax.plot([0,1],[0,1], color=COL_WARN, lw=1, linestyle='--', label='Random Classifier')
ax.scatter([fpr_a[best_idx]], [tpr_a[best_idx]], color=COL_RED, s=100, zorder=5,
           label=f'Threshold Optimal = {best_thr:.3f}')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Kurva ROC — Random Forest', color=COL_BLUE, fontsize=13,
             fontweight='bold', pad=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('04_roc_curve.png')

# ── 7.3 Feature Importance (bawaan RF) ──────────────────────────
sub("7.3 Feature Importance (Gini Impurity)")
fi = rf_model.feature_importances_
fi_df = pd.DataFrame({'Fitur': feature_cols, 'Importance': fi})
fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
colors_bar = [COL_BLUE if i < len(fi_df)-5 else COL_GREEN
              for i in range(len(fi_df))]
bars = ax.barh(fi_df['Fitur'], fi_df['Importance'],
               color=colors_bar, edgecolor='none', height=0.65)
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Feature Importance — Random Forest\n(Gini Impurity, semua pohon)',
             color=COL_BLUE, fontsize=13, fontweight='bold', pad=12)
for bar, val in zip(bars, fi_df['Importance']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, color='#e2eaf5')
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, fi_df['Importance'].max() * 1.18)
plt.tight_layout()
save_fig('05_feature_importance_rf.png')

# ── 7.4 Cross-Validation score per fold ─────────────────────────
sub("7.4 Cross-Validation 5-Fold")
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars  = ax.bar(folds, cv_scores, color=COL_BLUE, width=0.5, edgecolor='none')
ax.axhline(cv_scores.mean(), color=COL_RED, lw=2, linestyle='--',
           label=f'Mean AUC = {cv_scores.mean():.4f}')
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Cross-Validation 5-Fold AUC',
             color=COL_BLUE, fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11)
ax.bar_label(bars, fmt='%.4f', color='white', fontsize=10, padding=3)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
save_fig('06_cross_validation.png')

# ================================================================
# LANGKAH 8 — SHAP EXPLAINABILITY
# ================================================================
hdr(8, "SHAP — EXPLAINABILITY (TreeExplainer)")

sub("8.1 Hitung SHAP Values")

explainer = shap.TreeExplainer(rf_model)

# =========================
# SHAP VALUES
# =========================
sv_raw = explainer.shap_values(X_test_sc)

# Untuk RandomForest binary classifier
if isinstance(sv_raw, list):
    sv_sakit = sv_raw[1]
else:
    if len(np.array(sv_raw).shape) == 3:
        sv_sakit = sv_raw[:, :, 1]
    else:
        sv_sakit = sv_raw

# Pastikan array numpy
sv_sakit = np.array(sv_sakit)

# =========================
# EXPECTED VALUE
# =========================
exp_val = explainer.expected_value

if isinstance(exp_val, list):
    base_value = exp_val[1]
elif isinstance(exp_val, np.ndarray):
    if len(exp_val.shape) > 0:
        base_value = exp_val[1]
    else:
        base_value = float(exp_val)
else:
    base_value = exp_val

# =========================
# SHAP IMPORTANCE
# =========================
mean_shap = np.abs(sv_sakit).mean(axis=0)

shap_df = pd.DataFrame({
    'Fitur': feature_cols,
    'SHAP_mean_abs': mean_shap
}).sort_values('SHAP_mean_abs', ascending=False).reset_index(drop=True)

print("\n  RANKING SHAP FEATURE:")
for i, row in shap_df.iterrows():
    print(f"  {i+1:2d}. {row['Fitur']:15s} : {row['SHAP_mean_abs']:.5f}")

# ================================================================
# 8.2 SHAP BAR PLOT
# ================================================================
sub("8.2 SHAP Bar Plot")

fig, ax = plt.subplots(figsize=(10,7))

top_shap = shap_df.head(15).sort_values(
    'SHAP_mean_abs',
    ascending=True
)

bars = ax.barh(
    top_shap['Fitur'],
    top_shap['SHAP_mean_abs'],
    color=COL_BLUE
)

ax.set_title(
    'SHAP Global Feature Importance',
    color=COL_BLUE,
    fontsize=13,
    fontweight='bold'
)

ax.set_xlabel("Mean |SHAP Value|")

for bar, val in zip(bars, top_shap['SHAP_mean_abs']):
    ax.text(
        val + 0.0005,
        bar.get_y() + bar.get_height()/2,
        f'{val:.4f}',
        va='center',
        fontsize=9
    )

plt.tight_layout()
save_fig('07_shap_bar_global.png')

# ================================================================
# 8.3 SHAP SUMMARY
# ================================================================
sub("8.3 SHAP Summary Plot")

plt.figure(figsize=(11,8))

shap.summary_plot(
    sv_sakit,
    X_test_sc,
    feature_names=feature_cols,
    show=False,
    max_display=15
)

plt.gcf().set_facecolor('#0d1526')

save_fig('08_shap_beeswarm.png')

# ================================================================
# 8.4 WATERFALL
# ================================================================
sub("8.4 SHAP Waterfall")

high_idx = int(np.argmax(y_prob))

exp = shap.Explanation(
    values=sv_sakit[high_idx],
    base_values=base_value,
    data=X_test_sc[high_idx],
    feature_names=feature_cols
)

plt.figure(figsize=(10,7))

shap.waterfall_plot(
    exp,
    max_display=12,
    show=False
)

save_fig('09_shap_waterfall_high_risk.png')

# ================================================================
# 8.5 DEPENDENCY PLOT
# ================================================================
sub("8.5 SHAP Dependency Plot")

top4 = shap_df['Fitur'].head(4).tolist()

feat_map = {
    f:i for i,f in enumerate(feature_cols)
}

fig, axes = plt.subplots(2,2, figsize=(13,10))

for ax, feat in zip(axes.flatten(), top4):

    idx = feat_map[feat]

    sc = ax.scatter(
        X_test_sc[:, idx],
        sv_sakit[:, idx],
        c=X_test_sc[:, idx],
        cmap='coolwarm',
        alpha=0.8
    )

    ax.set_title(feat)

    ax.set_xlabel(f'{feat} Value')
    ax.set_ylabel('SHAP Value')

    plt.colorbar(sc, ax=ax)

plt.tight_layout()

save_fig('10_shap_dependency_top4.png')

# ================================================================
# 8.6 FORCE PLOT
# ================================================================
sub("8.6 SHAP Force Plot")

low_idx = int(np.argmin(y_prob))

for label_fp, idx_fp in [
    ('high_risk', high_idx),
    ('low_risk', low_idx)
]:

    force_plot = shap.force_plot(
        base_value,
        sv_sakit[idx_fp],
        X_test_sc[idx_fp],
        feature_names=feature_cols,
        matplotlib=False
    )

    html_path = os.path.join(
        VIZ_DIR,
        f'11_shap_force_{label_fp}.html'
    )

    shap.save_html(html_path, force_plot)

    print(f"  ✓ Saved: {html_path}")

# ================================================================
# 8.7 POSITIVE VS NEGATIVE
# ================================================================
sub("8.7 SHAP Positif vs Negatif")

mean_pos = np.where(
    sv_sakit > 0,
    sv_sakit,
    0
).mean(axis=0)

mean_neg = np.where(
    sv_sakit < 0,
    sv_sakit,
    0
).mean(axis=0)

fig, ax = plt.subplots(figsize=(11,7))

y_axis = np.arange(len(feature_cols))

ax.barh(
    y_axis,
    mean_pos,
    color=COL_RED,
    alpha=0.8,
    label='Positif'
)

ax.barh(
    y_axis,
    mean_neg,
    color=COL_BLUE,
    alpha=0.8,
    label='Negatif'
)

ax.set_yticks(y_axis)
ax.set_yticklabels(feature_cols)

ax.axvline(0, color='white')

ax.set_title(
    'SHAP Positive vs Negative Impact',
    color=COL_BLUE,
    fontsize=13,
    fontweight='bold'
)

ax.legend()

plt.tight_layout()

save_fig('12_shap_pos_neg.png')

# ================================================================
# LANGKAH 9 — SIMPAN MODEL & ARTEFAK
# ================================================================
hdr(9, "MENYIMPAN MODEL DAN ARTEFAK")

joblib.dump(rf_model,    os.path.join(OUT, 'rf_model.pkl'));         print("  ✓ rf_model.pkl")
joblib.dump(imputer,     os.path.join(OUT, 'imputer.pkl'));          print("  ✓ imputer.pkl")
joblib.dump(scaler,      os.path.join(OUT, 'scaler.pkl'));           print("  ✓ scaler.pkl")
joblib.dump(feature_cols,os.path.join(OUT, 'feature_cols.pkl'));     print("  ✓ feature_cols.pkl")
joblib.dump(best_thr,    os.path.join(OUT, 'threshold.pkl'));        print(f"  ✓ threshold.pkl ({best_thr:.4f})")

shap_df.to_csv(os.path.join(OUT, 'shap_importance.csv'), index=False)
print("  ✓ shap_importance.csv")

hasil_df = pd.DataFrame({
    'Probabilitas_Sakit': y_prob.round(4),
    'Prediksi'          : y_pred,
    'Aktual'            : y_test.values
})
hasil_df.to_csv(os.path.join(OUT, 'hasil_prediksi_test.csv'), index=False)
print("  ✓ hasil_prediksi_test.csv")

metrik = {
    'accuracy' : round(acc_v,  4), 'precision': round(prec_v, 4),
    'recall'   : round(rec_v,  4), 'f1'       : round(f1_v,   4),
    'auc_roc'  : round(auc_v,  4), 'threshold': round(best_thr, 4),
    'total_test': len(y_test),
    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    'best_params' : best_params,
    'cv_auc_mean' : round(float(cv_scores.mean()), 4),
    'cv_auc_std'  : round(float(cv_scores.std()),  4),
    'model'       : 'RandomForest + GridSearchCV + SHAP',
    'n_features'  : len(feature_cols),
    'feature_cols': feature_cols
}
with open(os.path.join(OUT, 'metrik.json'), 'w') as f:
    json.dump(metrik, f, indent=2)
print("  ✓ metrik.json")

# Daftar visualisasi
viz_list = sorted(os.listdir(VIZ_DIR))
with open(os.path.join(OUT, 'viz_list.json'), 'w') as f:
    json.dump(viz_list, f, indent=2)
print(f"  ✓ viz_list.json  ({len(viz_list)} file visualisasi)")


# ================================================================
# SELESAI
# ================================================================
print(f"""
\n{'='*65}
  TRAINING SELESAI — RANDOM FOREST + SHAP
{'='*65}
  ┌─────────────────────────────────────────────────────┐
  │  Accuracy  : {acc_v*100:.2f}%                                │
  │  AUC-ROC   : {auc_v:.4f}                                     │
  │  Recall    : {rec_v:.4f}                                     │
  │  F1-Score  : {f1_v:.4f}                                      │
  │  Threshold : {best_thr:.4f}  (Youden's J)                    │
  │  CV AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  │
  │  TN={tn:3d}  FP={fp:3d}  FN={fn:3d}  TP={tp:3d}              │
  └─────────────────────────────────────────────────────┘

  Visualisasi ({len(viz_list)} file) → ./output_model/visualisasi/
  Selanjutnya : python app.py
  Buka browser: http://localhost:5000
""")
