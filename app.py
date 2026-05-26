# ================================================================
# BACKEND FLASK — CardioAI Random Forest
# ================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import joblib
import shap

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================================================================
# PATH
# ================================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(BASE_DIR, 'output_model')

MODEL_PATH     = os.path.join(MODEL_DIR, 'rf_model.pkl')
IMPUTER_PATH   = os.path.join(MODEL_DIR, 'imputer.pkl')
SCALER_PATH    = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURES_PATH  = os.path.join(MODEL_DIR, 'feature_cols.pkl')
THRESHOLD_PATH = os.path.join(MODEL_DIR, 'threshold.pkl')
SHAP_CSV_PATH  = os.path.join(MODEL_DIR, 'shap_importance.csv')
METRIK_PATH    = os.path.join(MODEL_DIR, 'metrik.json')
HASIL_CSV_PATH = os.path.join(MODEL_DIR, 'hasil_prediksi_test.csv')

# ================================================================
# GLOBAL VARIABLE
# ================================================================

model          = None
imputer        = None
scaler         = None
feature_cols   = None
explainer      = None
MODEL_LOADED   = False
THRESHOLD      = 0.5

# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    global model, imputer, scaler, feature_cols
    global explainer, MODEL_LOADED, THRESHOLD

    print("\n" + "="*60)
    print(" MEMUAT RANDOM FOREST MODEL")
    print("="*60)

    try:
        model        = joblib.load(MODEL_PATH)
        imputer      = joblib.load(IMPUTER_PATH)
        scaler       = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURES_PATH)

        if os.path.exists(THRESHOLD_PATH):
            THRESHOLD = float(joblib.load(THRESHOLD_PATH))

        explainer = shap.TreeExplainer(model)

        MODEL_LOADED = True

        print("✓ Random Forest loaded")
        print("✓ Imputer loaded")
        print("✓ Scaler loaded")
        print("✓ Feature columns loaded:", feature_cols)
        print(f"✓ Threshold: {THRESHOLD:.4f}")
        print("✓ SHAP TreeExplainer loaded")

        return True

    except Exception as e:
        print(f"\n✗ ERROR LOAD MODEL: {e}")
        MODEL_LOADED = False
        return False


# ================================================================
# HELPER — One-Hot Encode input sesuai feature_cols dari syntax.py
#
# syntax.py melakukan:
#   CAT_COLS = ['cp', 'thal', 'slope']
#   df_ohe = pd.get_dummies(df_clean, columns=CAT_COLS, drop_first=False)
#
# Sehingga kolom yang dihasilkan adalah:
#   cp_0, cp_1, cp_2, cp_3
#   thal_0, thal_1, thal_2
#   slope_0, slope_1, slope_2
#
# Sebelum OHE, syntax.py juga menggeser nilai:
#   cp   : dikurangi 1 jika max == 4  → nilai 1-4 menjadi 0-3
#   thal : di-map {3.0:0, 6.0:1, 7.0:2}
#          NAMUN input dari HTML sudah 0/1/2, jadi tidak perlu di-map ulang
#   slope: dikurangi 1 jika min == 1  → nilai 1-3 menjadi 0-2
#          Input dari HTML sudah 0/1/2, jadi tidak perlu di-shift
# ================================================================

def preprocess_input(raw: dict) -> np.ndarray:
    """
    Terima dict dengan 13 fitur mentah dari form HTML,
    kembalikan array numpy siap dimasukkan ke model.
    """

    # ── Fitur numerik (langsung dipakai) ─────────────────────────
    row = {
        'age'     : float(raw.get('age', 0)),
        'sex'     : float(raw.get('sex', 0)),
        'trestbps': float(raw.get('trestbps', 0)),
        'chol'    : float(raw.get('chol', 0)),
        'fbs'     : float(raw.get('fbs', 0)),
        'restecg' : float(raw.get('restecg', 0)),
        'thalach' : float(raw.get('thalach', 0)),
        'exang'   : float(raw.get('exang', 0)),
        'oldpeak' : float(raw.get('oldpeak', 0)),
        'ca'      : float(raw.get('ca', 0)),
    }

    # ── One-Hot Encode: cp (0–3) ──────────────────────────────────
    cp_val = int(raw.get('cp', 0))
    for i in range(4):
        row[f'cp_{i}'] = 1.0 if cp_val == i else 0.0

    # ── One-Hot Encode: thal (0–2) ────────────────────────────────
    # HTML kirim 0=Normal, 1=Fixed Defect, 2=Reversible Defect
    # sudah sesuai dengan hasil thal_map di syntax.py
    thal_val = int(raw.get('thal', 0))
    for i in range(3):
        row[f'thal_{i}'] = 1.0 if thal_val == i else 0.0

    # ── One-Hot Encode: slope (0–2) ───────────────────────────────
    # HTML kirim 0=Upsloping, 1=Flat, 2=Downsloping
    # sudah sesuai, tidak perlu shift
    slope_val = int(raw.get('slope', 0))
    for i in range(3):
        row[f'slope_{i}'] = 1.0 if slope_val == i else 0.0

    # ── Susun DataFrame sesuai urutan feature_cols ───────────────
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols, fill_value=0.0)

    # ── Imputasi & Scaling ────────────────────────────────────────
    arr_imputed = imputer.transform(df)
    arr_scaled  = scaler.transform(arr_imputed)

    return arr_scaled


# ================================================================
# HELPER — Buat teks rekomendasi klinis
# ================================================================

def buat_rekomendasi(kelas: str) -> str:
    if kelas == 'rendah':
        return (
            "✓ <strong>Risiko Rendah</strong> — kondisi jantung tampak baik.<br>"
            "• Pertahankan gaya hidup sehat dan aktif berolahraga<br>"
            "• Kontrol tekanan darah & kolesterol rutin setiap tahun<br>"
            "• Hindari rokok dan batasi konsumsi alkohol"
        )
    elif kelas == 'sedang':
        return (
            "⚠ <strong>Risiko Sedang</strong> — perlu perhatian medis.<br>"
            "• Segera konsultasi ke dokter spesialis jantung<br>"
            "• Lakukan pemeriksaan EKG dan stress test<br>"
            "• Kontrol kesehatan setiap 3 bulan"
        )
    else:
        return (
            "🚨 <strong>Risiko TINGGI</strong> — tindakan segera diperlukan!<br>"
            "• <strong>Segera rujuk ke kardiolog</strong> dalam 24–48 jam<br>"
            "• Pertimbangkan angiografi koroner<br>"
            "• <strong>Jangan tunda penanganan</strong>"
        )


# ================================================================
# ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template('indeks.html')


# ── /api/status ──────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def api_status():
    """
    Dipakai oleh indeks.html → cekServer()
    Response yang diharapkan HTML:
      { model_loaded, selected_features, explainer_type }
    """
    if not MODEL_LOADED:
        return jsonify({
            'model_loaded'     : False,
            'message'          : 'Model belum dimuat'
        })

    # Baca metrik jika ada
    auc = None
    if os.path.exists(METRIK_PATH):
        with open(METRIK_PATH) as f:
            m = json.load(f)
        auc = m.get('auc_roc')

    return jsonify({
        'model_loaded'     : True,
        'model_type'       : 'Random Forest',
        'explainer_type'   : 'SHAP TreeExplainer',
        'selected_features': feature_cols,
        'n_features'       : len(feature_cols),
        'threshold'        : round(THRESHOLD, 4),
        'auc_roc'          : auc,
    })


# ── /api/shap-global ─────────────────────────────────────────────
@app.route('/api/shap-global', methods=['GET'])
def api_shap_global():
    """
    Dipakai oleh indeks.html → muatSHAPGlobal()
    Baca shap_importance.csv dari output_model/
    """
    if not os.path.exists(SHAP_CSV_PATH):
        return jsonify({'error': True, 'message': 'shap_importance.csv tidak ditemukan'}), 404

    df = pd.read_csv(SHAP_CSV_PATH)

    # Kolom di CSV: Fitur, SHAP_mean_abs
    result = []
    for _, row in df.iterrows():
        result.append({
            'fitur'    : row['Fitur'],
            'shap_mean': round(float(row['SHAP_mean_abs']), 5),
        })

    return jsonify({'error': False, 'data': result})


# ── /api/evaluasi ────────────────────────────────────────────────
@app.route('/api/evaluasi', methods=['GET'])
def api_evaluasi():
    """
    Dipakai oleh indeks.html → muatEvaluasi()
    Baca metrik.json dan hasil_prediksi_test.csv
    """
    if not os.path.exists(METRIK_PATH):
        return jsonify({'error': True, 'message': 'metrik.json tidak ditemukan'}), 404

    with open(METRIK_PATH) as f:
        m = json.load(f)

    # Hitung per-kelas dari metrik (sklearn classification_report-like)
    # Dari metrik.json kita punya: tn, fp, fn, tp, precision, recall, f1
    tn  = m.get('tn', 0)
    fp  = m.get('fp', 0)
    fn  = m.get('fn', 0)
    tp  = m.get('tp', 0)

    total_test = m.get('total_test', tn + fp + fn + tp)

    # Per-kelas precision / recall / f1
    prec_sakit = m.get('precision', 0)
    rec_sakit  = m.get('recall',    0)
    f1_sakit   = m.get('f1',        0)

    # Sehat = inverse
    prec_sehat = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_sehat  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_sehat   = (2 * prec_sehat * rec_sehat / (prec_sehat + rec_sehat)
                  if (prec_sehat + rec_sehat) > 0 else 0)

    support_sehat = tn + fp
    support_sakit = fn + tp

    def pct_str(v):
        return f"{round(v * 100, 1)}%"

    return jsonify({
        'error'         : False,
        'accuracy'      : m.get('accuracy', 0),
        'auc_roc'       : m.get('auc_roc',  0),
        'threshold'     : m.get('threshold', THRESHOLD),
        'total_test'    : total_test,
        'cv_auc_mean'   : m.get('cv_auc_mean'),
        'cv_auc_std'    : m.get('cv_auc_std'),
        'best_params'   : m.get('best_params'),
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        },
        'pct': {
            'accuracy' : f"{round(m.get('accuracy', 0) * 100, 1)}%",
            'precision': pct_str(m.get('precision', 0)),
            'recall'   : pct_str(m.get('recall',    0)),
            'f1'       : pct_str(m.get('f1',        0)),
            'auc_roc'  : f"{round(m.get('auc_roc', 0) * 100, 1)}%",
        },
        'per_kelas': {
            'sehat': {
                'precision': pct_str(prec_sehat),
                'recall'   : pct_str(rec_sehat),
                'f1'       : pct_str(f1_sehat),
                'sup'      : support_sehat,
            },
            'sakit': {
                'precision': pct_str(prec_sakit),
                'recall'   : pct_str(rec_sakit),
                'f1'       : pct_str(f1_sakit),
                'sup'      : support_sakit,
            },
        }
    })


# ── /api/predict ─────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint prediksi utama.
    Input  : JSON dengan 13 fitur mentah (sama seperti form HTML)
    Output : probabilitas, kategori, top_shap, rekomendasi
    """
    if not MODEL_LOADED:
        return jsonify({'error': True, 'message': 'Model belum dimuat'}), 500

    try:
        raw = request.get_json()

        # ── Preprocessing (OHE + impute + scale) ─────────────────
        input_scaled = preprocess_input(raw)

        # ── Prediksi ──────────────────────────────────────────────
        prob  = float(model.predict_proba(input_scaled)[0][1])
        persen = round(prob * 100, 2)

        # ── Label risiko ──────────────────────────────────────────
        if prob < THRESHOLD * 0.6:
            kategori = "RISIKO RENDAH"
            kelas    = "rendah"
            simbol   = "🟢"
        elif prob < THRESHOLD:
            kategori = "RISIKO SEDANG"
            kelas    = "sedang"
            simbol   = "🟡"
        else:
            kategori = "RISIKO TINGGI"
            kelas    = "tinggi"
            simbol   = "🔴"

        # ── SHAP Values ───────────────────────────────────────────
        shap_values = explainer.shap_values(input_scaled)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            arr = np.array(shap_values)
            if len(arr.shape) == 3:
                sv = arr[0, :, 1]
            else:
                sv = arr[0]

        kontribusi = sorted(
            zip(feature_cols, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_shap = []
        for f, v in kontribusi:
            top_shap.append({
                'fitur': f,
                'nilai': round(float(v), 4),
                'arah' : 'pos' if v >= 0 else 'neg',
                'label': '↑ Meningkatkan Risiko' if v >= 0 else '↓ Menurunkan Risiko'
            })

        # ── Rekomendasi klinis ────────────────────────────────────
        rekomendasi = buat_rekomendasi(kelas)

        return jsonify({
            'error'       : False,
            'probabilitas': prob,
            'persen'      : persen,
            'kategori'    : kategori,
            'kelas'       : kelas,
            'simbol'      : simbol,
            'threshold'   : round(THRESHOLD, 4),
            'top_shap'    : top_shap,
            'rekomendasi' : rekomendasi,
            'detail'      : {
                'model'       : 'Random Forest + GridSearchCV',
                'jumlah_fitur': len(feature_cols),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': True, 'message': str(e)}), 500


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)