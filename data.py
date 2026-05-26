# ============================================================
# Ambil Dataset Heart Disease dari UCI & Simpan sebagai CSV
# ============================================================
# Install dulu: pip install ucimlrepo pandas tabulate
# ============================================================

from ucimlrepo import fetch_ucirepo
import pandas as pd

# 1. Ambil dataset dari UCI (id=45 = Heart Disease)
heart_disease = fetch_ucirepo(id=45)

# 2. Pisahkan fitur (X) dan target (y)
X = heart_disease.data.features   # 13 kolom fitur
y = heart_disease.data.targets    # 1 kolom target (num)

# 3. Gabungkan menjadi satu DataFrame lengkap
df = pd.concat([X, y], axis=1)

# 4. Tampilkan info ringkas
print("=" * 60)
print(f"Jumlah data : {df.shape[0]} baris")
print(f"Jumlah kolom: {df.shape[1]} kolom")
print(f"Kolom       : {list(df.columns)}")
print("=" * 60)

# 5. Tampilkan semua data dalam bentuk tabel
pd.set_option("display.max_rows", None)       # tampilkan semua baris
pd.set_option("display.max_columns", None)    # tampilkan semua kolom
pd.set_option("display.width", None)          # lebar otomatis
pd.set_option("display.float_format", "{:.2f}".format)

print("\nSELURUH DATA:\n")
print(df.to_string(index=True))

# 6. Simpan ke file CSV
output_path = "heart_disease.csv"
df.to_csv(output_path, index=False)
print(f"\nData berhasil disimpan ke: {output_path}")
print(f"Total: {len(df)} baris x {len(df.columns)} kolom")