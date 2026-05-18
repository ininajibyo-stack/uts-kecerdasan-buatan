import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# Load Dataset
# =========================
df = pd.read_csv('data/dataset_gabungan.csv')

# Ambil kolom yang diperlukan
df = df[['Tahun', 'Rata-rata Produksi Padi Sawah(Kw/Ha)']]

# Rename kolom
df.columns = ['tahun', 'produktivitas']

# Hapus data kosong
df.dropna(inplace=True)

print("Jumlah data:", len(df))

# =========================
# Group per Tahun
# =========================
df_group = df.groupby('tahun')['produktivitas'].mean().reset_index()

# Feature dan Target
X = df_group[['tahun']]
y = df_group['produktivitas']

# =========================
# Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Model Linear Regression
# =========================
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediksi
pred = model.predict(X_test)

# =========================
# Evaluasi Model
# =========================
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n=== HASIL LINEAR REGRESSION ===")
print("MAE :", round(mae,4))
print("MSE :", round(mse,4))
print("R2  :", round(r2,4))

# =========================
# Grafik Aktual vs Prediksi
# =========================
plt.figure(figsize=(10,6))

plt.scatter(
    X_test,
    y_test,
    label='Aktual'
)

plt.scatter(
    X_test,
    pred,
    label='Prediksi'
)

plt.title('Aktual vs Prediksi')
plt.xlabel('Tahun')
plt.ylabel('Produktivitas')

plt.legend()
plt.grid(True)

plt.show()

# =========================
# Residual Plot
# =========================
residual = y_test - pred

plt.figure(figsize=(10,6))

plt.scatter(
    pred,
    residual
)

plt.axhline(y=0)

plt.title('Residual Plot')
plt.xlabel('Prediksi')
plt.ylabel('Residual')

plt.grid(True)

plt.show()

# =========================
# Simpan Model
# =========================
joblib.dump(
    model,
    'models/linear_regression.pkl'
)

print("\nModel Linear Regression berhasil disimpan!")