import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# Load dataset
# =========================
df = pd.read_csv("data/dataset_gabungan.csv")

# pilih kolom yang dipakai
df = df[['Tahun', 'Rata-rata Produksi Padi Sawah(Kw/Ha)']]

# rename supaya mudah
df.columns = ['tahun', 'produktivitas']

# ubah ke numeric
df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
df['produktivitas'] = pd.to_numeric(df['produktivitas'], errors='coerce')

# hapus data kosong
df.dropna(inplace=True)

print("Jumlah data:", len(df))

# =========================
# Group per tahun
# =========================
df_group = df.groupby('tahun')['produktivitas'].mean().reset_index()

X = df_group[['tahun']].values
y = df_group['produktivitas'].values

# =========================
# Normalisasi
# =========================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# reshape untuk LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# =========================
# Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Model LSTM
# =========================
model = Sequential()

model.add(
    LSTM(
        50,
        activation='relu',
        input_shape=(1,1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

# =========================
# Training
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# =========================
# Prediksi
# =========================
pred = model.predict(X_test)

# =========================
# Evaluasi
# =========================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n=== HASIL LSTM ===")
print("MAE :", round(mae,4))
print("RMSE:", round(rmse,4))
print("R2  :", round(r2,4))

# =========================
# Grafik Training
# =========================
plt.figure(figsize=(10,6))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training Loss LSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])
plt.grid(True)

plt.show()

# =========================
# Save model
# =========================
os.makedirs("models", exist_ok=True)

model.save("models/lstm_model.h5")

print("\nModel LSTM berhasil disimpan!")