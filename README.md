# 🌾 Sistem Prediksi Produktivitas Padi Menggunakan Machine Learning

## 👨‍🎓 Identitas Mahasiswa

- Nama :Najib Fadhil Akbar
- NIM : 301240019
- Mata Kuliah : Kecerdasan Buatan
- Universitas : Universitas Bale Bandung

---

# 📌 Deskripsi Proyek

Proyek ini merupakan aplikasi berbasis web yang digunakan untuk melakukan prediksi produktivitas padi menggunakan teknologi Machine Learning.

Aplikasi dibangun menggunakan framework Flask dengan tampilan modern berbasis Bootstrap 5 dan visualisasi data interaktif menggunakan Chart.js.

Sistem dapat:
- melakukan prediksi produktivitas padi,
- menampilkan grafik hasil prediksi,
- menampilkan confidence range,
- menampilkan grafik training model LSTM,
- menampilkan perbandingan model Machine Learning.

---

# 🤖 Algoritma yang Digunakan

Berikut algoritma yang digunakan pada proyek ini:

1. Linear Regression
2. LSTM (Long Short-Term Memory)
3. K-Means Clustering
4. Backpropagation Neural Network

---

# 🛠️ Teknologi yang Digunakan

- Python
- Flask
- Bootstrap 5
- Chart.js
- NumPy
- Pandas
- Scikit-Learn
- TensorFlow / Keras

---

# 📂 Struktur Folder

```plaintext
UTS KCR/
├── ann.py
├── backprogation.py
├── gabung_data.py
├── kmeans.py
├── linear_regression.py
├── lstm.py
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── linear_regression.pkl
│   └── lstm_model.h5
│
├── templates/
│   ├── home.html
│   ├── predict.html
│   ├── comparison.html
│   └── about.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── data/
    └── dataset_gabungan.csv
```

---

# ⚙️ Cara Instalasi

## 1. Clone Repository

```bash
git clone https://github.com/username/nama-repository.git
```

---

## 2. Masuk ke Folder Project

```bash
cd "UTS KCR"
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Jalankan Aplikasi

```bash
python app.py
```

---

# ▶️ Cara Menjalankan

Buka browser lalu akses:

```plaintext
http://127.0.0.1:5000
```

---

# 📊 Fitur Utama

✅ Prediksi produktivitas padi
✅ Grafik interaktif Chart.js
✅ Grafik training LSTM
✅ Confidence range prediksi
✅ Loading indicator
✅ Validasi form input
✅ Responsive Bootstrap 5 UI
✅ Halaman About dan Comparison

---

# 🌐 Link Demo Aplikasi

```plaintext
https://your-demo-app-link.com
```

---

# 📄 Link Laporan

```plaintext
https://drive.google.com/your-laporan-link
```

---

# 🎥 Link Video YouTube

```plaintext
https://youtube.com/your-video-link
```

---

# 📈 Dataset

Dataset yang digunakan berasal dari:
- Open Data Kota Tasikmalaya
- Kaggle
- Data produksi padi Indonesia

Jumlah data:
- ±1000 data produktivitas padi

---

# 📌 Kesimpulan

Aplikasi ini berhasil menerapkan metode Machine Learning untuk memprediksi produktivitas padi dengan visualisasi interaktif berbasis web sehingga dapat membantu analisis data pertanian secara lebih modern dan informatif.