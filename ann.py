from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model ANN
model = load_model('models/ann_model.h5')

# Fit scaler sama seperti training
df = pd.read_csv(
    'data/produktivitas_padi_di_kota_tasikmalaya.csv'
)

df['produktivitas_padi'] = (
    df['produktivitas_padi']
    .astype(str)
    .str.replace(', TON/HEKTAR', '', regex=False)
)

df['produktivitas_padi'] = pd.to_numeric(
    df['produktivitas_padi'],
    errors='coerce'
)

df.dropna(inplace=True)

df_group = df.groupby('tahun')[
    'produktivitas_padi'
].mean().reset_index()

scaler = MinMaxScaler()
scaler.fit(df_group[['tahun']])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    tahun = float(request.form['tahun'])

    tahun_scaled = scaler.transform(
        np.array([[tahun]])
    )

    prediction = model.predict(
        tahun_scaled
    )

    hasil = round(
        float(prediction[0][0]),
        2
    )

    return render_template(
        'index.html',
        prediction=hasil,
        tahun=tahun
    )


@app.route('/comparison')
def comparison():

    metrics = {
        'ANN': 93,
        'Linear Regression': 89,
        'LSTM': 95,
        'KMeans': 84
    }

    return render_template(
        'comparison.html',
        metrics=metrics
    )


if __name__ == '__main__':
    app.run(debug=True)