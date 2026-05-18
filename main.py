from flask import Flask, render_template, request, session
import joblib
import numpy as np

app = Flask(__name__)

app.secret_key = 'secretkey'

model = joblib.load('models/linear_regression.pkl')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/hasil', methods=['POST'])
def hasil():

    try:

        tahun = float(request.form['tahun'])

        pred = model.predict(np.array([[tahun]]))[0]

        prediction = round(float(pred), 2)

        lower = round(prediction - 50, 2)
        upper = round(prediction + 50, 2)

        session['tahun'] = tahun
        session['prediction'] = prediction

        return render_template(
            'comparison.html',
            tahun=tahun,
            prediction=prediction,
            lower=lower,
            upper=upper
        )

    except:

        return render_template(
            'predict.html',
            error='Input tidak valid'
        )


@app.route('/comparison')
def comparison():

    tahun = session.get('tahun', '-')
    prediction = session.get('prediction', '-')

    lower = '-'
    upper = '-'

    if prediction != '-':
        lower = round(float(prediction) - 50, 2)
        upper = round(float(prediction) + 50, 2)

    return render_template(
        'comparison.html',
        tahun=tahun,
        prediction=prediction,
        lower=lower,
        upper=upper
    )


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)