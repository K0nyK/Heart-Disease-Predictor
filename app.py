from flask import Flask, request, render_template, Response, session, redirect, url_for
import joblib
import numpy as np
import pandas as pd

model = joblib.load('random_forest_model.pkl')
encoder = joblib.load('encoder.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Необхідно для безпечного використання сесій


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'Sex': request.form['Sex'],
        'ChestPainType': request.form['ChestPainType'],
        'RestingECG': request.form['RestingECG'],
        'ExerciseAngina': request.form['ExerciseAngina'],
        'ST_Slope': request.form['ST_Slope'],
        'Age': request.form['Age'],
        'RestingBP': request.form['RestingBP'],
        'Cholesterol': request.form['Cholesterol'],
        'FastingBS': request.form['FastingBS'],
        'MaxHR': request.form['MaxHR'],
        'Oldpeak': request.form['Oldpeak']
    }

    # конвертація даних для моделі
    features_df = pd.DataFrame({k: [v] for k, v in data.items()})
    categorical_features = encoder.transform(
        features_df[['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
    numerical_features = np.array(features_df[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']])
    final_features = np.hstack((categorical_features, numerical_features))

    prediction = model.predict(final_features)
    output = int(prediction[0])  # Конвертація в int для забезпечення сумісності

    # Збереження результатів і даних у сесії
    session['data'] = {k: str(v) for k, v in data.items()}  # Конвертувати значення у строки
    session['result'] = output

    return redirect(url_for('results'))


@app.route('/results')
def results():
    data = session.get('data', {})
    result = session.get('result', '')
    risk_class = "low-risk" if result == 0 else "high-risk"

    results_html = "<table><tr><th>Parameter</th><th>Value</th></tr>"
    for key, value in data.items():
        results_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    results_html += f"<tr class='{risk_class}'><td>Predicted Risk of Heart Disease</td><td>{result}</td></tr>"
    results_html += "</table>"
    return render_template('results.html', results_html=results_html)


@app.route('/download')
def download():
    # Отримання даних і результату з сесії
    data = session.get('data', {})
    result = session.get('result', 'No result')

    # Додавання результату прогнозування до даних
    data['Predicted Risk of Heart Disease'] = result

    # Створення списку кортежів для побудови DataFrame
    items = list(data.items())
    df = pd.DataFrame(items, columns=['Parameter', 'Value'])

    # Генерація CSV
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                     "attachment; filename=prediction_result.csv"})


if __name__ == '__main__':
    app.run(debug=True)
