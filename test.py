from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from flask import Flask, render_template, request
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
# Utility to get unique elements
def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)
# Globals
n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
              'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist',
              'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds',
              'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow',
              'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow',
              'thunderstorm with rain', 'thunderstorm with heavy rain', 'thunderstorm with light rain',
              'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm',
              'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle',
              'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle',
              'thunderstorm with light drizzle']
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)
app = Flask(__name__, static_url_path='')
@app.route('/')
def root():
    return render_template('home.html')
@app.route('/train')
def train():
    data = pd.read_csv('C:/Users/hp/OneDrive/Desktop/project/traffic predict/Traffic_Prediction-main/mp/static/Train.csv')
    data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)
    # Add traffic volume history features
    for n in [1, 2, 3, 4, 5, 6]:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data = data.dropna().reset_index(drop=True)
    # Convert holidays to binary
    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)
    # Extract time features
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['month_day'] = data['date_time'].dt.day
    data['weekday'] = data['date_time'].dt.weekday + 1
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    # Display unique weather types for debugging
    unique(data['weather_type'])
    unique(data['weather_description'])
    # Encode weather features
    n11 = [(n1features.index(w) + 1) if w in n1features else 0 for w in data['weather_type']]
    n22 = [(n2features.index(w) + 1) if w in n2features else 0 for w in data['weather_description']]
    data['weather_type'] = n11
    data['weather_description'] = n22
    # Features & Target
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']
    features = numeric_columns + label_columns
    target = ['traffic_volume']
    # Safely sample 10,000 rows or fewer
    sample_size = min(10000, len(data))
    data = data.sample(sample_size, random_state=1).reset_index(drop=True)
    X = data[features]
    y = data[target]
    # Scaling
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y).flatten()
    # Train model
    regr.fit(X, y)
    # Evaluate error
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
    # Debug output
    print('Predicted output :=', regr.predict(X[:10]))
    print('Actual output :=', y[:10])
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    ip = []
    ip.append(1 if request.form['isholiday'] == 'yes' else 0)
    ip.append(int(request.form['temperature']))
    ip.append(int(request.form['day']))
    ip.append(int(request.form['time'][:2]))  # hour from HH:MM
    date = request.form['date']
    ip.append(int(date[8:]))  # day
    ip.append(int(date[:4]))  # year
    ip.append(int(date[5:7]))  # month
    # Weather inputs
    s1 = request.form.get('x0')
    s2 = request.form.get('x1')
    ip.append((n1features.index(s1) + 1) if s1 in n1features else 0)
    ip.append((n2features.index(s2) + 1) if s2 in n2features else 0)
    # Scaling and prediction
    ip_scaled = x_scaler.transform([ip])
    out = regr.predict(ip_scaled)
    y_pred = y_scaler.inverse_transform([out])[0][0]
    print("Predicted Traffic Volume:", y_pred)
    # Traffic classification
    if y_pred <= 1000:
        status = "No Traffic"
    elif y_pred <= 3000:
        status = "Busy or Normal Traffic"
    elif y_pred <= 5500:
        status = "Heavy Traffic"
    else:
        status = "Worst Case"
    return render_template('output.html', data1=ip, op=y_pred, statement=status)
if __name__ == '__main__':
    app.run(debug=True)
