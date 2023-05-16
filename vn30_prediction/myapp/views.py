import csv
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os


def get_vn30_history(request):
    data = []
    with open('VN_30_history.csv', 'r',encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Date'] = row['Date'].replace('\ufeff', '')
            row['Price'] = float(row['Price'].replace(',', ''))
            row['Open'] = float(row['Open'].replace(',', ''))
            row['High'] = float(row['High'].replace(',', ''))
            row['Low'] = float(row['Low'].replace(',', ''))
            row['Vol'] = float(row['Vol'].replace(',', '').replace('K', '')) * 1000
            row['Change'] = float(row['Change'].replace('%', ''))
            data.append(row)
    return JsonResponse(data, safe=False)

@require_GET
def predict_view(request):
    # predict_range = 30
    # algorithm = "lstm"
    n_days_lag = 5 
    dataset = read_csv('https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv', header=0, index_col=0)
    n_features = len(dataset.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = dataset.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # model = load_model('lstm.h5')
    # Get parameters from the request
    predict_range = int(request.GET.get('predict_range', 30))
    algorithm = request.GET.get('algorithm', 'lstm')

    # Load the appropriate model based on the algorithm parameter
    if algorithm == 'lstm':
        model = load_model(os.path.join(os.path.dirname(__file__), 'lstm.h5'))
    else:
        # Handle other algorithm cases or raise an error
        return JsonResponse({'error': 'Unsupported algorithm'})

    # get last n_days_lag days of data from dataset
    last_n_days = dataset.values[-n_days_lag:, :]
    # scale data using same scaler fit on training data
    last_n_days_scaled = scaler.transform(last_n_days)
    # reshape data into 3D shape for input into LSTM model
    input_data = last_n_days_scaled.reshape((1, n_days_lag, n_features))
    # generate predictions for next 30 days
    predictions = []
    for i in range(predict_range):
        # Make prediction for the next day
        yhat = model.predict(input_data)
        # Add prediction to the list of predictions
        predictions.append(yhat[0])
        # Update input_data with new prediction
        input_data = np.roll(input_data, -1)
        input_data[0, -1, :] = yhat

    # Inverse transform predictions to obtain final predicted values
    predictions = np.array(predictions)
    dummy_values = np.zeros((predictions.shape[0], n_features - 1))
    predictions_with_dummy = np.concatenate((dummy_values, predictions), axis=1)
    inv_yhat = scaler.inverse_transform(predictions_with_dummy)
    inv_yhat = inv_yhat[:, -1]

    # Define the start date
    start_date = datetime.now().date()  # Replace with your desired start date

    # Format the response with date and prediction values
    response = []
    for i in range(predict_range):
        prediction = {
            'Date': (start_date + timedelta(days=i)).strftime('%m/%d/%Y'),
            'Price': round(inv_yhat[i], 3)
        }
        response.append(prediction)

    # Return the predictions as a JSON response
    return JsonResponse(response, safe=False)