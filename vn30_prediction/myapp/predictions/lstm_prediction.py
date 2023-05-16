import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def generate_lstm_predictions(dataset,predict_range):
    
    model_path = os.path.join(os.path.dirname(__file__), 'model/lstm.h5')
    model = load_model(model_path)

    n_days_lag = 5 
    n_features = len(dataset.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = dataset.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(values)
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

    return inv_yhat
