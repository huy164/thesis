import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the data
url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
df = pd.read_csv(url)

# Preprocess the data
data = df['VN30'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Create the input sequence for the LSTM model
seq_length = 5
last_sequence = data[-seq_length:]
last_sequence = np.reshape(last_sequence, (1, seq_length, 1))

# Load the saved model from the H5 file
model = load_model('lstm_model.h5')

# Predict the "VN30" of the next day after the last record
next_day_prediction = model.predict(last_sequence)
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print("Predicted VN30 for the next day:", next_day_prediction[0, 0])
