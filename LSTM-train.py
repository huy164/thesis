import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
df = pd.read_csv(url)

data = df['VN30'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - 1):
        x.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(x), np.array(y)

seq_length = 5
x, y = create_sequences(data, seq_length)

split_idx = int(len(x)*0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')



checkpoint = ModelCheckpoint('lstm_model.h5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
          callbacks=callbacks_list)

# Load the best model weights from the checkpoint
model.load_weights('lstm_model.h5')

# Make predictions on the test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_true = scaler.inverse_transform([y_test])

# Predict the "VN30" of the next day after the last record
last_sequence = data[-seq_length:]
last_sequence = np.reshape(last_sequence, (1, seq_length, 1))
next_day_prediction = model.predict(last_sequence)
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print("Predicted VN30 for the next day:", next_day_prediction[0, 0])
