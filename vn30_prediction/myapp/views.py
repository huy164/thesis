import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from rest_framework.decorators import api_view
from rest_framework.response import Response


# url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
# df = pd.read_csv(url)


# data = df['VN30'].values.reshape(-1, 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data)


# model = load_model('/home/huy/Documents/Desktop/thesis/vn30_prediction/myapp/lstm_model.h5')


@api_view(['GET'])
def predict_vn30(request):
    url = 'https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv'
    df = pd.read_csv(url)

    # Preprocess the data
    data = df['VN30'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    seq_length = 5
    last_sequence = data[-seq_length:]
    last_sequence = np.reshape(last_sequence, (1, seq_length, 1))


    model = load_model('lstm_model.h5')

    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)


    return Response({'vn30_prediction': next_day_prediction[0, 0]})
