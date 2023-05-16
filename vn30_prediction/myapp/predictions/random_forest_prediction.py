import numpy as np
import os
from tensorflow.keras.models import load_model
from joblib import dump, load
from math import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# convert series to supervised learning
def series_to_supervised(stock_price_df, data, n_in=1, n_out=1, dropnan=True):
	column_names = stock_price_df.columns
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('{}(t-{})'.format(column_names[j], i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('{}(t)'.format(column_names[j])) for j in range(n_vars)]
		else:
			names += [('{}(t+{})'.format(column_names[j], i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def generate_random_forest_predictions(stock_price_df,predict_range):
    
    model_path = os.path.join(os.path.dirname(__file__), 'model/RFRoost.joblib')

    RFRoost_model = load(model_path)

    Shift_df = stock_price_df.copy()
    column_names = stock_price_df.columns
    for col in column_names:
        stock_price_df[f"Diff{col}"] = stock_price_df[col].diff()

    n_days_lag = 10
    n_features = len(column_names)
    reframed = series_to_supervised(stock_price_df,stock_price_df[[f"Diff{col}" for col in column_names]], n_days_lag, 1)
    n_obs = n_days_lag * n_features

    last_input_sequence = reframed.values[-1:, :n_obs]


    predictions = []
    last_values = stock_price_df[column_names].iloc[-1].values.reshape(1, n_features)
    n_predict = predict_range
    for i in range(n_predict):
        prediction = RFRoost_model.predict(last_input_sequence)

        prediction = prediction.reshape((1, n_features))


        last_input_sequence = np.roll(last_input_sequence, -n_features)
        last_input_sequence[:, -n_features:] = prediction
        # print(type(prediction))
        last_values = last_values + prediction
        predictions.append(last_values)
    predictions = np.array(predictions).reshape(n_predict, n_features)
    predictions.shape
    predicted_values_df = pd.DataFrame(data=predictions, columns=column_names)
    
    return predicted_values_df["VN30"]



