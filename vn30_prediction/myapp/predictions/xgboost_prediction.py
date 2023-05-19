import os
import numpy as np
from pandas import DataFrame
from pandas import concat
import pickle
import xgboost



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	column_names = data.columns
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


def generate_xgboost_predictions(stock_price_df,predict_range):

    column_names = stock_price_df.columns
    Shift_df = stock_price_df.copy()

    for col in column_names:
        stock_price_df["Diff" + column_names] = stock_price_df[column_names].diff()

    n_days_lag = 10
    n_features = len(column_names)

    reframed = series_to_supervised(stock_price_df[[f"Diff{col}" for col in column_names]], n_days_lag, 1)

    n_obs = n_days_lag * n_features
    values = reframed.values
    n_train_days = int(len(reframed.index) * 0.8)
    test = values[n_train_days:, :]

    test_X, test_y = test[:, :n_obs], test[:, n_obs:]
    train_idx = stock_price_df.index <= reframed[:n_train_days].index[-1]
    test_idx = ~train_idx

    model_path = os.path.join(os.path.dirname(__file__), 'model/xgboost_model.pkl')
    with open(model_path, 'rb') as file:
        xgboost_model = pickle.load(file)
	
    for i, col in enumerate(column_names):
        stock_price_df[f'Shift{col}'] = stock_price_df[col].shift(1)
        prev = stock_price_df[f'Shift{col}']
        stock_price_df.loc[test_idx, f'{col}-XGB_1_step_test'] = prev[test_idx] + xgboost_model.predict(test_X)[:, i]



    # Create a copy of stock_price_df
    predictions_df = stock_price_df.copy()
    
    for i, col in enumerate(column_names):
        predictions_df[f'Shift{col}'] = predictions_df[col].shift(1)
        prev = predictions_df[f'Shift{col}']
        predictions_df.loc[test_idx, f'{col}-XGB_1_step_test'] = prev[test_idx] + xgboost_model.predict(test_X)[:, i]
    
    # Get the latest predicted test values within the predict_range
    predictions = predictions_df.loc[test_idx, f'{column_names[-1]}-XGB_1_step_test'].values[-predict_range:]
    
    return predictions
