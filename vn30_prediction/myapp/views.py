import csv
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from pandas import read_csv
from datetime import datetime, timedelta
from myapp.predictions.lstm_prediction import generate_lstm_predictions
from myapp.predictions.random_forest_prediction import generate_random_forest_predictions
from myapp.predictions.xgboost_prediction import generate_xgboost_predictions
import firebase_admin
from firebase_admin import credentials, firestore
import os
# Initialize Firebase Admin SDK
# Get the absolute path to the serviceAccountKey.json file
current_dir = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(current_dir, 'firebase-model-key.json')
cred = credentials.Certificate(service_account_path)  # Update with the path to your service account key file
firebase_admin.initialize_app(cred)
db = firestore.client()

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

    # Get parameters from the request
    predict_range = int(request.GET.get('predict_range', 7))
    algorithm = request.GET.get('algorithm', 'lstm')
    dataset = read_csv('https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv', header=0, index_col=0)
    # Load the appropriate model based on the algorithm parameter
    inv_yhat = []
    if algorithm == 'lstm':
        inv_yhat = generate_lstm_predictions(dataset,predict_range)
    elif algorithm == 'random_forest':
        inv_yhat = generate_random_forest_predictions(dataset,predict_range)
    elif algorithm == 'xgboost':
        inv_yhat = generate_xgboost_predictions(dataset,predict_range)
    else:
        # Handle other algorithm cases or raise an error
        return JsonResponse({'error': 'Unsupported algorithm'})

    # Define the start date
    latest_date = datetime.strptime(dataset.index[-1], '%Y-%m-%d').date()
    start_date = latest_date + timedelta(days=1)

    # Format the response with date and prediction values
    response = []
    for i in range(predict_range):
        prediction = {
            'Date': (start_date + timedelta(days=i)).strftime('%m/%d/%Y'),
            'Price': round(inv_yhat[i], 3)
        }
        response.append(prediction)


    # Create a separate list for modified predictions
    predictions_to_insert = []
    for prediction in response:
        modified_prediction = dict(prediction)  # Create a copy of the prediction dictionary
        modified_prediction['Algorithm'] = algorithm
        predictions_to_insert.append(modified_prediction)

    # Insert the modified predictions into Firebase Firestore
    for prediction in predictions_to_insert:
        db.collection('prediction').add(prediction)




    return JsonResponse(response, safe=False)