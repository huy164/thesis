import csv
import urllib.request
from django.http import JsonResponse

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
