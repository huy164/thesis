from django.shortcuts import render
import requests


try:
    response = requests.get('https://raw.githubusercontent.com/huy164/datasets/master/VN30_price.csv')
    response.raise_for_status()
    data = response.json()
    print
except requests.exceptions.HTTPError as err:
    print(err)
except requests.exceptions.RequestException as err:
    print(err)
