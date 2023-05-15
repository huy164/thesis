import requests
from bs4 import BeautifulSoup
import csv

# URL of the website to scrape
url = "https://www.investing.com/indices/vn-30-historical-data"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table containing the historical data
table = soup.find('table', {'data-test': 'historical-data-table'})

# Get the first row of the table
row = table.tbody.find_all('tr')[0]

# Extract the data from the row
date = row.find('time').text
price = row.find_all('td')[1].text.replace(',', '')
open_price = row.find_all('td')[2].text.replace(',', '')
high = row.find_all('td')[3].text.replace(',', '')
low = row.find_all('td')[4].text.replace(',', '')
volume = row.find_all('td')[5].text
change = row.find_all('td')[6].text.strip()

# Write the data to a CSV file
with open('stock_prices.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
    writer.writerow([date, price, open_price, high, low, volume, change])
