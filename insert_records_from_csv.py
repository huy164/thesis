import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['stock']
collection = db['history']

# Read CSV file into a DataFrame
data = pd.read_csv('https://raw.githubusercontent.com/huy164/datasets/master/VN30_price_2.csv')

# Convert DataFrame to a list of dictionaries
records = []
for _, row in data.iterrows():
    record = row.to_dict()  
    records.append(record)

# Insert records into MongoDB collection
collection.insert_many(records)

# Close the MongoDB connection
client.close()