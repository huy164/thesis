import os
import pandas as pd

# Path to the directory containing the CSV files
csv_directory = '/path/to/csv/files'

# Get a list of all CSV file names in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Initialize an empty DataFrame to store the concatenated data
concatenated_data = pd.DataFrame()

# Iterate over each CSV file
for csv_file in csv_files:
    # Read the CSV file
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)
    
    # Extract the file name without the extension
    file_name = os.path.splitext(csv_file)[0]
    
    # Replace the "Price" column with the file name
    df['Price'] = file_name
    
    # Concatenate the data with the previously loaded files
    concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

# Save the concatenated data to a new CSV file
concatenated_data.to_csv('VN30_price.csv', index=False)
