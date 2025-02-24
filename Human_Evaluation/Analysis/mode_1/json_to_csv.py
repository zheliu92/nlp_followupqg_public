import json
import csv
import pandas as pd

input_json_file_path = "data/train_manual.json"
output_csv_file_path = 'data/train_manual.csv'

data = (pd.read_json(input_json_file_path))

# Open a new CSV file for writing
with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['id', 'question', 'answer', 'follow-up']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write each JSON dictionary as a row in the CSV

    for index, entry in data.iterrows():
        # Write the row with each key mapped to the appropriate value
        writer.writerow({
            'id': entry['id'],
            'question': entry['question'],
            'answer': entry['answer'],
            'follow-up': entry['follow-up']
        })

print(f"CSV file '{output_csv_file_path}' created successfully.")