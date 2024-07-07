import os
import csv
import nltk
from nltk.tokenize import word_tokenize

# Download the NLTK tokenizer models if not already done
nltk.download('punkt')

# Define the input CSV file path
input_csv = 'cases_dataset.csv'
output_csv = 'cases_dataset_tokenized.csv'

# Read the CSV file and tokenize the "Facts of the Case" column
with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        facts = row['Facts of the Case']
        if facts:
            tokenized_facts = ' '.join(word_tokenize(facts))
            row['Facts of the Case'] = tokenized_facts
        writer.writerow(row)

print(f"Tokenized CSV file '{output_csv}' created successfully.")
