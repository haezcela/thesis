import os
import csv
import string
import re
import pandas as pd

# Define the input and output CSV file paths
input_csv = 'cases_dataset_preprocessed.csv'
output_csv = 'cases_dataset_reduced.csv'

# Function to preprocess text
def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    return tokens

# Function to remove decision and judicial reasoning from facts of the case
def remove_decision_and_judicial_reasoning_from_facts(decision, judicial_reasoning, facts):
    decision_tokens = ' '.join(preprocess(decision))
    judicial_reasoning_tokens = ' '.join(preprocess(judicial_reasoning))
    combined_text = decision_tokens + ' ' + judicial_reasoning_tokens
    
    combined_pattern = re.compile(re.escape(combined_text), re.IGNORECASE)
    if pd.isnull(facts):
        facts_reduced = ""
    else:
        facts_reduced = combined_pattern.sub('', facts)
    
    return facts_reduced

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Process the DataFrame to remove decision and judicial reasoning from facts of the case
df['Facts of the Case'] = df.apply(lambda row: remove_decision_and_judicial_reasoning_from_facts(
    row['Decision'], row['Judicial Reasoning'], row['Facts of the Case']
), axis=1)

# Write the updated DataFrame to a new CSV file
df.to_csv(output_csv, index=False)

print(f"CSV file '{output_csv}' created successfully with reduced facts of the case.")
