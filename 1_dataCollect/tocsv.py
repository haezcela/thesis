import os
import csv
import re

# Define the folder paths
affirmed_folder = 'affirmed_criminal_appeal_cases'
non_affirmed_folder = 'nonAffirmed_criminal_appeal_cases'

# Define the output CSV file path
output_csv = 'cases_dataset.csv'

# Define the headers for the CSV file
headers = ['CaseNumber', 'Decision', 'Laws', 'Judicial Reasoning', 'Facts of the Case']

# Function to get the titles of the .txt files in a given folder
def get_case_numbers(folder, decision):
    case_numbers = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            case_number = filename.replace('.txt', '')
            case_numbers.append((case_number, decision, os.path.join(folder, filename)))
    return case_numbers

# Function to extract laws based on keywords
def extract_laws(text):
    keywords = ['article', 'art', 'republic act', 'ra', 'batas pambansa', 'bp', 'act', 
                'presidential decree', 'pd', 'commonwealth act', 'crime of', 'offense of', 
                'charge', 'charged with', 'violation of']
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    matches = pattern.finditer(text)
    laws_list = []
    for match in matches:
        start = match.start()
        words_after = text[start:].split()[:21]  # Get the keyword and the next 20 words
        laws_list.append(' '.join(words_after))
    return '; '.join(laws_list) if laws_list else ''

# Get case numbers from both folders
affirmed_cases = get_case_numbers(affirmed_folder, 'affirmed')
non_affirmed_cases = get_case_numbers(non_affirmed_folder, 'non-affirmed')

# Combine all case numbers with their decisions
all_cases = affirmed_cases + non_affirmed_cases

# Write the case details to the CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header
    for case_number, decision, filepath in all_cases:
        # Read the content of the case file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract laws from the content
        laws = extract_laws(content)
        
        # Default values for the columns other than CaseNumber, Decision, and Laws
        row = [case_number, decision, laws, '', '']
        writer.writerow(row)

print(f"CSV file '{output_csv}' created successfully with case details.")
