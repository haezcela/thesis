import os
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (uncomment the lines below if you haven't downloaded them yet)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the folder paths
affirmed_folder = 'train_affirmed_criminal_appeal_cases'
non_affirmed_folder = 'train_nonAffirmed_criminal_appeal_cases'

# Define the output CSV file path
output_csv = 'cases_dataset_preprocessed.csv'

# Define the headers for the CSV file
headers = ['CaseNumber', 'Verdict', 'Decision', 'Judicial Reasoning', 'Facts of the Case']

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to get the titles of the .txt files in a given folder
def get_case_numbers(folder, verdict):
    case_numbers = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            case_number = filename.replace('.txt', '')
            case_numbers.append((case_number, verdict, os.path.join(folder, filename)))
    return case_numbers

# Function to extract laws based on keywords (Judicial Reasoning)
def extract_judicial_reasoning(text):
    keywords = ['article', 'art', 'republic act', 'ra', 'batas pambansa', 'bp', 'act', 
                'presidential decree', 'pd', 'commonwealth act', 'crime of', 'offense of', 
                'charge', 'charged with', 'violation of']
    pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    matches = pattern.finditer(text)
    judicial_reasoning_list = []
    for match in matches:
        start = match.start()
        words_after = text[start:].split()[:21]  # Get the keyword and the next 20 words
        judicial_reasoning_list.append(' '.join(words_after))
    return '; '.join(judicial_reasoning_list) if judicial_reasoning_list else ''

# Function to extract decisions from the content
def extract_decisions(text):
    decisions = re.findall(r'WHEREFORE(.*?)SO ORDERED', text, re.IGNORECASE | re.DOTALL)
    return [decision.strip() for decision in decisions] if decisions else []

# Function to preprocess text for Word2Vec
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lowercase all tokens
    tokens = [token.lower() for token in tokens]
    
    # Remove stopwords and non-alphanumeric characters
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Get case numbers from both folders
affirmed_cases = get_case_numbers(affirmed_folder, 'affirmed')
non_affirmed_cases = get_case_numbers(non_affirmed_folder, 'non-affirmed')

# Combine all case numbers with their verdicts
all_cases = affirmed_cases + non_affirmed_cases

# Write the preprocessed case details to the CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header
    for case_number, verdict, filepath in all_cases:
        # Read the content of the case file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract judicial reasoning, decisions, and facts of the case from the content
        judicial_reasoning = extract_judicial_reasoning(content)
        decisions = extract_decisions(content)
        facts_of_the_case = preprocess_text(content)
        
        # Default values for the columns other than CaseNumber, Verdict, Decision, Judicial Reasoning, and Facts of the Case
        row = [case_number, verdict, decisions, preprocess_text(judicial_reasoning), facts_of_the_case]
        writer.writerow(row)

print(f"Preprocessed CSV file '{output_csv}' created successfully with case details.")
