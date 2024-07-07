import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.utils import resample
import numpy as np

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return words

def load_cases_and_labels(source_folder):
    sentences = []
    labels = []
    # Iterate through each year folder
    for year in os.listdir(source_folder):
        year_path = os.path.join(source_folder, year)
        if os.path.isdir(year_path):
            # Iterate through each month folder within the year folder
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):
                    # Iterate through each case file within the month folder
                    for case_file in os.listdir(month_path):
                        case_file_path = os.path.join(month_path, case_file)
                        if os.path.isfile(case_file_path):
                            with open(case_file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                sentences.append(preprocess_text(text))
                                # Extract label from the file name
                                if "affirmed" in case_file.lower():
                                    labels.append(1)
                                else:
                                    labels.append(0)
    return sentences, labels

# Define the source folder containing the appeal criminal cases
source_folder = 'appeal_criminal_cases'
sentences, labels = load_cases_and_labels(source_folder)

# Display the counts of each class to check balance
affirmed_count = sum(labels)
non_affirmed_count = len(labels) - affirmed_count
print(f"Number of affirmed cases: {affirmed_count}")
print(f"Number of non-affirmed cases: {non_affirmed_count}")

# Balance the dataset by resampling
if affirmed_count > non_affirmed_count:
    affirmed_sentences = [s for s, l in zip(sentences, labels) if l == 1]
    non_affirmed_sentences = [s for s, l in zip(sentences, labels) if l == 0]
    balanced_affirmed = resample(affirmed_sentences, replace=False, n_samples=non_affirmed_count, random_state=42)
    balanced_sentences = balanced_affirmed + non_affirmed_sentences
    balanced_labels = [1] * len(balanced_affirmed) + [0] * non_affirmed_count
else:
    affirmed_sentences = [s for s, l in zip(sentences, labels) if l == 1]
    non_affirmed_sentences = [s for s, l in zip(sentences, labels) if l == 0]
    balanced_non_affirmed = resample(non_affirmed_sentences, replace=False, n_samples=affirmed_count, random_state=42)
    balanced_sentences = affirmed_sentences + balanced_non_affirmed
    balanced_labels = [1] * affirmed_count + [0] * len(balanced_non_affirmed)

# Ensure the lists are numpy arrays for compatibility with machine learning models
balanced_sentences = np.array(balanced_sentences)
balanced_labels = np.array(balanced_labels)

print("Data preprocessing and balancing complete.")
