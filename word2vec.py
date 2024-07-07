import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load preprocessed data from CSV
csv_file = 'cases_dataset_preprocessed.csv'
df = pd.read_csv(csv_file)

# Tokenize function using NLTK
def tokenize_text(text):
    return word_tokenize(text)

# Tokenize and prepare data for Word2Vec
sentences = df['Facts of the Case'].apply(tokenize_text).tolist()

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Example usage of Word2Vec model
word = 'crime'
vector = model.wv[word]

print(f"Word vector for '{word}': {vector}")

# Save the model (optional)
model.save('word2vec_model.bin')

print("Word2Vec model trained and saved successfully.")
