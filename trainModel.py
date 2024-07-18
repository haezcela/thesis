import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def load_dataset(csv_file, text_column, label_column):
    df = pd.read_csv(csv_file)
    df[text_column + '_Tokens'] = df[text_column].apply(lambda x: word_tokenize(x.lower()) if pd.notnull(x) else [])
    return df

def train_word2vec(texts, vector_size=100, window=5, min_count=1):
    model = Word2Vec(texts, vector_size=vector_size, window=window, min_count=min_count)
    return model

def preprocess_texts(texts, word_index, maxlen=100):
    tokenized_texts = [[word_index.get(word, 0) for word in text] for text in texts]
    padded_sequences = pad_sequences(tokenized_texts, maxlen=maxlen)
    return padded_sequences

class MetricsCallback(Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []
        self.loss_history = []
        self.affirmed_precision_history = []
        self.affirmed_recall_history = []
        self.affirmed_f1_history = []
        self.non_affirmed_precision_history = []
        self.non_affirmed_recall_history = []
        self.non_affirmed_f1_history = []
        self.confusion_matrices = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X) > 0.5).astype("int32")

        loss = logs['loss']
        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred, zero_division=1)
        recall = recall_score(self.y, y_pred, zero_division=1)
        f1 = f1_score(self.y, y_pred, zero_division=1)

        self.accuracy_history.append(accuracy)
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)
        self.loss_history.append(loss)

        affirmed_precision = precision_score(self.y, y_pred, pos_label=1, zero_division=1)
        affirmed_recall = recall_score(self.y, y_pred, pos_label=1, zero_division=1)
        affirmed_f1 = f1_score(self.y, y_pred, pos_label=1, zero_division=1)

        self.affirmed_precision_history.append(affirmed_precision)
        self.affirmed_recall_history.append(affirmed_recall)
        self.affirmed_f1_history.append(affirmed_f1)

        non_affirmed_precision = precision_score(self.y, y_pred, pos_label=0, zero_division=1)
        non_affirmed_recall = recall_score(self.y, y_pred, pos_label=0, zero_division=1)
        non_affirmed_f1 = f1_score(self.y, y_pred, pos_label=0, zero_division=1)

        self.non_affirmed_precision_history.append(non_affirmed_precision)
        self.non_affirmed_recall_history.append(non_affirmed_recall)
        self.non_affirmed_f1_history.append(non_affirmed_f1)

        cm = confusion_matrix(self.y, y_pred)
        self.confusion_matrices.append(cm)

        print(f'Epoch {epoch+1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        print(classification_report(self.y, y_pred, zero_division=1))
        print("\nConfusion Matrix:")
        print(cm)

def train_cnn(X, y, embedding_matrix, epochs=10, batch_size=32, callbacks=None):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=X.shape[1], trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return model, history

def plot_metrics(loss_history, accuracy_history, precision_history, recall_history, f1_history, affirmed_precision_history, affirmed_recall_history, affirmed_f1_history, non_affirmed_precision_history, non_affirmed_recall_history, non_affirmed_f1_history, epochs, confusion_matrices):
    plt.figure(figsize=(12, 16))

    # Overall metrics
    plt.subplot(5, 1, 1)
    plt.plot(epochs, loss_history, label='Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.plot(epochs, accuracy_history, label='Accuracy', marker='o')
    plt.plot(epochs, precision_history, label='Precision', marker='o')
    plt.plot(epochs, recall_history, label='Recall', marker='o')
    plt.plot(epochs, f1_history, label='F1-score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Overall Training Metrics per Epoch')
    plt.legend()
    plt.grid(True)

    # Affirmed class metrics
    plt.subplot(5, 1, 3)
    plt.plot(epochs, affirmed_precision_history, label='Affirmed Precision', marker='o')
    plt.plot(epochs, affirmed_recall_history, label='Affirmed Recall', marker='o')
    plt.plot(epochs, affirmed_f1_history, label='Affirmed F1-score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Affirmed Class Training Metrics per Epoch')
    plt.legend()
    plt.grid(True)

    # Non-affirmed class metrics
    plt.subplot(5, 1, 4)
    plt.plot(epochs, non_affirmed_precision_history, label='Non-affirmed Precision', marker='o')
    plt.plot(epochs, non_affirmed_recall_history, label='Non-affirmed Recall', marker='o')
    plt.plot(epochs, non_affirmed_f1_history, label='Non-affirmed F1-score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Non-affirmed Class Training Metrics per Epoch')
    plt.legend()
    plt.grid(True)

    # Confusion matrix values
    tp_values = [cm[1, 1] for cm in confusion_matrices]
    tn_values = [cm[0, 0] for cm in confusion_matrices]
    fp_values = [cm[0, 1] for cm in confusion_matrices]
    fn_values = [cm[1, 0] for cm in confusion_matrices]

    plt.subplot(5, 1, 5)
    plt.plot(epochs, tp_values, label='True Positives', marker='o')
    plt.plot(epochs, tn_values, label='True Negatives', marker='o')
    plt.plot(epochs, fp_values, label='False Positives', marker='o')
    plt.plot(epochs, fn_values, label='False Negatives', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Confusion Matrix Values')
    plt.title('Confusion Matrix Values per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Displaying confusion matrices for each epoch
    for epoch, cm in enumerate(confusion_matrices, 1):
        print(f'Epoch {epoch} Confusion Matrix:')
        print(cm)
        print()

def experiment(csv_file, text_column, label_column, epochs=10, vector_size=100, window=5, min_count=1, model_filename=None):
    df = load_dataset(csv_file, text_column, label_column)

    texts = df[text_column + '_Tokens'].tolist()
    labels = (df[label_column] == 'affirmed').astype(int).values

    word2vec_model = train_word2vec(texts, vector_size=vector_size, window=window, min_count=min_count)

    combined_vocab = set(word2vec_model.wv.index_to_key)
    embedding_matrix = np.zeros((len(combined_vocab) + 1, vector_size))  # +1 for padding index 0
    word_index = {word: idx + 1 for idx, word in enumerate(combined_vocab)}

    for word, idx in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]

    X = preprocess_texts(texts, word_index)

    metrics_callback = MetricsCallback(X, labels)
    model, history = train_cnn(X, labels, embedding_matrix, epochs=epochs, callbacks=[metrics_callback])

    # Plotting metrics
    plot_metrics(metrics_callback.loss_history,
                 metrics_callback.accuracy_history,
                 metrics_callback.precision_history,
                 metrics_callback.recall_history,
                 metrics_callback.f1_history,
                 metrics_callback.affirmed_precision_history,
                 metrics_callback.affirmed_recall_history,
                 metrics_callback.affirmed_f1_history,
                 metrics_callback.non_affirmed_precision_history,
                 metrics_callback.non_affirmed_recall_history,
                 metrics_callback.non_affirmed_f1_history,
                 epochs=np.arange(1, epochs + 1),
                 confusion_matrices=metrics_callback.confusion_matrices)

    # Save model if specified
    if model_filename:
        model.save(model_filename)

# Based on Decision with 10 epochs:
experiment('/content/drive/MyDrive/dataset/cases_dataset_reduced.csv', 'Decision', 'Verdict', epochs=10, vector_size=100, window=5, min_count=1, model_filename='opBalancedDecision_cnn_model.keras')
