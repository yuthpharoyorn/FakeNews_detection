# Backend/train_model.py
import pandas as pd
import os
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
# Define the maximum number of words to keep in the vocabulary
MAX_WORDS = 10000 
# Define the maximum length of a sequence/article (e.g., first 500 words)
MAX_LEN = 500
# Dimension for the word embeddings
EMBEDDING_DIM = 100 
# ---------------------

def clean_text(text):
    """Performs basic text cleaning and removal of stop words."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)       # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text)        # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text)            # Remove newlines
    
    # Optional: Remove Stop Words
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

def prepare_data():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    fake_path = os.path.join(project_root, 'news_dataset', 'Fake.csv')
    true_path = os.path.join(project_root, 'news_dataset', 'True.csv')

    # 2. Load Data
    print("Loading data...")
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Check path: {e.filename}")
        return None, None

    # 3. Add Labels
    df_fake['label'] = 0  # Fake
    df_true['label'] = 1  # Real

    # 4. Merge & Clean
    df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
    
    # Combine title and text for stronger feature extraction
    df['content'] = df['title'] + ' ' + df['text']
    
    print(f"Total dataset size: {len(df)}")
    print("Cleaning data...")
    
    # Ensure NLTK data is available
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    
    df['content'] = df['content'].apply(clean_text)
    
    print("Data cleaned! Ready for vectorization.")
    return df

def train_and_evaluate_model(df):
    
    # --- 1. Split Data ---
    X = df['content'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 2. Tokenization and Sequencing (Word Embedding Prep) ---
    print("Tokenizing text and padding sequences...")
    
    # Tokenizer converts words to integer indices
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences of integers
    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to ensure all input vectors have the same length (MAX_LEN)
    X_train_padded = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # --- 3. Build Bidirectional LSTM Model ---
    print("Building model...")
    
    model = Sequential([
        # Embedding Layer: Maps word indices to dense vectors
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # Bidirectional LSTM: Processes sequence data in both forward and backward directions
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        
        # Dense layers for classification
        Dense(24, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting
        Dense(1, activation='sigmoid') # Sigmoid for binary classification (0 or 1)
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print(model.summary())
    
    # --- 4. Train Model ---
    print("Starting training...")
    history = model.fit(X_train_padded, y_train, 
                        epochs=5, # Start with a few epochs
                        batch_size=32, 
                        validation_split=0.1, # Use 10% of training data for validation
                        verbose=1)

    # --- 5. Evaluate Model ---
    print("\nEvaluating model performance on test set...")
    loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate Classification Report
    y_pred_probs = model.predict(X_test_padded)
    y_pred = (y_pred_probs > 0.5).astype("int32")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # --- 6. Save Model ---
    model_save_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.h5'))
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

if __name__ == "__main__":
    df = prepare_data()
    if df is not None:
        train_and_evaluate_model(df)