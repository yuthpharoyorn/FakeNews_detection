# predict.py
import re
import string
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from nltk.corpus import stopwords

# --- CONFIGURATION (Must be the same as training) ---
MAX_LEN = 500
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.keras')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pkl')

# --- LOAD MODELS ---
print("Loading model and tokenizer...")
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

print("Model and tokenizer loaded successfully.")

# --- UTILITY FUNCTIONS ---
# You must use the *exact* same cleaning function from training
def clean_text(text):
    """Performs basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r'^[a-z\s]+\(reuters\)\s*-\s*', '', text) # Remove reuters
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    
    # Optional: Stop word removal (uncomment if you used it in training)
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

def predict_fake_news(text):
    """
    Cleans, tokenizes, and predicts the likelihood of a text being fake news.
    """
    # 1. Clean the text
    cleaned_text = clean_text(text)
    
    # 2. Tokenize and convert to sequence
    # Note: We must put it in a list [cleaned_text] because
    # texts_to_sequences expects a list of documents
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    
    # 3. Pad the sequence
    padded_sequence = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Make a prediction
    # model.predict returns a 2D array, e.g., [[0.99]]
    prediction_prob = model.predict(padded_sequence)[0][0]
    
    # 5. Interpret the result
    if prediction_prob > 0.5:
        return {"label": "Real News", "probability": float(prediction_prob)}
    else:
        return {"label": "Fake News", "probability": float(prediction_prob)}

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Example 1: A "fake" sounding text
    test_text_fake = "BREAKING: Sources say the moon is made of cheese and aliens are landing tomorrow. This is 100% true, share this now!"
    
    # Example 2: A "real" sounding text
    test_text_real = "The central bank announced today that interest rates would remain unchanged for the next quarter, citing stable inflation figures."
    
    print("\n--- Prediction 1 ---")
    result1 = predict_fake_news(test_text_fake)
    print(f"Label: {result1['label']}")
    print(f"Probability (Real): {result1['probability']:.4f}")

    print("\n--- Prediction 2 ---")
    result2 = predict_fake_news(test_text_real)
    print(f"Label: {result2['label']}")
    print(f"Probability (Real): {result2['probability']:.4f}")