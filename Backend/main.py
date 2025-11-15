from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # Import BaseModel for request body
import os
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
# Note: You might need to run `nltk.download('stopwords')` once from your terminal
# if you get an error here.
from nltk.corpus import stopwords

# --- 1. DEFINE REQUEST BODY ---
# This tells FastAPI what the incoming JSON should look like
class NewsArticle(BaseModel):
    text: str

# --- 2. CONFIGURATION ---
MAX_LEN = 500  # Must be the same as in training
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_model.keras')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pkl')

# --- 3. LOAD MODELS ON STARTUP ---
# These will be loaded once when the server starts
# and reused for every request.
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model or tokenizer: {e}")
    # In a real app, you might exit or handle this more gracefully
    model = None
    tokenizer = None

# --- 4. TEXT CLEANING FUNCTION ---
# This must be the *exact* same function used during training
def clean_text(text):
    """Performs basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r'^[a-z\s]+\(reuters\)\s*-\s*', '', text) # Remove reuters
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'httpsS?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    
    # Optional: Stop word removal (uncomment if you used it in training)
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

# --- 5. SETUP FASTAPI APP & CORS ---
app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  # Example: VS Code Live Server
    "http://localhost:5500",
    "http://localhost:3000",  # Example: React/Next.js frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. DEFINE API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}

@app.post("/predict")
def predict_news(article: NewsArticle):
    """
    Cleans, tokenizes, and predicts the likelihood of a text being fake news.
    """
    if not model or not tokenizer:
        return {"error": "Model or tokenizer not loaded. Check server logs."}

    # 1. Get text from the request
    text = article.text
    
    # 2. Clean the text
    cleaned_text = clean_text(text)
    
    # 3. Tokenize and pad
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 4. Make a prediction
    # model.predict returns a 2D array, e.g., [[0.99]]
    prediction_prob = model.predict(padded_sequence)[0][0]
    
    # 5. Interpret the result
    if prediction_prob > 0.5:
        label = "Real News"
    else:
        label = "Fake News"
        
    return {
        "label": label, 
        "probability_real": float(prediction_prob),
        "prediction_raw": float(prediction_prob) # Send back the raw score
    }

# --- 7. (Optional) Run the server with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # This allows you to run: `python main.py`
    # But the recommended way is: `uvicorn main:app --reload`
    uvicorn.run(app, host="127.0.0.1", port=8000)