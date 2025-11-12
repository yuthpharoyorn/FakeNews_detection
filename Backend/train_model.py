# Backend/train_model.py
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import re
import string

def clean_text(text):
    
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    return text

def prepare_data():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    fake_path = os.path.join(project_root, 'news_dataset', 'Fake.csv')
    true_path = os.path.join(project_root, 'news_dataset', 'True.csv')

    # 2. Load Data
    print("Loading data...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    # 3. Add Labels
    df_fake['label'] = 0
    df_true['label'] = 1

    # 4. Merge & Clean
    df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
    
    print("Cleaning data... this might take a minute.")
    # Download NLTK data only once
    nltk.download('stopwords')
    
    df['text'] = df['text'].apply(clean_text)
    
    print("Data cleaned!")
    print(df.head())
    
    # LATER: We will save the model here
    # model.save("fake_news_model.pkl")

if __name__ == "__main__":
    prepare_data()