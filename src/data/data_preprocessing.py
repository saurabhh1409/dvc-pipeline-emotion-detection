import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from typing import Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- Logging Configuration ----------------

logger = logging.getLogger("text_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler 
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------- Download NLTK Resources ----------------

try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"NLTK resource download failed: {e}")

# ---------------- Read Data ----------------

def read_data(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV file: {path}")
    return None

# ---------------- Text Cleaning Functions ----------------

def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text: str) -> str:
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# ---------------- Filter very short texts ----------------

def remove_small_sentences(df: pd.DataFrame) -> None:
    df['text'] = df['text'].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

# ---------------- Normalize Text Column ----------------

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        for func in [
            lower_case,
            remove_stop_words,
            removing_numbers,
            removing_punctuations,
            removing_urls,
            lemmatization
        ]:
            df['content'] = df['content'].astype(str).apply(func)
        logger.info("Text normalization completed.")
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
    return df

# ---------------- Main Logic ----------------

def main():
    train_data = read_data('./data/raw/train.csv')
    test_data = read_data('./data/raw/test.csv')

    if train_data is None or test_data is None:
        logger.error("Cannot proceed without valid training and testing data.")
        return

    # Apply cleaning
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # Save data
    data_path = os.path.join("data", "processed")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.info("Processed data saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    main()
