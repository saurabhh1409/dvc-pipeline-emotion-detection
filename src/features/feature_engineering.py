import pandas as pd
import numpy as np
import os
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple
import logging

# ----------------- Logging Configuration -----------------
logger = logging.getLogger("feature_engineering")
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

# ----------------- Functions -----------------

def load_params(params_path: str) -> int:
    """Load max_features from params.yaml"""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features: {max_features}")
        return max_features
    except FileNotFoundError:
        logger.error(f"Params file not found: {params_path}. Using default max_features=1000.")
        return 1000  # default fallback
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}. Using default max_features=1000.")
        return 1000
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}. Using default max_features=1000.")
        return 1000

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate training/test data"""
    try:
        train_df = pd.read_csv(train_path).fillna('')
        test_df = pd.read_csv(test_path).fillna('')
        logger.info(f"Loaded training data from {train_path} with shape {train_df.shape}")
        logger.info(f"Loaded test data from {test_path} with shape {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def extract_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform text into BoW features"""
    try:
        X_train = train_df['content'].values
        y_train = train_df['sentiment'].values
        X_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_bow = pd.DataFrame(X_train_bow.toarray())
        train_bow['label'] = y_train

        test_bow = pd.DataFrame(X_test_bow.toarray())
        test_bow['label'] = y_test

        logger.info(f"Feature extraction completed with max_features={max_features}")
        return train_bow, test_bow
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        return pd.DataFrame(), pd.DataFrame()

def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save the feature-engineered train and test data"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_bow.csv'), index=False)
        logger.info(f"Feature data saved successfully in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save feature data: {e}")

def main():
    params_path = 'params.yaml'
    train_path = './data/processed/train_processed.csv'
    test_path = './data/processed/test_processed.csv'
    output_dir = os.path.join("data", "features")

    max_features = load_params(params_path)

    train_df, test_df = load_data(train_path, test_path)
    if train_df.empty or test_df.empty:
        logger.error("Empty dataframes encountered. Exiting.")
        return

    train_features, test_features = extract_features(train_df, test_df, max_features)
    if train_features.empty or test_features.empty:
        logger.error("Feature extraction failed. Exiting.")
        return

    save_features(train_features, test_features, output_dir)


if __name__ == "__main__":
    main()
