import numpy as np
import pandas as pd
import pickle
import os
import yaml

from typing import Tuple, Dict, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
import logging

# ----------------- Logging Configuration -----------------
logger = logging.getLogger("model_training")
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


def load_params(params_path: str) -> Dict:
    """Load model parameters from a YAML config file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Model parameters loaded: {params['model_building']}")
        return params['model_building']
    except FileNotFoundError:
        logger.error(f"Params file not found at {params_path}. Using default parameters.")
        return {'n_estimators': 100, 'learning_rate': 0.1}
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}. Using default parameters.")
        return {'n_estimators': 100, 'learning_rate': 0.1}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}. Using default parameters.")
        return {'n_estimators': 100, 'learning_rate': 0.1}


def load_training_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data from a CSV file."""
    try:
        df = pd.read_csv(path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logger.info(f"Training data loaded from {path} with shape {df.shape}")
        return X, y
    except FileNotFoundError:
        logger.error(f"Training data file not found at {path}")
        return pd.DataFrame(), pd.Series(dtype='int')
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing training data CSV: {e}")
        return pd.DataFrame(), pd.Series(dtype='int')


def train_model(X: pd.DataFrame, y: pd.Series, params: Dict) -> Optional[BaseEstimator]:
    """Train the GradientBoostingClassifier model."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1)
        )
        clf.fit(X, y)
        logger.info("Model training completed successfully.")
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None


def save_model(model: BaseEstimator, path: str) -> None:
    """Save the trained model using pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")


def main():
    params_path = 'params.yaml'
    data_path = './data/features/train_bow.csv'
    model_output_path = 'model.pkl'

    params = load_params(params_path)
    logger.info(f"Using model parameters: {params}")

    X_train, y_train = load_training_data(data_path)
    if X_train.empty or y_train.empty:
        logger.error("Training data is empty. Exiting.")
        return

    model = train_model(X_train, y_train, params)
    if model is None:
        logger.error("Model training failed. Exiting.")
        return

    save_model(model, model_output_path)


if __name__ == "__main__":
    main()
