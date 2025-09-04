import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.base import BaseEstimator
import logging

# ----------------- Logging Configuration -----------------
logger = logging.getLogger("model_evaluation")
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


def load_model(model_path: str) -> BaseEstimator:
    """Load a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_test_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data from a CSV file."""
    try:
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        logger.info(f"Test data loaded from {path} with shape {df.shape}")
        return X, y
    except FileNotFoundError:
        logger.error(f"Test data file not found at {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing test data CSV: {e}")
        raise


def evaluate_model(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate the model using standard metrics."""
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba)
        }
        logger.info(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to JSON: {e}")
        raise


def main():
    model_path = 'model.pkl'
    test_data_path = './data/features/test_bow.csv'
    metrics_output_path = 'metrics.json'

    try:
        model = load_model(model_path)
        X_test, y_test = load_test_data(test_data_path)
        metrics = evaluate_model(model, X_test, y_test)
        logger.info("Evaluation metrics:")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
        save_metrics(metrics, metrics_output_path)
    except Exception:
        logger.error("Evaluation failed due to an error.")


if __name__ == "__main__":
    main()
