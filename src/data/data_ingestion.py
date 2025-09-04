import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

#logging Configure
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Create a file handler 
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

# Create and set the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Create and set the formatter
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    logger.info(f"Loading parameters from {params_path}")
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.debug(f"Test size loaded : {test_size}")
        return test_size
    except FileNotFoundError:
        logger.error(f"Params file not found: {params_path}")
        raise 
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise 
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format in {params_path}: {e}")
        raise 
      
'''problems which identifying:
1. File may not exist
2. YAML may be malformed
3. 'test_size' key may be missing
'''
# Corrected URL (removed trailing space)
def read_data(url: str) -> pd.DataFrame:
    logger.info(f"Reading data from URL: {url}")
    try:
        df = pd.read_csv(url)
        logger.debug(f"Data Shape : {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to read data from URL: {url}. Error: {e}")
        raise 


# Drop unnecessary column
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    #Logging.info
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.debug(f"Processed data shape: {final_df.shape} ")
        return final_df
    except KeyError as e:
        logging.error(f"Missing expected column in data: {e}")
        raise 
      
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    logger.info(f"Saving processed data to {data_path}")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Failed to save data to {data_path}. Error: {e}")
        raise 
  

def main(): 
    try:
        logger.info("Data ingestion pipeline started.")
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        logger.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logger.exception("Pipeline failed due to an error.")


if __name__ == "__main__":
  main()

