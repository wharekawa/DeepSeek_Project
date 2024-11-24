# code/utils.py

import yaml
import json
import pandas as pd
import logging

def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as file:
            config = json.load(file)
    else:
        logging.error('Unsupported configuration file format.')
        raise ValueError('Unsupported configuration file format.')
    logging.info('Configuration loaded successfully')
    return config

def preprocess_data(data_path, config):
    logging.info(f'Loading data from {data_path}')
    data = pd.read_csv(data_path)
    # Example preprocessing steps
    if config.get('drop_duplicates', False):
        data = data.drop_duplicates()
        logging.info('Dropped duplicate records')
    if 'fillna' in config:
        data = data.fillna(config['fillna'])
        logging.info('Filled missing values')
    return data

def train_model(data, config):
    # Placeholder for model training logic
    logging.info('Starting model training')
    # Example: Implement training based on config
    model = "trained_model_placeholder"
    logging.info('Model training completed')
    return model
