# code/utils.py

import yaml
import json
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

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
        before = len(data)
        data = data.drop_duplicates()
        after = len(data)
        logging.info(f'Dropped duplicate records: {before - after} duplicates removed')
    if 'fillna' in config:
        data = data.fillna(config['fillna'])
        logging.info('Filled missing values')
    return data

def train_model(data, config):
    logging.info('Starting model training')
    
    # Assuming the last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split data
    test_size = config.get('test_size', 0.2)
    random_state = config.get('model_parameters', {}).get('random_state', 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Initialize the model
    algorithm = config.get('model_parameters', {}).get('algorithm', 'random_forest')
    
    if algorithm == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=config.get('model_parameters', {}).get('n_estimators', 100),
            random_state=random_state
        )
    else:
        logging.error(f'Unsupported algorithm: {algorithm}')
        raise ValueError(f'Unsupported algorithm: {algorithm}')
    
    # Train the model
    model.fit(X_train, y_train)
    logging.info('Model training completed')
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    # Additional Metrics
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    
    logging.info(f'Classification Report:\n{report}')
    logging.info(f'Confusion Matrix:\n{confusion}')
    
    # Save the model
    model_path = config.get('output', {}).get('model_path', 'models/model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f'Model saved to {model_path}')
    
    return model


