# code/main.py

import argparse
import logging
import sys
import os
from .utils import load_config, preprocess_data, train_model


def setup_logging():
    """
    Sets up logging to both console and a log file.
    """
    # Determine the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the logs directory and ensure it exists
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define the full path to the log file
    log_file = os.path.join(logs_dir, 'deepseek.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the minimum logging level
    
    # Prevent adding multiple handlers if they already exist
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(levelname)s: %(message)s')
        f_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='DeepSeek: Advanced Data Analysis and Machine Learning Platform')
    
    # Determine the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the default config path relative to the project root
    default_config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    return parser.parse_args()

def main():
    """
    Main function to run the DeepSeek application.
    """
    setup_logging()
    args = parse_arguments()
    logging.info('DeepSeek started')

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logging.error(f'Configuration file not found: {args.config}')
        sys.exit(1)
    except Exception as e:
        logging.error(f'Error loading configuration: {e}')
        sys.exit(1)

    try:
        data = preprocess_data(args.data, config)
    except FileNotFoundError:
        logging.error(f'Data file not found: {args.data}')
        sys.exit(1)
    except pd.errors.ParserError:
        logging.error(f'Error parsing data file: {args.data}')
        sys.exit(1)
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        sys.exit(1)

    try:
        model = train_model(data, config)
    except Exception as e:
        logging.error(f'Error during model training: {e}')
        sys.exit(1)

    logging.info('DeepSeek completed successfully')

if __name__ == '__main__':
    main()
