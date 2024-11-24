# code/main.py

import argparse
import logging
import sys
from utils import load_config, preprocess_data, train_model

def setup_logging():
    logging.basicConfig(
        filename='../logs/deepseek.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepSeek: Advanced Data Analysis and Machine Learning Platform')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_arguments()
    logging.info('DeepSeek started')

    try:
        config = load_config(args.config)
        data = preprocess_data(args.data, config)
        model = train_model(data, config)
        logging.info('DeepSeek completed successfully')
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
