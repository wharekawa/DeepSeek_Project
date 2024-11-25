# tests/test_utils.py

import unittest
from unittest.mock import mock_open, patch
import pandas as pd
from code.utils import load_config, preprocess_data, train_model

class TestUtils(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='drop_duplicates: true\nfillna: 0')
    def test_load_config_yaml(self, mock_file):
        config = load_config('config/config.yaml')
        self.assertTrue(config['drop_duplicates'])
        self.assertEqual(config['fillna'], 0)

    @patch('builtins.open', new_callable=mock_open, read_data='{"logging_level": "INFO"}')
    def test_load_config_json(self, mock_file):
        config = load_config('config/settings.json')
        self.assertEqual(config['logging_level'], 'INFO')

    @patch('code.utils.pd.read_csv')
    def test_preprocess_data(self, mock_read_csv):
        mock_df = pd.DataFrame({'A': [1, 2, 2, None]})
        mock_read_csv.return_value = mock_df
        config = {'drop_duplicates': True, 'fillna': 0}
        processed_data = preprocess_data('data/sample_data.csv', config)
        self.assertEqual(len(processed_data), 2)
        self.assertEqual(processed_data['A'].iloc[-1], 0)

    def test_train_model(self):
        data = pd.DataFrame({
            'A': [1, 4, 7],
            'B': [2, 5, 8],
            'C': [3, 6, 9]
        })
        config = {
            'drop_duplicates': True,
            'fillna': 0,
            'test_size': 0.33,
            'model_parameters': {
                'algorithm': 'random_forest',
                'n_estimators': 10,
                'random_state': 42
            },
            'output': {
                'model_path': 'models/test_model.pkl',
                'report_path': 'reports/test_report.txt'
            }
        }
        # Assuming train_model returns the trained model
        model = train_model(data, config)
        # Replace 'trained_model_placeholder' with actual model object or properties
        self.assertIsNotNone(model)  # Adjust based on actual implementation

if __name__ == '__main__':
    unittest.main()
