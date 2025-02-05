# tests/test_main.py

import unittest
from unittest.mock import patch, MagicMock
from code.main import main

class TestMain(unittest.TestCase):

    @patch('code.main.train_model')
    @patch('code.main.preprocess_data')
    @patch('code.main.load_config')
    @patch('code.main.parse_arguments')
    def test_main_success(self, mock_parse_args, mock_load_config, mock_preprocess, mock_train):
        mock_parse_args.return_value = MagicMock(config='config/config.yaml', data='data/sample_data.csv')
        mock_load_config.return_value = {'drop_duplicates': True, 'fillna': 0}
        mock_preprocess.return_value = MagicMock()
        mock_train.return_value = MagicMock()
        with self.assertLogs(level='INFO') as log:
            main()
            self.assertIn('DeepSeek started', log.output[0])
            self.assertIn('DeepSeek completed successfully', log.output[-1])

    @patch('code.main.train_model', side_effect=Exception('Test Exception'))
    @patch('code.main.preprocess_data')
    @patch('code.main.load_config')
    @patch('code.main.parse_arguments')
    def test_main_failure(self, mock_parse_args, mock_load_config, mock_preprocess, mock_train):
        mock_parse_args.return_value = MagicMock(config='config/config.yaml', data='data/sample_data.csv')
        mock_load_config.return_value = {'drop_duplicates': True, 'fillna': 0}
        mock_preprocess.return_value = MagicMock()

        with self.assertLogs(level='ERROR') as log:
            with self.assertRaises(SystemExit):
                main()
            self.assertIn('Error during model training: Test Exception', log.output[0])
            

if __name__ == '__main__':
    unittest.main()



