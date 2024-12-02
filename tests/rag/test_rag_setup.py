# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import unittest
from unittest.mock import patch, MagicMock
import os
import configparser
from opensearch_py_ml.ml_commons.rag_pipeline.rag.rag_setup import Setup
from colorama import Fore, Style

class TestSetup(unittest.TestCase):
    def setUp(self):
        # Sample configuration
        self.sample_config = {
            'service_type': 'managed',
            'region': 'us-west-2',
            'iam_principal': 'arn:aws:iam::123456789012:user/test-user',
            'collection_name': 'test-collection',
            'opensearch_endpoint': 'https://search-hashim-test5.us-west-2.es.amazonaws.com',
            'opensearch_username': '*****',
            'opensearch_password': 'password',
            'default_search_method': 'neural',
            'index_name': 'test-index',
            'embedding_dimension': '768',
            'space_type': 'cosinesimil',
            'ef_construction': '512',
        }

        # Initialize Setup instance
        self.setup_instance = Setup()

        # Set index_name for tests
        self.setup_instance.index_name = self.sample_config['index_name']

        # Mock AWS clients
        self.mock_boto3_client = patch('boto3.client').start()
        self.addCleanup(patch.stopall)

        # Mock OpenSearch client
        self.mock_opensearch_client = MagicMock()
        self.setup_instance.opensearch_client = self.mock_opensearch_client

        # Mock os.path.exists
        self.patcher_os_path_exists = patch('os.path.exists', return_value=True)
        self.mock_os_path_exists = self.patcher_os_path_exists.start()
        self.addCleanup(self.patcher_os_path_exists.stop)

        # Mock configparser
        self.patcher_configparser = patch('configparser.ConfigParser')
        self.mock_configparser_class = self.patcher_configparser.start()
        self.mock_configparser = MagicMock()
        self.mock_configparser_class.return_value = self.mock_configparser
        self.addCleanup(self.patcher_configparser.stop)

    def test_load_config_existing(self):
        with patch('os.path.exists', return_value=True):
            self.mock_configparser.read.return_value = None
            self.mock_configparser.__getitem__.return_value = self.sample_config
            config = self.setup_instance.load_config()
            self.assertEqual(config, self.sample_config)

    def test_load_config_no_file(self):
        with patch('os.path.exists', return_value=False):
            self.mock_configparser.read.return_value = None
            config = self.setup_instance.load_config()
            self.assertEqual(config, {})

    def test_save_config(self):
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.setup_instance.save_config(self.sample_config)
            mock_file.assert_called_with(self.setup_instance.CONFIG_FILE, 'w')
            self.mock_configparser.write.assert_called()

    def test_get_opensearch_domain_name(self):
        with patch.object(Setup, 'load_config', return_value=self.sample_config.copy()):
            domain_name = self.setup_instance.get_opensearch_domain_name()
            self.assertEqual(domain_name, 'hashim-test5')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.rag_setup.OpenSearch')
    def test_initialize_opensearch_client_managed(self, mock_opensearch):
        with patch.object(Setup, 'load_config', return_value=self.sample_config.copy()):
            self.setup_instance = Setup()
            self.setup_instance.opensearch_username = '*****'
            self.setup_instance.opensearch_password = 'password'
            result = self.setup_instance.initialize_opensearch_client()
            self.assertTrue(result)
            mock_opensearch.assert_called_once()

    def test_initialize_opensearch_client_no_endpoint(self):
        self.setup_instance.opensearch_endpoint = ''
        with patch('builtins.print') as mock_print:
            result = self.setup_instance.initialize_opensearch_client()
            self.assertFalse(result)
            mock_print.assert_called_with(f"{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n")

    def test_verify_and_create_index_exists(self):
        self.setup_instance.index_name = self.sample_config['index_name']
        self.mock_opensearch_client.indices.exists.return_value = True
        with patch('builtins.print') as mock_print:
            result = self.setup_instance.verify_and_create_index(768, 'cosinesimil', 512)
            self.assertTrue(result)
            mock_print.assert_called_with(f"{Fore.GREEN}KNN index '{self.setup_instance.index_name}' already exists.{Style.RESET_ALL}\n")

    def test_verify_and_create_index_create(self):
        self.setup_instance.index_name = self.sample_config['index_name']
        self.mock_opensearch_client.indices.exists.return_value = False
        self.setup_instance.create_index = MagicMock()
        with patch('builtins.print') as mock_print:
            result = self.setup_instance.verify_and_create_index(768, 'cosinesimil', 512)
            self.assertTrue(result)
            self.setup_instance.create_index.assert_called_with(768, 'cosinesimil', 512)

    def test_create_index_success(self):
        with patch('builtins.print') as mock_print:
            self.setup_instance.create_index(768, 'cosinesimil', 512)
            self.mock_opensearch_client.indices.create.assert_called_once()
            mock_print.assert_called_with(f"\n{Fore.GREEN}KNN index '{self.setup_instance.index_name}' created successfully with dimension 768, space type cosinesimil, and ef_construction 512.{Style.RESET_ALL}\n")

    def test_create_index_already_exists(self):
        self.mock_opensearch_client.indices.create.side_effect = Exception('resource_already_exists_exception')
        with patch('builtins.print') as mock_print:
            self.setup_instance.create_index(768, 'cosinesimil', 512)
            mock_print.assert_called_with(f"\n{Fore.YELLOW}Index '{self.setup_instance.index_name}' already exists.{Style.RESET_ALL}\n")

    def test_get_knn_index_details_default(self):
        with patch('builtins.input', side_effect=['', '', '']):
            with patch('builtins.print'):
                embedding_dimension, space_type, ef_construction = self.setup_instance.get_knn_index_details()
                self.assertEqual(embedding_dimension, 768)
                self.assertEqual(space_type, 'l2')
                self.assertEqual(ef_construction, 512)

    def test_get_truncated_name_within_limit(self):
        name = 'short-name'
        truncated_name = self.setup_instance.get_truncated_name(name, max_length=32)
        self.assertEqual(truncated_name, name)

    def test_get_truncated_name_exceeds_limit(self):
        name = 'a' * 35
        truncated_name = self.setup_instance.get_truncated_name(name, max_length=32)
        self.assertEqual(truncated_name, 'a' * 29 + '...')

    def test_initialize_clients_success(self):
        with patch.object(Setup, 'load_config', return_value=self.sample_config.copy()):
            self.setup_instance = Setup()
            self.setup_instance.service_type = 'managed'
            with patch('boto3.client') as mock_boto_client:
                mock_boto_client.return_value = MagicMock()
                with patch('time.sleep'):
                    with patch('builtins.print') as mock_print:
                        result = self.setup_instance.initialize_clients()
                        self.assertTrue(result)
                        mock_print.assert_called_with(f"{Fore.GREEN}AWS clients initialized successfully.{Style.RESET_ALL}\n")

    def test_initialize_clients_failure(self):
        self.setup_instance.service_type = 'managed'
        with patch('boto3.client', side_effect=Exception('Initialization failed')):
            with patch('builtins.print') as mock_print:
                result = self.setup_instance.initialize_clients()
                self.assertFalse(result)
                mock_print.assert_called_with(f"{Fore.RED}Failed to initialize AWS clients: Initialization failed{Style.RESET_ALL}")

    def test_check_and_configure_aws_already_configured(self):
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.get_credentials.return_value = MagicMock()
            with patch('builtins.input', return_value='no'):
                with patch('builtins.print') as mock_print:
                    self.setup_instance.check_and_configure_aws()
                    mock_print.assert_called_with("AWS credentials are already configured.")

    def test_check_and_configure_aws_not_configured(self):
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.get_credentials.return_value = None
            self.setup_instance.configure_aws = MagicMock()
            with patch('builtins.print'):
                self.setup_instance.check_and_configure_aws()
                self.setup_instance.configure_aws.assert_called_once()

    def test_configure_aws(self):
        with patch('builtins.input', side_effect=['AKIA...', 'SECRET...', 'us-west-2']):
            with patch('subprocess.run') as mock_subprocess_run:
                with patch('builtins.print') as mock_print:
                    self.setup_instance.configure_aws()
                    self.assertEqual(mock_subprocess_run.call_count, 3)
                    mock_print.assert_called_with(f"{Fore.GREEN}AWS credentials have been successfully configured.{Style.RESET_ALL}")

if __name__ == '__main__':
    unittest.main()
