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
from unittest.mock import patch, MagicMock, Mock
from opensearchpy import OpenSearch, AWSV4SignerAuth, exceptions as opensearch_exceptions
from urllib.parse import urlparse
from opensearchpy import RequestsHttpConnection

# Adjust the import to match your project structure
from opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector import OpenSearchConnector

class TestOpenSearchConnector(unittest.TestCase):
    def setUp(self):
        # Sample configuration
        self.config = {
            'region': 'us-east-1',
            'index_name': 'test-index',
            'is_serverless': 'False',
            'opensearch_endpoint': 'https://search-example.us-east-1.es.amazonaws.com',
            'opensearch_username': 'admin',
            'opensearch_password': 'admin',
            'service_type': 'managed',
        }

        # Update the patch target to match the actual import location
        self.patcher_opensearch = patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector.OpenSearch')
        self.MockOpenSearch = self.patcher_opensearch.start()

        # Mocked OpenSearch client instance
        self.mock_opensearch_client = MagicMock()
        self.MockOpenSearch.return_value = self.mock_opensearch_client

        # Patch boto3 Session
        self.patcher_boto3_session = patch('boto3.Session')
        self.MockBoto3Session = self.patcher_boto3_session.start()

        # Mocked boto3 credentials
        self.mock_credentials = Mock()
        self.MockBoto3Session.return_value.get_credentials.return_value = self.mock_credentials

    def tearDown(self):
        self.patcher_opensearch.stop()
        self.patcher_boto3_session.stop()

    def test_initialize_opensearch_client_managed(self):
        connector = OpenSearchConnector(self.config)
        result = connector.initialize_opensearch_client()
        self.assertTrue(result)
        self.MockOpenSearch.assert_called_once()
        self.MockOpenSearch.assert_called_with(
            hosts=[{'host': 'search-example.us-east-1.es.amazonaws.com', 'port': 443}],
            http_auth=('admin', 'admin'),
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )

    def test_initialize_opensearch_client_serverless(self):
        self.config['service_type'] = 'serverless'
        connector = OpenSearchConnector(self.config)
        result = connector.initialize_opensearch_client()
        self.assertTrue(result)
        self.MockOpenSearch.assert_called_once()
        # Check that AWSV4SignerAuth is used
        args, kwargs = self.MockOpenSearch.call_args
        self.assertIsInstance(kwargs['http_auth'], AWSV4SignerAuth)

    def test_initialize_opensearch_client_missing_endpoint(self):
        self.config['opensearch_endpoint'] = ''
        connector = OpenSearchConnector(self.config)
        with patch('builtins.print') as mock_print:
            result = connector.initialize_opensearch_client()
            self.assertFalse(result)
            mock_print.assert_called_with("OpenSearch endpoint not set. Please run setup first.")

    def test_create_index_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        connector.create_index(embedding_dimension=768, space_type='cosinesimil')
        self.mock_opensearch_client.indices.create.assert_called_once()

    def test_create_index_already_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Simulate index already exists exception
        self.mock_opensearch_client.indices.create.side_effect = opensearch_exceptions.RequestError(
            '400', 'resource_already_exists_exception', 'Index already exists'
        )
        with patch('builtins.print') as mock_print:
            connector.create_index(embedding_dimension=768, space_type='cosinesimil')
            mock_print.assert_called_with(f"Index '{self.config['index_name']}' already exists.")

    def test_create_index_other_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Simulate a different exception
        self.mock_opensearch_client.indices.create.side_effect = opensearch_exceptions.RequestError(
            '400', 'some_other_exception', 'Some other error'
        )
        with patch('builtins.print') as mock_print:
            connector.create_index(embedding_dimension=768, space_type='cosinesimil')
            expected_message = f"Error creating index '{self.config['index_name']}': RequestError(400, 'some_other_exception')"
            mock_print.assert_called_with(expected_message)

    def test_verify_and_create_index_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.return_value = True
        with patch('builtins.print') as mock_print:
            result = connector.verify_and_create_index(embedding_dimension=768, space_type='cosinesimil')
            self.assertTrue(result)
            mock_print.assert_called_with(f"KNN index '{self.config['index_name']}' already exists.")

    def test_verify_and_create_index_not_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.return_value = False
        with patch.object(connector, 'create_index') as mock_create_index:
            result = connector.verify_and_create_index(embedding_dimension=768, space_type='cosinesimil')
            self.assertTrue(result)
            mock_create_index.assert_called_once_with(768, 'cosinesimil')

    def test_verify_and_create_index_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.side_effect = Exception('Connection error')
        with patch('builtins.print') as mock_print:
            result = connector.verify_and_create_index(embedding_dimension=768, space_type='cosinesimil')
            self.assertFalse(result)
            mock_print.assert_called_with("Error verifying or creating index: Connection error")

    @patch('opensearchpy.helpers.bulk')
    def test_bulk_index_success(self, mock_bulk):
        mock_bulk.return_value = (100, [])
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{'index': {'_index': 'test-index', '_id': i}} for i in range(100)]
        with patch('builtins.print') as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 100)
            self.assertEqual(error_count, 0)
            mock_print.assert_called_with("Indexed 100 documents successfully. Failed to index 0 documents.")

    @patch('opensearchpy.helpers.bulk')
    def test_bulk_index_with_errors(self, mock_bulk):
        mock_bulk.return_value = (90, [{'index': {'_id': '10', 'error': 'Some error'}}])
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{'index': {'_index': 'test-index', '_id': i}} for i in range(100)]
        with patch('builtins.print') as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 90)
            self.assertEqual(error_count, 1)
            mock_print.assert_called_with("Indexed 90 documents successfully. Failed to index 1 documents.")

    @patch('opensearchpy.helpers.bulk')
    def test_bulk_index_exception(self, mock_bulk):
        mock_bulk.side_effect = Exception('Bulk indexing error')
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{'index': {'_index': 'test-index', '_id': i}} for i in range(100)]
        with patch('builtins.print') as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 0)
            self.assertEqual(error_count, 100)
            mock_print.assert_called_with("Error during bulk indexing: Bulk indexing error")

    def test_search_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Mock search response
        self.mock_opensearch_client.search.return_value = {
            'hits': {'hits': [{'id': 1}, {'id': 2}]}
        }
        results = connector.search(query_text='test', model_id='model-123', k=5)
        self.assertEqual(results, [{'id': 1}, {'id': 2}])
        self.mock_opensearch_client.search.assert_called_once()

    def test_search_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.side_effect = Exception('Search error')
        with patch('builtins.print') as mock_print:
            results = connector.search(query_text='test', model_id='model-123', k=5)
            self.assertEqual(results, [])
            mock_print.assert_called_with("Error during search: Search error")

    def test_search_by_vector_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.return_value = {
            'hits': {'hits': [{'id': 1}, {'id': 2}]}
        }
        results = connector.search_by_vector(vector=[0.1, 0.2, 0.3], k=5)
        self.assertEqual(results, [{'id': 1}, {'id': 2}])
        self.mock_opensearch_client.search.assert_called_once()

    def test_search_by_vector_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.side_effect = Exception('Vector search error')
        with patch('builtins.print') as mock_print:
            results = connector.search_by_vector(vector=[0.1, 0.2, 0.3], k=5)
            self.assertEqual(results, [])
            mock_print.assert_called_with("Error during search: Vector search error")

    def test_check_connection_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Mock info method
        self.mock_opensearch_client.info.return_value = {'version': {'number': '7.10.2'}}
        result = connector.check_connection()
        self.assertTrue(result)
        self.mock_opensearch_client.info.assert_called_once()

    def test_check_connection_failure(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.info.side_effect = Exception('Connection error')
        with patch('builtins.print') as mock_print:
            result = connector.check_connection()
            self.assertFalse(result)
            mock_print.assert_called_with("Error connecting to OpenSearch: Connection error")

if __name__ == '__main__':
    unittest.main()