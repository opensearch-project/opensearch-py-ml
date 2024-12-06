# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch, MagicMock
from opensearch_py_ml.ml_commons.rag_pipeline.rag.query import Query
from opensearchpy import exceptions as opensearch_exceptions
import json

class TestQuery(unittest.TestCase):
    def setUp(self):
        # Patch 'print' to suppress output during tests
        self.print_patcher = patch('builtins.print')
        self.mock_print = self.print_patcher.start()

        self.config = {
            'index_name': 'test-index',
            'embedding_model_id': 'test-embedding-model-id',
            'llm_model_id': 'test-llm-model-id',
            'region': 'us-east-1',
            'default_search_method': 'neural',
            'llm_max_token_count': '1000',
            'llm_temperature': '0.7',
            'llm_top_p': '0.9',
            'llm_stop_sequences': ''
        }
        # Do not instantiate Query here to avoid unmocked initialization.

    def tearDown(self):
        # Stop the 'print' patcher
        self.print_patcher.stop()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.boto3.client')
    def test_initialize_clients_success(self, mock_boto3_client, mock_opensearch_connector):
        # Mock OpenSearch client initialization
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True
        mock_opensearch_connector_instance.check_connection.return_value = True

        # Mock Bedrock client initialization
        mock_bedrock_client = MagicMock()
        mock_boto3_client.return_value = mock_bedrock_client

        # Initialize Query instance after patches
        query_instance = Query(self.config)

        self.assertIsNotNone(query_instance.opensearch)
        self.assertEqual(query_instance.bedrock_client, mock_bedrock_client)
        mock_opensearch_connector_instance.initialize_opensearch_client.assert_called_once()
        mock_boto3_client.assert_called_once_with('bedrock-runtime', region_name='us-east-1')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_initialize_clients_opensearch_failure(self, mock_opensearch_connector):
        # Mock OpenSearch client initialization failure
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = False

        # Since 'print' is mocked in setUp, we can use 'self.mock_print' to assert calls
        query_instance = Query(self.config)
        mock_opensearch_connector_instance.initialize_opensearch_client.assert_called_once()
        self.mock_print.assert_any_call("Failed to initialize OpenSearch client.")
        self.mock_print.assert_any_call("Failed to initialize clients. Aborting.")

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_extract_relevant_sentences(self, mock_opensearch_connector):
        # Mock OpenSearch client to prevent initialization
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        query_instance = Query(self.config)
        query_text = 'What is the capital of France?'
        text = 'Paris is the capital of France. It is known for the Eiffel Tower.'
        expected_sentences = ['Paris is the capital of France']

        result = query_instance.extract_relevant_sentences(query_text, text)
        self.assertIn(expected_sentences[0], result)

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_bulk_query_neural_success(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        queries = ['What is the capital of France?']
        mock_hits = [
            {
                '_score': 1.0,
                '_source': {'content': 'Paris is the capital of France.'}
            }
        ]
        query_instance = Query(self.config)
        with patch.object(query_instance.opensearch, 'search', return_value=mock_hits):
            results = query_instance.bulk_query_neural(queries, k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['num_results'], 1)
            self.assertEqual(results[0]['documents'][0]['source']['content'], 'Paris is the capital of France.')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_bulk_query_neural_failure(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        queries = ['What is the capital of France?']
        query_instance = Query(self.config)
        with patch.object(query_instance.opensearch, 'search', side_effect=Exception('Search error')):
            results = query_instance.bulk_query_neural(queries, k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['num_results'], 0)
            self.mock_print.assert_any_call("\x1b[31mError performing search for query 'What is the capital of France?': Search error\x1b[0m")

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_bulk_query_semantic_success(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        queries = ['What is the capital of France?']
        embedding = [0.1, 0.2, 0.3]
        mock_hits = [
            {
                '_score': 1.0,
                '_source': {'nominee_text': 'Paris is the capital of France.'}
            }
        ]
        query_instance = Query(self.config)
        with patch.object(query_instance, 'text_embedding', return_value=embedding):
            with patch.object(query_instance.opensearch, 'search_by_vector', return_value=mock_hits):
                results = query_instance.bulk_query_semantic(queries, k=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]['num_results'], 1)
                self.assertIn('Paris is the capital of France.', results[0]['context'])

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_bulk_query_semantic_embedding_failure(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        queries = ['What is the capital of France?']
        query_instance = Query(self.config)
        with patch.object(query_instance, 'text_embedding', return_value=None):
            results = query_instance.bulk_query_semantic(queries, k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]['num_results'], 0)
            self.mock_print.assert_any_call('\x1b[31mFailed to generate embedding for query: What is the capital of France?\x1b[0m')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_text_embedding_success(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        text = 'Sample text'
        embedding = [0.1, 0.2, 0.3]
        response = {
            'inference_results': [
                {
                    'output': [
                        {'data': embedding}
                    ]
                }
            ]
        }
        query_instance = Query(self.config)
        with patch.object(query_instance.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.transport.perform_request.return_value = response

            result = query_instance.text_embedding(text)
            self.assertEqual(result, embedding)
            mock_opensearch_client.transport.perform_request.assert_called_once()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_text_embedding_failure(self, mock_opensearch_connector):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        text = 'Sample text'
        query_instance = Query(self.config)
        with patch.object(query_instance.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.transport.perform_request.side_effect = Exception('Test exception')

            with self.assertRaises(Exception) as context:
                query_instance.text_embedding(text, max_retries=1)
            self.assertTrue('Test exception' in str(context.exception))
            self.mock_print.assert_any_call('Error on attempt 1: Test exception')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.tiktoken.get_encoding')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.boto3.client')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_generate_answer_success(self, mock_opensearch_connector, mock_boto3_client, mock_get_encoding):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        prompt = 'Sample prompt'
        llm_config = {
            'maxTokenCount': 100,
            'temperature': 0.7,
            'topP': 0.9,
            'stopSequences': []
        }
        encoding_instance = MagicMock()
        encoding_instance.encode.return_value = [1, 2, 3]
        encoding_instance.decode.return_value = prompt
        mock_get_encoding.return_value = encoding_instance

        mock_bedrock_client = mock_boto3_client.return_value
        response_stream = MagicMock()
        response_stream.read.return_value = json.dumps({
            'results': [
                {'outputText': 'Generated answer'}
            ]
        })
        response = {'body': response_stream}
        mock_bedrock_client.invoke_model.return_value = response

        query_instance = Query(self.config)
        query_instance.bedrock_client = mock_bedrock_client  # Set the mocked bedrock client

        answer = query_instance.generate_answer(prompt, llm_config)
        self.assertEqual(answer, 'Generated answer')
        mock_bedrock_client.invoke_model.assert_called_once()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.tiktoken.get_encoding')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.boto3.client')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.query.OpenSearchConnector')
    def test_generate_answer_failure(self, mock_opensearch_connector, mock_boto3_client, mock_get_encoding):
        # Mock OpenSearch client
        mock_opensearch_connector_instance = mock_opensearch_connector.return_value
        mock_opensearch_connector_instance.initialize_opensearch_client.return_value = True

        prompt = 'Sample prompt'
        llm_config = {
            'maxTokenCount': 100,
            'temperature': 0.7,
            'topP': 0.9,
            'stopSequences': []
        }
        encoding_instance = MagicMock()
        encoding_instance.encode.return_value = [1, 2, 3]
        encoding_instance.decode.return_value = prompt
        mock_get_encoding.return_value = encoding_instance

        mock_bedrock_client = mock_boto3_client.return_value
        mock_bedrock_client.invoke_model.side_effect = Exception('LLM error')

        query_instance = Query(self.config)
        query_instance.bedrock_client = mock_bedrock_client  # Set the mocked bedrock client

        answer = query_instance.generate_answer(prompt, llm_config)
        self.assertIsNone(answer)
        self.mock_print.assert_any_call('Error generating answer from LLM: LLM error')

if __name__ == '__main__':
    unittest.main()