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
from unittest.mock import patch, MagicMock, mock_open
import os
import io
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest import Ingest
from opensearchpy import exceptions as opensearch_exceptions
import json

class TestIngest(unittest.TestCase):
    def setUp(self):
        self.config = {
            'region': 'us-east-1',
            'index_name': 'test-index',
            'embedding_model_id': 'test-embedding-model-id',
            'ingest_pipeline_name': 'test-ingest-pipeline'
        }
        self.ingest = Ingest(self.config)

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.OpenSearchConnector')
    def test_initialize_clients_success(self, mock_opensearch_connector):
        mock_instance = mock_opensearch_connector.return_value
        mock_instance.initialize_opensearch_client.return_value = True

        ingest = Ingest(self.config)
        result = ingest.initialize_clients()

        self.assertTrue(result)
        mock_instance.initialize_opensearch_client.assert_called_once()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.OpenSearchConnector')
    def test_initialize_clients_failure(self, mock_opensearch_connector):
        mock_instance = mock_opensearch_connector.return_value
        mock_instance.initialize_opensearch_client.return_value = False

        ingest = Ingest(self.config)
        result = ingest.initialize_clients()

        self.assertFalse(result)
        mock_instance.initialize_opensearch_client.assert_called_once()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isfile')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.walk')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isdir')
    def test_ingest_command_with_valid_files(self, mock_isdir, mock_walk, mock_isfile):
        paths = ['/path/to/dir', '/path/to/file.txt']
        mock_isfile.side_effect = lambda x: x == '/path/to/file.txt'
        mock_isdir.side_effect = lambda x: x == '/path/to/dir'
        mock_walk.return_value = [('/path/to/dir', [], ['file3.pdf'])]

        with patch.object(self.ingest, 'process_and_ingest_data') as mock_process_and_ingest_data:
            self.ingest.ingest_command(paths)
            mock_process_and_ingest_data.assert_called_once()
            args, kwargs = mock_process_and_ingest_data.call_args
            expected_files = ['/path/to/file.txt', '/path/to/dir/file3.pdf']
            self.assertCountEqual(args[0], expected_files)

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isfile')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.walk')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isdir')
    def test_ingest_command_no_valid_files(self, mock_isdir, mock_walk, mock_isfile):
        paths = ['/invalid/path']
        mock_isfile.return_value = False
        mock_isdir.return_value = False

        with patch('builtins.print') as mock_print:
            self.ingest.ingest_command(paths)
            mock_print.assert_any_call('\x1b[33mInvalid path: /invalid/path\x1b[0m')
            mock_print.assert_any_call('\x1b[31mNo valid files found for ingestion.\x1b[0m')

    @patch.object(Ingest, 'initialize_clients', return_value=True)
    @patch.object(Ingest, 'create_ingest_pipeline')
    @patch.object(Ingest, 'process_file')
    @patch.object(Ingest, 'text_embedding', return_value=[0.1, 0.2, 0.3])
    def test_process_and_ingest_data(self, mock_text_embedding, mock_process_file, mock_create_pipeline, mock_initialize_clients):
        file_paths = ['/path/to/file1.txt']
        documents = [{'text': 'Sample text'}]
        mock_process_file.return_value = documents

        # Patch the 'bulk_index' method on the instance's 'opensearch' attribute
        with patch.object(self.ingest.opensearch, 'bulk_index', return_value=(1, 0)) as mock_bulk_index:
            self.ingest.process_and_ingest_data(file_paths)

            mock_initialize_clients.assert_called_once()
            mock_create_pipeline.assert_called_once_with(self.ingest.pipeline_name)
            mock_process_file.assert_called_once_with('/path/to/file1.txt')
            mock_text_embedding.assert_called_once_with('Sample text')
            mock_bulk_index.assert_called_once()

    def test_create_ingest_pipeline_exists(self):
        pipeline_id = 'test-pipeline'
        with patch.object(self.ingest.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.ingest.get_pipeline.return_value = {}

            with patch('builtins.print') as mock_print:
                self.ingest.create_ingest_pipeline(pipeline_id)
                mock_opensearch_client.ingest.get_pipeline.assert_called_once_with(id=pipeline_id)
                mock_print.assert_any_call(f"\nIngest pipeline '{pipeline_id}' already exists.")

    def test_create_ingest_pipeline_not_exists(self):
        pipeline_id = 'test-pipeline'
        pipeline_body = {
            "description": "A text chunking ingest pipeline",
            "processors": [
                {
                    "text_chunking": {
                        "algorithm": {
                            "fixed_token_length": {
                                "token_limit": 384,
                                "overlap_rate": 0.2,
                                "tokenizer": "standard"
                            }
                        },
                        "field_map": {
                            "nominee_text": "passage_chunk"
                        }
                    }
                }
            ]
        }

        with patch.object(self.ingest.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.ingest.get_pipeline.side_effect = opensearch_exceptions.NotFoundError(
                404, "Not Found", {"error": "Pipeline not found"}
            )

            with patch('builtins.print') as mock_print:
                self.ingest.create_ingest_pipeline(pipeline_id)

                mock_opensearch_client.ingest.get_pipeline.assert_called_once_with(id=pipeline_id)
                mock_opensearch_client.ingest.put_pipeline.assert_called_once_with(id=pipeline_id, body=pipeline_body)
                mock_print.assert_any_call(f"\nIngest pipeline '{pipeline_id}' created successfully.")

    @patch('builtins.open', new_callable=mock_open, read_data='col1,col2\nvalue1,value2\n')
    def test_process_csv(self, mock_file):
        file_path = '/path/to/file.csv'
        with patch('csv.DictReader') as mock_csv_reader:
            mock_csv_reader.return_value = [{'col1': 'value1', 'col2': 'value2'}]
            result = self.ingest.process_csv(file_path)
            mock_file.assert_called_once_with(file_path, 'r', newline='', encoding='utf-8')
            self.assertEqual(result, [{'text': json.dumps({'col1': 'value1', 'col2': 'value2'})}])

    @patch('builtins.open', new_callable=mock_open, read_data='Sample TXT data')
    def test_process_txt(self, mock_file):
        file_path = '/path/to/file.txt'
        result = self.ingest.process_txt(file_path)
        mock_file.assert_called_once_with(file_path, 'r')
        self.assertEqual(result, [{'text': 'Sample TXT data'}])

    @patch('PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_pdf(self, mock_file, mock_pdf_reader):
        file_path = '/path/to/file.pdf'
        mock_pdf_reader_instance = mock_pdf_reader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = 'Sample PDF page text'
        mock_pdf_reader_instance.pages = [mock_page]

        result = self.ingest.process_pdf(file_path)

        mock_file.assert_called_once_with(file_path, 'rb')
        mock_pdf_reader.assert_called_once_with(mock_file.return_value)
        self.assertEqual(result, [{'text': 'Sample PDF page text'}])

    @patch('time.sleep', return_value=None)
    def test_text_embedding_failure(self, mock_sleep):
        text = 'Sample text'

        with patch.object(self.ingest.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.transport.perform_request.side_effect = Exception('Test exception')

            with patch('builtins.print') as mock_print:
                with self.assertRaises(Exception) as context:
                    self.ingest.text_embedding(text, max_retries=1)
                self.assertTrue('Test exception' in str(context.exception))
                mock_print.assert_any_call('Error on attempt 1: Test exception')

    def test_text_embedding_success(self):
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

        with patch.object(self.ingest.opensearch, 'opensearch_client') as mock_opensearch_client:
            mock_opensearch_client.transport.perform_request.return_value = response

            result = self.ingest.text_embedding(text)

            self.assertEqual(result, embedding)
            mock_opensearch_client.transport.perform_request.assert_called_once()

if __name__ == '__main__':
    unittest.main()