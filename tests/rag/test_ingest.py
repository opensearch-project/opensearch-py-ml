# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

from opensearchpy import exceptions as opensearch_exceptions

from opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest import Ingest


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "region": "us-east-1",
            "index_name": "test-index",
            "embedding_model_id": "test-embedding-model-id",
            "ingest_pipeline_name": "test-ingest-pipeline",
        }
        self.ingest = Ingest(self.config)
        self.ingest.embedding_client = MagicMock()

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.OpenSearchConnector")
    def test_initialize_clients_success(self, mock_opensearch_connector):
        mock_instance = mock_opensearch_connector.return_value
        mock_instance.initialize_opensearch_client.return_value = True

        ingest = Ingest(self.config)
        result = ingest.initialize_clients()

        self.assertTrue(result)
        mock_instance.initialize_opensearch_client.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.OpenSearchConnector")
    def test_initialize_clients_failure(self, mock_opensearch_connector):
        mock_instance = mock_opensearch_connector.return_value
        mock_instance.initialize_opensearch_client.return_value = False

        ingest = Ingest(self.config)
        result = ingest.initialize_clients()

        self.assertFalse(result)
        mock_instance.initialize_opensearch_client.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isfile")
    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.walk")
    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isdir")
    def test_ingest_command_with_valid_files(self, mock_isdir, mock_walk, mock_isfile):
        paths = ["/path/to/dir", "/path/to/file.txt"]
        mock_isfile.side_effect = lambda x: x == "/path/to/file.txt"
        mock_isdir.side_effect = lambda x: x == "/path/to/dir"
        mock_walk.return_value = [("/path/to/dir", [], ["file3.pdf"])]

        with patch.object(
            self.ingest, "process_and_ingest_data"
        ) as mock_process_and_ingest_data:
            self.ingest.ingest_command(paths)
            mock_process_and_ingest_data.assert_called_once()
            args, kwargs = mock_process_and_ingest_data.call_args
            expected_files = ["/path/to/file.txt", "/path/to/dir/file3.pdf"]
            self.assertCountEqual(args[0], expected_files)

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isfile")
    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.walk")
    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.ingest.os.path.isdir")
    def test_ingest_command_no_valid_files(self, mock_isdir, mock_walk, mock_isfile):
        paths = ["/invalid/path"]
        mock_isfile.return_value = False
        mock_isdir.return_value = False

        with patch("builtins.print") as mock_print:
            self.ingest.ingest_command(paths)
            mock_print.assert_any_call("\x1b[33mInvalid path: /invalid/path\x1b[0m")
            mock_print.assert_any_call(
                "\x1b[31mNo valid files found for ingestion.\x1b[0m"
            )

    @patch.object(Ingest, "initialize_clients", return_value=True)
    @patch.object(Ingest, "create_ingest_pipeline")
    @patch.object(Ingest, "process_file")
    def test_process_and_ingest_data(
        self, mock_process_file, mock_create_pipeline, mock_initialize_clients
    ):
        file_paths = ["/path/to/file1.txt"]
        documents = [{"text": "Sample text"}]
        mock_process_file.return_value = documents

        mock_embedding_client = MagicMock()
        mock_embedding_client.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        self.ingest.embedding_client = mock_embedding_client

        with patch.object(
            self.ingest.opensearch, "bulk_index", return_value=(1, 0)
        ) as mock_bulk_index:
            self.ingest.process_and_ingest_data(file_paths)

            mock_initialize_clients.assert_called_once()
            mock_create_pipeline.assert_called_once_with(self.ingest.pipeline_name)
            mock_process_file.assert_called_once_with("/path/to/file1.txt")
            mock_embedding_client.get_text_embedding.assert_called_once_with(
                "Sample text"
            )
            mock_bulk_index.assert_called_once()

    def test_create_ingest_pipeline_exists(self):
        pipeline_id = "test-pipeline"
        with patch.object(
            self.ingest.opensearch, "opensearch_client"
        ) as mock_opensearch_client:
            mock_opensearch_client.ingest.get_pipeline.return_value = {}

            with patch("builtins.print") as mock_print:
                self.ingest.create_ingest_pipeline(pipeline_id)
                mock_opensearch_client.ingest.get_pipeline.assert_called_once_with(
                    id=pipeline_id
                )
                mock_print.assert_any_call(
                    f"\nIngest pipeline '{pipeline_id}' already exists."
                )

    def test_create_ingest_pipeline_not_exists(self):
        pipeline_id = "test-pipeline"
        expected_pipeline_body = {
            "description": "A text chunking and embedding ingest pipeline",
            "processors": [
                {
                    "text_chunking": {
                        "algorithm": {"delimiter": {"delimiter": "."}},
                        "field_map": {"passage_text": "passage_chunk"},
                    }
                },
                {
                    "text_embedding": {
                        "model_id": self.config["embedding_model_id"],
                        "field_map": {"passage_chunk": "passage_embedding"},
                    }
                },
            ],
        }

        with patch.object(
            self.ingest.opensearch, "opensearch_client"
        ) as mock_opensearch_client:
            mock_opensearch_client.ingest.get_pipeline.side_effect = (
                opensearch_exceptions.NotFoundError(
                    404, "Not Found", {"error": "Pipeline not found"}
                )
            )

            with patch("builtins.print") as mock_print:
                self.ingest.create_ingest_pipeline(pipeline_id)

                mock_opensearch_client.ingest.get_pipeline.assert_called_once_with(
                    id=pipeline_id
                )
                mock_opensearch_client.ingest.put_pipeline.assert_called_once_with(
                    id=pipeline_id, body=expected_pipeline_body
                )
                mock_print.assert_any_call(
                    f"\nIngest pipeline '{pipeline_id}' created successfully."
                )

    @patch(
        "builtins.open", new_callable=mock_open, read_data="col1,col2\nvalue1,value2\n"
    )
    def test_process_csv(self, mock_file):
        file_path = "/path/to/file.csv"
        with patch("csv.DictReader") as mock_csv_reader:
            mock_csv_reader.return_value = [{"col1": "value1", "col2": "value2"}]
            result = self.ingest.process_csv(file_path)
            mock_file.assert_called_once_with(
                file_path, "r", newline="", encoding="utf-8"
            )
            self.assertEqual(
                result, [{"text": json.dumps({"col1": "value1", "col2": "value2"})}]
            )

    @patch("builtins.open", new_callable=mock_open, read_data="Sample TXT data")
    def test_process_txt(self, mock_file):
        file_path = "/path/to/file.txt"
        result = self.ingest.process_txt(file_path)
        mock_file.assert_called_once_with(file_path, "r")
        self.assertEqual(result, [{"text": "Sample TXT data"}])

    @patch("PyPDF2.PdfReader")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_pdf(self, mock_file, mock_pdf_reader):
        file_path = "/path/to/file.pdf"
        mock_pdf_reader_instance = mock_pdf_reader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF page text"
        mock_pdf_reader_instance.pages = [mock_page]

        result = self.ingest.process_pdf(file_path)

        mock_file.assert_called_once_with(file_path, "rb")
        mock_pdf_reader.assert_called_once_with(mock_file.return_value)
        self.assertEqual(result, [{"text": "Sample PDF page text"}])

    def test_text_embedding_failure(self):
        text = "Sample text"
        self.ingest.embedding_client.get_text_embedding.side_effect = Exception(
            "Test exception"
        )

        with self.assertRaises(Exception) as context:
            self.ingest.embedding_client.get_text_embedding(text)

        self.assertTrue("Test exception" in str(context.exception))

    def test_text_embedding_success(self):
        text = "Sample text"
        embedding = [0.1, 0.2, 0.3]
        self.ingest.embedding_client.get_text_embedding.return_value = embedding

        result = self.ingest.embedding_client.get_text_embedding(text)

        self.assertEqual(result, embedding)
        self.ingest.embedding_client.get_text_embedding.assert_called_once_with(text)


if __name__ == "__main__":
    unittest.main()
