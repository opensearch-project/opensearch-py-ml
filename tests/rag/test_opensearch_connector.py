# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, Mock, patch

from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection
from opensearchpy import exceptions as opensearch_exceptions

# Adjust the import to match your project structure
from opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector import (
    OpenSearchConnector,
)


class TestOpenSearchConnector(unittest.TestCase):
    def setUp(self):
        # Sample configuration
        self.config = {
            "region": "us-east-1",
            "index_name": "test-index",
            "is_serverless": "False",
            "opensearch_endpoint": "https://search-example.us-east-1.es.amazonaws.com",
            "opensearch_username": "*****",
            "opensearch_password": "*****",
            "service_type": "managed",
        }

        # Update the patch target to match the actual import location
        self.patcher_opensearch = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector.OpenSearch"
        )
        self.MockOpenSearch = self.patcher_opensearch.start()

        # Mocked OpenSearch client instance
        self.mock_opensearch_client = MagicMock()
        self.MockOpenSearch.return_value = self.mock_opensearch_client

        # Patch boto3 Session
        self.patcher_boto3_session = patch("boto3.Session")
        self.MockBoto3Session = self.patcher_boto3_session.start()

        # Mocked boto3 credentials
        self.mock_credentials = Mock()
        self.MockBoto3Session.return_value.get_credentials.return_value = (
            self.mock_credentials
        )

    def tearDown(self):
        self.patcher_opensearch.stop()
        self.patcher_boto3_session.stop()

    def test_initialize_opensearch_client_managed(self):
        connector = OpenSearchConnector(self.config)
        result = connector.initialize_opensearch_client()
        self.assertTrue(result)
        self.MockOpenSearch.assert_called_once()
        self.MockOpenSearch.assert_called_with(
            hosts=[{"host": "search-example.us-east-1.es.amazonaws.com", "port": 443}],
            http_auth=("*****", "*****"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
        )

    def test_initialize_opensearch_client_serverless(self):
        self.config["service_type"] = "serverless"
        connector = OpenSearchConnector(self.config)
        result = connector.initialize_opensearch_client()
        self.assertTrue(result)
        self.MockOpenSearch.assert_called_once()
        # Check that AWSV4SignerAuth is used
        args, kwargs = self.MockOpenSearch.call_args
        self.assertIsInstance(kwargs["http_auth"], AWSV4SignerAuth)

    def test_initialize_opensearch_client_missing_endpoint(self):
        self.config["opensearch_endpoint"] = ""
        connector = OpenSearchConnector(self.config)
        with patch("builtins.print") as mock_print:
            result = connector.initialize_opensearch_client()
            self.assertFalse(result)
            mock_print.assert_called_with(
                "OpenSearch endpoint not set. Please run setup first."
            )

    def test_create_index_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        connector.create_index(
            embedding_dimension=768,
            space_type="cosinesimil",
            ef_construction=512,
            number_of_shards=1,
            number_of_replicas=1,
            passage_text_field="passage_text",
            passage_chunk_field="passage_chunk",
            embedding_field="passage_embedding",
        )
        self.mock_opensearch_client.indices.create.assert_called_once()

    def test_create_index_already_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.create.side_effect = (
            opensearch_exceptions.RequestError(
                "400", "resource_already_exists_exception", "Index already exists"
            )
        )
        with patch("builtins.print") as mock_print:
            connector.create_index(
                embedding_dimension=768,
                space_type="cosinesimil",
                ef_construction=512,
                number_of_shards=1,
                number_of_replicas=1,
                passage_text_field="passage_text",
                passage_chunk_field="passage_chunk",
                embedding_field="passage_embedding",
            )
            mock_print.assert_called_with(
                f"Index '{self.config['index_name']}' already exists."
            )

    def test_create_index_other_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.create.side_effect = (
            opensearch_exceptions.RequestError(
                400, "some_other_exception", "Some other error"
            )
        )
        with patch("builtins.print") as mock_print:
            connector.create_index(
                embedding_dimension=768,
                space_type="cosinesimil",
                ef_construction=512,
                number_of_shards=1,
                number_of_replicas=1,
                passage_text_field="passage_text",
                passage_chunk_field="passage_chunk",
                embedding_field="passage_embedding",
            )
            expected_message = f"Error creating index '{self.config['index_name']}': RequestError(400, 'some_other_exception')"
            mock_print.assert_called_with(expected_message)

    def test_verify_and_create_index_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.return_value = True
        with patch("builtins.print") as mock_print:
            result = connector.verify_and_create_index(
                embedding_dimension=768,
                space_type="cosinesimil",
                ef_construction=512,
                number_of_shards=1,
                number_of_replicas=1,
                passage_text_field="passage_text",
                passage_chunk_field="passage_chunk",
                embedding_field="passage_embedding",
            )
            self.assertTrue(result)
            mock_print.assert_called_with(
                f"KNN index '{self.config['index_name']}' already exists."
            )

    def test_verify_and_create_index_not_exists(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.return_value = False
        with patch.object(connector, "create_index") as mock_create_index:
            result = connector.verify_and_create_index(
                embedding_dimension=768,
                space_type="cosinesimil",
                ef_construction=512,
                number_of_shards=1,
                number_of_replicas=1,
                passage_text_field="passage_text",
                passage_chunk_field="passage_chunk",
                embedding_field="passage_embedding",
            )
            self.assertTrue(result)
            mock_create_index.assert_called_once_with(
                768,
                "cosinesimil",
                512,
                1,
                1,
                "passage_text",
                "passage_chunk",
                "passage_embedding",
            )

    def test_verify_and_create_index_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.indices.exists.side_effect = Exception(
            "Connection error"
        )
        with patch("builtins.print") as mock_print:
            result = connector.verify_and_create_index(
                embedding_dimension=768,
                space_type="cosinesimil",
                ef_construction=512,
                number_of_shards=1,
                number_of_replicas=1,
                passage_text_field="passage_text",
                passage_chunk_field="passage_chunk",
                embedding_field="passage_embedding",
            )
            self.assertFalse(result)
            mock_print.assert_called_with(
                "Error verifying or creating index: Connection error"
            )

    @patch("opensearchpy.helpers.bulk")
    def test_bulk_index_success(self, mock_bulk):
        mock_bulk.return_value = (100, [])
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{"index": {"_index": "test-index", "_id": i}} for i in range(100)]
        with patch("builtins.print") as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 100)
            self.assertEqual(error_count, 0)
            mock_print.assert_called_with(
                "Indexed 100 documents successfully. Failed to index 0 documents."
            )

    @patch("opensearchpy.helpers.bulk")
    def test_bulk_index_with_errors(self, mock_bulk):
        mock_bulk.return_value = (90, [{"index": {"_id": "10", "error": "Some error"}}])
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{"index": {"_index": "test-index", "_id": i}} for i in range(100)]
        with patch("builtins.print") as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 90)
            self.assertEqual(error_count, 1)
            mock_print.assert_called_with(
                "Indexed 90 documents successfully. Failed to index 1 documents."
            )

    @patch("opensearchpy.helpers.bulk")
    def test_bulk_index_exception(self, mock_bulk):
        mock_bulk.side_effect = Exception("Bulk indexing error")
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        actions = [{"index": {"_index": "test-index", "_id": i}} for i in range(100)]
        with patch("builtins.print") as mock_print:
            success_count, error_count = connector.bulk_index(actions)
            self.assertEqual(success_count, 0)
            self.assertEqual(error_count, 100)
            mock_print.assert_called_with(
                "Error during bulk indexing: Bulk indexing error"
            )

    def test_search_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Mock search response
        self.mock_opensearch_client.search.return_value = {
            "hits": {"hits": [{"id": 1}, {"id": 2}]}
        }
        results = connector.search(query_text="test", model_id="model-123", k=5)
        self.assertEqual(results, [{"id": 1}, {"id": 2}])
        self.mock_opensearch_client.search.assert_called_once()

    def test_search_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.side_effect = Exception("Search error")
        with patch("builtins.print") as mock_print:
            results = connector.search(query_text="test", model_id="model-123", k=5)
            self.assertEqual(results, [])
            mock_print.assert_called_with("Error during search: Search error")

    def test_search_by_vector_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.return_value = {
            "hits": {"hits": [{"id": 1}, {"id": 2}]}
        }
        results = connector.search_by_vector(vector=[0.1, 0.2, 0.3], k=5)
        self.assertEqual(results, [{"id": 1}, {"id": 2}])
        self.mock_opensearch_client.search.assert_called_once()

    def test_search_by_vector_exception(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.search.side_effect = Exception(
            "Vector search error"
        )
        with patch("builtins.print") as mock_print:
            results = connector.search_by_vector(vector=[0.1, 0.2, 0.3], k=5)
            self.assertEqual(results, [])
            mock_print.assert_called_with("Error during search: Vector search error")

    def test_check_connection_success(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        # Mock info method
        self.mock_opensearch_client.info.return_value = {
            "version": {"number": "7.10.2"}
        }
        result = connector.check_connection()
        self.assertTrue(result)
        self.mock_opensearch_client.info.assert_called_once()

    def test_check_connection_failure(self):
        connector = OpenSearchConnector(self.config)
        connector.opensearch_client = self.mock_opensearch_client
        self.mock_opensearch_client.info.side_effect = Exception("Connection error")
        with patch("builtins.print") as mock_print:
            result = connector.check_connection()
            self.assertFalse(result)
            mock_print.assert_called_with(
                "Error connecting to OpenSearch: Connection error"
            )


if __name__ == "__main__":
    unittest.main()
