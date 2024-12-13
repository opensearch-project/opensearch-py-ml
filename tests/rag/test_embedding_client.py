# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, patch

from opensearch_py_ml.ml_commons.rag_pipeline.rag.embedding_client import (
    EmbeddingClient,
)


class TestEmbeddingClient(unittest.TestCase):

    def setUp(self):
        self.mock_opensearch_client = MagicMock()
        self.embedding_model_id = "test_model_id"
        self.client = EmbeddingClient(
            self.mock_opensearch_client, self.embedding_model_id
        )

    def test_initialization(self):
        self.assertEqual(self.client.opensearch_client, self.mock_opensearch_client)
        self.assertEqual(self.client.embedding_model_id, self.embedding_model_id)

    @patch("time.sleep")
    def test_get_text_embedding_success(self, mock_sleep):
        mock_response = {"inference_results": [{"output": [{"data": [0.1, 0.2, 0.3]}]}]}
        self.mock_opensearch_client.transport.perform_request.return_value = (
            mock_response
        )

        result = self.client.get_text_embedding("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        self.mock_opensearch_client.transport.perform_request.assert_called_once_with(
            method="POST",
            url=f"/_plugins/_ml/_predict/text_embedding/{self.embedding_model_id}",
            body={"text_docs": ["test text"]},
        )

    @patch("time.sleep")
    def test_get_text_embedding_no_results(self, mock_sleep):
        mock_response = {"inference_results": []}
        self.mock_opensearch_client.transport.perform_request.return_value = (
            mock_response
        )

        result = self.client.get_text_embedding("test text")
        self.assertIsNone(result)

    @patch("time.sleep")
    def test_get_text_embedding_unexpected_format(self, mock_sleep):
        mock_response = {"inference_results": [{"output": "unexpected"}]}
        self.mock_opensearch_client.transport.perform_request.return_value = (
            mock_response
        )

        result = self.client.get_text_embedding("test text")
        self.assertIsNone(result)

    @patch("time.sleep")
    def test_get_text_embedding_retry_success(self, mock_sleep):
        self.mock_opensearch_client.transport.perform_request.side_effect = [
            Exception("Test error"),
            {"inference_results": [{"output": [{"data": [0.1, 0.2, 0.3]}]}]},
        ]

        result = self.client.get_text_embedding("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(
            self.mock_opensearch_client.transport.perform_request.call_count, 2
        )

    @patch("time.sleep")
    def test_get_text_embedding_max_retries_exceeded(self, mock_sleep):
        self.mock_opensearch_client.transport.perform_request.side_effect = Exception(
            "Test error"
        )

        with self.assertRaises(Exception):
            self.client.get_text_embedding("test text", max_retries=3)

        self.assertEqual(
            self.mock_opensearch_client.transport.perform_request.call_count, 3
        )

    @patch("time.sleep")
    def test_get_text_embedding_alternative_output_format(self, mock_sleep):
        mock_response = {"inference_results": [{"output": {"data": [0.1, 0.2, 0.3]}}]}
        self.mock_opensearch_client.transport.perform_request.return_value = (
            mock_response
        )

        result = self.client.get_text_embedding("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
