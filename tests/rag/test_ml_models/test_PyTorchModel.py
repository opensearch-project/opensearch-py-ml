# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, mock_open, patch

from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.PyTorchModel import (
    CustomPyTorchModel,
)


class TestCustomPyTorchModel(unittest.TestCase):

    def setUp(self):
        self.aws_region = "us-west-2"
        self.opensearch_domain_name = "test-domain"
        self.opensearch_username = "test-user"
        self.opensearch_password = "test-password"
        self.mock_iam_role_helper = Mock()
        self.custom_pytorch_model = CustomPyTorchModel(
            self.aws_region,
            self.opensearch_domain_name,
            self.opensearch_username,
            self.opensearch_password,
            self.mock_iam_role_helper,
        )

    def test_init(self):
        self.assertEqual(self.custom_pytorch_model.aws_region, self.aws_region)
        self.assertEqual(
            self.custom_pytorch_model.opensearch_domain_name,
            self.opensearch_domain_name,
        )
        self.assertEqual(
            self.custom_pytorch_model.opensearch_username, self.opensearch_username
        )
        self.assertEqual(
            self.custom_pytorch_model.opensearch_password, self.opensearch_password
        )
        self.assertEqual(
            self.custom_pytorch_model.iam_role_helper, self.mock_iam_role_helper
        )

    @patch("builtins.input", side_effect=["1", "/path/to/model.pt"])
    @patch("os.path.isfile", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"model_content")
    def test_register_custom_pytorch_model_default(
        self, mock_file, mock_isfile, mock_input
    ):
        mock_opensearch_client = Mock()
        mock_opensearch_client.transport.perform_request.side_effect = [
            {"model_id": "uploaded_model_id"},
            {"task_id": "test-task-id"},
            {"state": "COMPLETED", "model_id": "registered_model_id"},
            {},  # for model deployment
        ]

        mock_config = {"embedding_dimension": 768}
        mock_save_config = Mock()

        self.custom_pytorch_model.register_custom_pytorch_model(
            mock_opensearch_client, mock_config, mock_save_config
        )

        self.assertEqual(mock_opensearch_client.transport.perform_request.call_count, 4)
        mock_save_config.assert_called_once_with(
            {"embedding_dimension": 768, "embedding_model_id": "registered_model_id"}
        )

    @patch(
        "builtins.input",
        side_effect=[
            "2",
            "/path/to/model.pt",
            '{"name": "custom_model", "model_format": "TORCH_SCRIPT", "model_config": {"embedding_dimension": 512, "framework_type": "CUSTOM", "model_type": "bert"}, "description": "Custom model"}',
        ],
    )
    @patch("os.path.isfile", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"model_content")
    def test_register_custom_pytorch_model_custom(
        self, mock_file, mock_isfile, mock_input
    ):
        mock_opensearch_client = Mock()
        mock_opensearch_client.transport.perform_request.side_effect = [
            {"model_id": "uploaded_model_id"},
            {"task_id": "test-task-id"},
            {"state": "COMPLETED", "model_id": "registered_model_id"},
            {},  # for model deployment
        ]

        mock_config = {}
        mock_save_config = Mock()

        self.custom_pytorch_model.register_custom_pytorch_model(
            mock_opensearch_client, mock_config, mock_save_config
        )

        self.assertEqual(mock_opensearch_client.transport.perform_request.call_count, 4)
        mock_save_config.assert_called_once_with(
            {"embedding_model_id": "registered_model_id"}
        )

    @patch("builtins.input", side_effect=["1", "/nonexistent/path.pt"])
    @patch("os.path.isfile", return_value=False)
    def test_register_custom_pytorch_model_file_not_found(
        self, mock_isfile, mock_input
    ):
        mock_opensearch_client = Mock()
        mock_config = {}
        mock_save_config = Mock()

        self.custom_pytorch_model.register_custom_pytorch_model(
            mock_opensearch_client, mock_config, mock_save_config
        )

        mock_opensearch_client.transport.perform_request.assert_not_called()
        mock_save_config.assert_not_called()

    @patch("builtins.input", return_value="3")
    def test_register_custom_pytorch_model_invalid_choice(self, mock_input):
        mock_opensearch_client = Mock()
        mock_config = {}
        mock_save_config = Mock()

        self.custom_pytorch_model.register_custom_pytorch_model(
            mock_opensearch_client, mock_config, mock_save_config
        )

        mock_opensearch_client.transport.perform_request.assert_not_called()
        mock_save_config.assert_not_called()

    def test_get_custom_json_input_valid(self):
        with patch("builtins.input", return_value='{"key": "value"}'):
            result = self.custom_pytorch_model.get_custom_json_input()
            self.assertEqual(result, {"key": "value"})

    def test_get_custom_json_input_invalid(self):
        with patch("builtins.input", return_value="invalid json"):
            result = self.custom_pytorch_model.get_custom_json_input()
            self.assertIsNone(result)

    @patch("time.time", side_effect=[0, 10, 20, 30])
    @patch("time.sleep", return_value=None)
    def test_wait_for_model_registration_success(self, mock_sleep, mock_time):
        mock_opensearch_client = Mock()
        mock_opensearch_client.transport.perform_request.side_effect = [
            {"state": "RUNNING"},
            {"state": "RUNNING"},
            {"state": "COMPLETED", "model_id": "test-model-id"},
        ]

        result = self.custom_pytorch_model.wait_for_model_registration(
            mock_opensearch_client, "test-task-id"
        )
        self.assertEqual(result, "test-model-id")

    @patch("time.time", side_effect=[0, 10, 20, 30])
    @patch("time.sleep", return_value=None)
    def test_wait_for_model_registration_failure(self, mock_sleep, mock_time):
        mock_opensearch_client = Mock()
        mock_opensearch_client.transport.perform_request.side_effect = [
            {"state": "RUNNING"},
            {"state": "FAILED"},
        ]

        result = self.custom_pytorch_model.wait_for_model_registration(
            mock_opensearch_client, "test-task-id"
        )
        self.assertIsNone(result)

    @patch("time.time", side_effect=[0, 1000])
    @patch("time.sleep", return_value=None)
    def test_wait_for_model_registration_timeout(self, mock_sleep, mock_time):
        mock_opensearch_client = Mock()
        mock_opensearch_client.transport.perform_request.return_value = {
            "state": "RUNNING"
        }

        result = self.custom_pytorch_model.wait_for_model_registration(
            mock_opensearch_client, "test-task-id", timeout=5
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
