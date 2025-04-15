# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.model_manager import ModelManager


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.model_manager = ModelManager()
        self.config_path = "test_config.yml"
        self.valid_config = {
            "service_type": ModelManager.AMAZON_OPENSEARCH_SERVICE,
            "opensearch_config": {
                "opensearch_domain_endpoint": "https://test-endpoint",
                "opensearch_domain_region": "us-west-2",
                "opensearch_domain_username": "admin",
                "opensearch_domain_password": "password",
            },
            "aws_credentials": {
                "aws_role_name": "test-role",
                "aws_user_name": "test-user",
                "aws_access_key": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": "test-token",
            },
        }

    @patch("builtins.print")
    @patch.object(ModelManager, "load_config")
    def test_initialize_predict_model_no_config(self, mock_load_config, mock_print):
        """Test initialize_predict_model with no configuration"""
        # Setup
        mock_load_config.return_value = None

        # Execute
        result = self.model_manager.initialize_predict_model(self.config_path)

        # Verify
        self.assertFalse(result)

    @patch("builtins.print")
    @patch(
        "builtins.input",
        side_effect=["test-model-id", '{"test": "payload"}', "", "", "no"],
    )
    @patch.object(ModelManager, "load_config")
    @patch.object(ModelManager, "get_opensearch_domain_name")
    @patch("boto3.client")
    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper.OpenSearch")
    def test_initialize_predict_model_with_input(
        self,
        mock_opensearch,
        mock_boto3_client,
        mock_get_domain,
        mock_load_config,
        mock_input,
        mock_print,
    ):
        """Test successful initialize_predict_model with user input"""
        # Setup
        self.model_manager.config = self.valid_config
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"

        mock_domain_response = {
            "DomainStatus": {
                "Endpoint": "test-endpoint",
                "ARN": "test-domain-arn",
            }
        }
        mock_opensearch_client = Mock()
        mock_opensearch_client.describe_domain.return_value = mock_domain_response
        mock_boto3_client.return_value = mock_opensearch_client

        mock_response = {
            "inference_results": [{"status_code": 200, "result": "success"}]
        }

        # Mock OpenSearch client and its transport
        mock_opensearch_instance = MagicMock()
        mock_transport = MagicMock()
        mock_opensearch_instance.transport = mock_transport
        mock_opensearch.return_value = mock_opensearch_instance

        # Mock transport.perform_request response
        mock_transport.perform_request.return_value = mock_response

        # Execute
        result = self.model_manager.initialize_predict_model(self.config_path)

        # Verify
        self.assertTrue(result)

        # Verify that perform_request was called with correct arguments
        mock_transport.perform_request.assert_called_once_with(
            method="POST",
            url="/_plugins/_ml/models/test-model-id/_predict",
            body={"test": "payload"},
            headers={"Content-Type": "application/json"},
        )

    @patch.object(ModelManager, "predict_model_output")
    @patch.object(ModelManager, "load_config")
    @patch.object(ModelManager, "get_opensearch_domain_name")
    @patch("boto3.client")
    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper.OpenSearch")
    @patch("builtins.input", side_effect=["yes"])
    def test_initialize_predict_model_with_params(
        self,
        mock_input,
        mock_opensearch,
        mock_boto3_client,
        mock_get_domain,
        mock_load_config,
        mock_predict_output,
    ):
        """Test initialize_predict_model with provided model_id and payload"""
        # Setup
        self.model_manager.config = self.valid_config
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"

        mock_domain_response = {
            "DomainStatus": {
                "Endpoint": "test-endpoint",
                "ARN": "test-domain-arn",
            }
        }
        mock_opensearch_client = Mock()
        mock_opensearch_client.describe_domain.return_value = mock_domain_response
        mock_boto3_client.return_value = mock_opensearch_client

        model_id = "test-model-id"
        body = '{"test": "body"}'
        mock_response = {
            "inference_results": [{"status_code": 200, "result": "success"}]
        }

        # Mock OpenSearch client and its transport
        mock_opensearch_instance = MagicMock()
        mock_transport = MagicMock()
        mock_opensearch_instance.transport = mock_transport
        mock_opensearch.return_value = mock_opensearch_instance

        # Mock transport.perform_request response
        mock_transport.perform_request.return_value = mock_response

        # Execute
        result = self.model_manager.initialize_predict_model(
            self.config_path, model_id, body
        )
        # Verify result
        self.assertTrue(result)

        # Verify that perform_request was called with correct arguments
        mock_transport.perform_request.assert_called_once_with(
            method="POST",
            url="/_plugins/_ml/models/test-model-id/_predict",
            body=json.loads(body),
            headers={"Content-Type": "application/json"},
        )
        mock_predict_output.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.model_manager.logger")
    @patch.object(ModelManager, "load_config")
    @patch.object(ModelManager, "get_opensearch_domain_name")
    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper.OpenSearch")
    def test_initialize_predict_model_failure_response(
        self, mock_opensearch, mock_get_domain, mock_load_config, mock_logger
    ):
        """Test initialize_predict_model with failed prediction"""
        # Setup
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"
        self.model_manager.config = self.valid_config

        mock_response = {"error": "failed"}

        # Mock OpenSearch client and its transport
        mock_opensearch_instance = MagicMock()
        mock_transport = MagicMock()
        mock_opensearch_instance.transport = mock_transport
        mock_opensearch.return_value = mock_opensearch_instance

        # Mock transport.perform_request response
        mock_transport.perform_request.return_value = mock_response

        # Execute
        result = self.model_manager.initialize_predict_model(
            self.config_path, model_id="test-model-id", body='{"key": "value"}'
        )

        # Verify
        self.assertFalse(result)
        mock_logger.warning.assert_called_with(
            f"{Fore.RED}Failed to predict model.{Style.RESET_ALL}"
        )

    @patch("opensearch_py_ml.ml_commons.cli.model_manager.logger")
    def test_initialize_predict_model_exception_handling(self, mock_logger):
        """Test initialize_predict_model for exception handling"""
        self.model_manager.load_config = MagicMock(side_effect=Exception("Test error"))
        result = self.model_manager.initialize_predict_model(self.config_path)
        mock_logger.error.assert_called_with(
            f"{Fore.RED}Error predicting model: Test error{Style.RESET_ALL}"
        )
        self.assertFalse(result)

    @patch("builtins.print")
    @patch.object(ModelManager, "load_config")
    def test_initialize_register_model_no_config(self, mock_load_config, mock_print):
        """Test initialize_register_model with no configuration"""
        # Setup
        mock_load_config.return_value = None

        # Execute
        result = self.model_manager.initialize_register_model(self.config_path)

        # Verify
        self.assertFalse(result)

    @patch(
        "builtins.input",
        side_effect=["test-model", "test-model-description", "test-connector-id"],
    )
    @patch.object(ModelManager, "load_and_check_config")
    @patch.object(ModelManager, "get_opensearch_domain_name")
    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper.OpenSearch")
    def test_initialize_register_model_with_input(
        self, mock_opensearch, mock_get_domain, mock_load_check_config, mock_input
    ):
        """Test successful initialize_register_model with user input"""
        # Setup
        mock_ai_helper = MagicMock()
        mock_ai_helper.register_model.return_value = "test-model-id"

        # Mock load_and_check_config to return a tuple with mock_ai_helper
        mock_load_check_config.return_value = (mock_ai_helper, None, None, None)

        # Execute
        result = self.model_manager.initialize_register_model(self.config_path)

        # Verify result
        self.assertTrue(result)

        # Verify that register_model was called with correct arguments
        mock_ai_helper.register_model.assert_called_once_with(
            "test-model", "test-model-description", "test-connector-id"
        )

    @patch("builtins.input", side_effect=[""])
    @patch.object(ModelManager, "load_and_check_config")
    @patch.object(ModelManager, "get_opensearch_domain_name")
    @patch("opensearch_py_ml.ml_commons.cli.ai_connector_helper.OpenSearch")
    def test_initialize_register_model_with_params(
        self, mock_opensearch, mock_get_domain, mock_load_check_config, mock_input
    ):
        """Test successful initialize_register_model with provided connector_id, model_name, and model_description"""
        # Setup
        mock_ai_helper = MagicMock()
        mock_ai_helper.register_model.return_value = "test-model-id"

        # Mock load_and_check_config to return a tuple with mock_ai_helper
        mock_load_check_config.return_value = (mock_ai_helper, None, None, None)

        connector_id = "test-connector-id"
        model_name = "test-model"
        model_description = "test-model-description"

        # Execute
        result = self.model_manager.initialize_register_model(
            self.config_path, connector_id, model_name, model_description
        )

        # Verify result
        self.assertTrue(result)

        # Verify that register_model was called with correct arguments
        mock_ai_helper.register_model.assert_called_once_with(
            model_name, model_description, connector_id
        )

    def test_initialize_register_model_exception_handling(self):
        """Test initialize_register_model for exception handling"""
        self.model_manager.load_config = MagicMock(side_effect=Exception("Test error"))
        result = self.model_manager.initialize_register_model(self.config_path)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
