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

from opensearch_py_ml.ml_commons.cli.model_predict import Predict


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.predict = Predict()
        self.config_path = "test_config.yml"
        self.valid_config = {
            "service_type": "amazon-opensearch-service",
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
    @patch.object(Predict, "load_config")
    def test_predict_command_no_config(self, mock_load_config, mock_print):
        """Test predict_command with no configuration"""
        # Setup
        mock_load_config.return_value = None

        # Execute
        result = self.predict.predict_command(self.config_path)

        # Verify
        self.assertFalse(result)

    @patch("builtins.print")
    @patch.object(Predict, "load_config")
    def test_predict_command_no_endpoint(self, mock_load_config, mock_print):
        """Test predict_command with no OpenSearch endpoint"""
        # Setup
        config = self.valid_config.copy()
        config["opensearch_config"]["opensearch_domain_endpoint"] = ""
        mock_load_config.return_value = config
        self.predict.config = config

        # Execute
        result = self.predict.predict_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("builtins.print")
    @patch.object(Predict, "load_config")
    def test_predict_command_opensource_no_credentials(
        self, mock_load_config, mock_print
    ):
        """Test predict_command with open-source and no credentials"""
        # Setup
        config = self.valid_config.copy()
        config["service_type"] = "open-source"
        config["opensearch_config"]["opensearch_domain_username"] = ""
        mock_load_config.return_value = config
        self.predict.config = config

        # Execute
        result = self.predict.predict_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("builtins.print")
    @patch.object(Predict, "load_config")
    @patch.object(Predict, "get_opensearch_domain_name")
    def test_predict_command_aws_no_region(
        self, mock_get_domain, mock_load_config, mock_print
    ):
        """Test predict_command with AWS service and no region"""
        # Setup
        config = self.valid_config.copy()
        config["opensearch_config"]["opensearch_domain_region"] = ""
        mock_load_config.return_value = config
        mock_get_domain.return_value = None
        self.predict.config = config

        # Execute
        result = self.predict.predict_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("builtins.print")
    @patch(
        "builtins.input",
        side_effect=["test-model-id", '{"test": "payload"}', "", "", "no"],
    )
    @patch.object(Predict, "load_config")
    @patch.object(Predict, "get_opensearch_domain_name")
    @patch("boto3.client")
    @patch("requests.post")
    def test_predict_command_with_input(
        self,
        mock_requests_post,
        mock_boto3_client,
        mock_get_domain,
        mock_load_config,
        mock_input,
        mock_print,
    ):
        """Test successful predict_command with user input"""
        # Setup
        self.predict.config = self.valid_config
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

        prediction_response = (
            '{"inference_results": [{"status_code": 200, "result": "success"}]}'
        )

        # Mock the requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = prediction_response
        mock_response.json.return_value = json.loads(prediction_response)
        mock_requests_post.return_value = mock_response

        # Execute
        result = self.predict.predict_command(self.config_path)

        # Verify
        self.assertTrue(result)
        mock_requests_post.assert_called_once()

    @patch.object(Predict, "load_config")
    @patch.object(Predict, "get_opensearch_domain_name")
    @patch("boto3.client")
    @patch("requests.post")
    @patch("builtins.input", side_effect=["no"])
    def test_predict_command_with_params(
        self,
        mock_input,
        mock_requests_post,
        mock_boto3_client,
        mock_get_domain,
        mock_load_config,
    ):
        """Test predict_command with provided model_id and payload"""
        # Setup
        self.predict.config = self.valid_config
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
        payload = '{"test": "payload"}'
        prediction_response = (
            '{"inference_results": [{"status_code": 200, "result": "success"}]}'
        )

        # Mock the requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = prediction_response
        mock_response.json.return_value = json.loads(prediction_response)
        mock_requests_post.return_value = mock_response

        # Execute
        result = self.predict.predict_command(self.config_path, model_id, payload)

        # Verify
        self.assertTrue(result)
        mock_requests_post.assert_called_once()

    @patch("builtins.print")
    @patch.object(Predict, "load_config")
    @patch.object(Predict, "get_opensearch_domain_name")
    @patch("requests.post")
    def test_predict_command_failure_response(
        self, mock_requests_post, mock_get_domain, mock_load_config, mock_print
    ):
        """Test predict_command with failed prediction"""
        # Setup
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"
        self.predict.config = self.valid_config

        prediction_response = '{"error": "failed"}'
        # Mock the requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = prediction_response
        mock_response.json.return_value = json.loads(prediction_response)
        mock_requests_post.return_value = mock_response

        # Execute
        result = self.predict.predict_command(
            self.config_path, model_id="test-model-id", payload='{"key": "value"}'
        )

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"{Fore.RED}Failed to predict model.{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    unittest.main()
