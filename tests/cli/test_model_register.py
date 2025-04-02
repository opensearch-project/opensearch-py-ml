# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from unittest.mock import MagicMock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.model_register import Register


class TestRegister(unittest.TestCase):
    def setUp(self):
        self.register = Register()
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
    @patch.object(Register, "load_config")
    def test_register_command_no_config(self, mock_load_config, mock_print):
        """Test register_command with no configuration"""
        # Setup
        mock_load_config.return_value = None

        # Execute
        result = self.register.register_command(self.config_path)

        # Verify
        self.assertFalse(result)

    @patch("builtins.print")
    @patch.object(Register, "load_config")
    def test_register_command_no_endpoint(self, mock_load_config, mock_print):
        """Test register_command with no OpenSearch endpoint"""
        # Setup
        config = self.valid_config.copy()
        config["opensearch_config"]["opensearch_domain_endpoint"] = ""
        mock_load_config.return_value = config
        self.register.config = config

        # Execute
        result = self.register.register_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("builtins.print")
    @patch.object(Register, "load_config")
    def test_register_command_opensource_no_credentials(
        self, mock_load_config, mock_print
    ):
        """Test register_command with open-source and no credentials"""
        # Setup
        config = self.valid_config.copy()
        config["service_type"] = "open-source"
        config["opensearch_config"]["opensearch_domain_username"] = ""
        mock_load_config.return_value = config
        self.register.config = config

        # Execute
        result = self.register.register_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("builtins.print")
    @patch.object(Register, "load_config")
    @patch.object(Register, "get_opensearch_domain_name")
    def test_register_command_aws_no_region(
        self, mock_get_domain, mock_load_config, mock_print
    ):
        """Test register_command with AWS service and no region"""
        # Setup
        config = self.valid_config.copy()
        config["opensearch_config"]["opensearch_domain_region"] = ""
        mock_load_config.return_value = config
        mock_get_domain.return_value = None
        self.register.config = config

        # Execute
        result = self.register.register_command(self.config_path)

        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with(
            f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch(
        "builtins.input",
        side_effect=["test-connector-id", "test-model", "test-model-description"],
    )
    @patch.object(Register, "load_config")
    @patch.object(Register, "get_opensearch_domain_name")
    @patch("requests.post")
    def test_register_command_with_input(
        self, mock_requests_post, mock_get_domain, mock_load_config, mock_input
    ):
        """Test successful register_command with user input"""
        # Setup
        self.register.config = self.valid_config
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"

        register_response = '{"model_id": "test-model-id"}'

        # Mock the requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = register_response
        mock_response.json.return_value = json.loads(register_response)
        mock_requests_post.return_value = mock_response

        # Execute
        result = self.register.register_command(self.config_path)

        # Verify
        self.assertTrue(result)
        mock_requests_post.assert_called_once()

    @patch("builtins.input", side_effect=[""])
    @patch.object(Register, "load_config")
    @patch.object(Register, "get_opensearch_domain_name")
    @patch("requests.post")
    def test_register_command_successful_with_params(
        self, mock_requests_post, mock_get_domain, mock_load_config, mock_input
    ):
        """Test successful register_command with provided connector_id, model_name, and model_description"""
        # Setup
        self.register.config = self.valid_config
        mock_load_config.return_value = self.valid_config
        mock_get_domain.return_value = "test-domain"

        connector_id = "test-connector-id"
        model_name = "test-model"
        model_description = "test-model-description"
        register_response = '{"model_id": "test-model-id"}'

        # Mock the requests.post response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = register_response
        mock_response.json.return_value = json.loads(register_response)
        mock_requests_post.return_value = mock_response

        # Execute
        result = self.register.register_command(
            self.config_path, connector_id, model_name, model_description
        )

        # Verify
        self.assertTrue(result)
        mock_requests_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
