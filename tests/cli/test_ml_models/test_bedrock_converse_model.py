# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.bedrock_converse_model import (
    BedrockConverseModel,
)


class TestBedrockConverseModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.service_type = "amazon-opensearch-service"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.bedrock_converse_model = BedrockConverseModel(
            opensearch_domain_region=self.region, service_type=self.service_type
        )
        self.connector_role_prefix = "test_role"
        self.model_arn = "test_model_arn"
        self.connector_body = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
            "parameters": {"model": "test_model"},
        }

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_claude_model_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Bedrock Converse connector with Claude model in managed service"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.bedrock_converse_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Anthropic Claude 3 Sonnet",
            model_arn=self.model_arn,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock Converse",
            "amazon-opensearch-service",
            "Anthropic Claude 3 Sonnet",
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_custom_model_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint, mock_custom_model
    ):
        """Test creating a Bedrock Converse connector with custom model in managed service"""
        # Setup mocks
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = self.connector_body

        result = self.bedrock_converse_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            model_arn=self.model_arn,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock Converse", "amazon-opensearch-service", "Custom model"
        )
        mock_custom_model.assert_called_once_with()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice_managed(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating a Bedrock Converse connector with an invalid model choice in managed service"""
        # Setup mocks
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = self.connector_body

        self.bedrock_converse_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            model_arn=self.model_arn,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with()

    def test_create_connector_failure(self):
        """Test creating a Bedrock Converse connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None, None
        result = self.bedrock_converse_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            region=self.region,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Anthropic Claude 3 Sonnet",
            model_arn=self.model_arn,
        )
        self.assertFalse(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value=["access_key", "secret_key", "session_token"],
    )
    def test_create_connector_claude_model_open_source(
        self, mock_get_password, mock_get_model_details
    ):
        """Test creating a Bedrock Converse connector with Claude model in open-source service"""
        # Create model with open-source service type
        open_source_model = BedrockConverseModel(
            opensearch_domain_region=self.region, service_type="open-source"
        )
        mock_get_model_details.return_value = "1"

        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Anthropic Claude 3 Sonnet",
            model_arn=self.model_arn,
        )

        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock Converse", "open-source", "Anthropic Claude 3 Sonnet"
        )
        # Verify that create_connector was called instead of create_connector_with_role
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value=["access_key", "secret_key", "session_token"],
    )
    def test_create_connector_custom_model_open_source(
        self, mock_get_password, mock_get_model_details, mock_custom_model
    ):
        """Test creating a Bedrock Converse connector with custom model in open-source service"""
        # Create model with open-source service type
        open_source_model = BedrockConverseModel(
            opensearch_domain_region=self.region, service_type="open-source"
        )
        mock_custom_model.return_value = self.connector_body
        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            model_arn=self.model_arn,
        )

        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock Converse", "open-source", "Custom model"
        )
        # Verify that create_connector was called instead of create_connector_with_role
        self.mock_helper.create_connector.assert_called_once()
        mock_custom_model.assert_called_once_with()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value=["access_key", "secret_key", "session_token"],
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice_open_source(
        self, mock_print, mock_get_password, mock_get_model_details, mock_custom_model
    ):
        """Test creating a Bedrock Converse connector with an invalid model choice in open-source service"""
        # Create model with open-source service type
        open_source_model = BedrockConverseModel(
            opensearch_domain_region=self.region, service_type="open-source"
        )
        mock_custom_model.return_value = self.connector_body

        open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            model_arn=self.model_arn,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with()

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        # Setup mock
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.bedrock_converse_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_arn=self.model_arn,
        )

        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
