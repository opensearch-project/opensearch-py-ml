# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.bedrock_model import BedrockModel


class TestBedrockModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.service_type = "amazon-opensearch-service"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.bedrock_model = BedrockModel(
            opensearch_domain_region=self.region, service_type=self.service_type
        )
        self.connector_role_prefix = "test_role"
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
    def test_create_connector_cohere_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Bedrock connector with Cohere embedding model in managed service"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Cohere Embed Model v3 - English",
        )

        # Verify method cals
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock",
            "amazon-opensearch-service",
            "Cohere Embed Model v3 - English",
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
        """Test creating a Bedrock connector with custom model in managed service"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = self.connector_body
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            model_arn="test-model-arn",
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock", "amazon-opensearch-service", "Custom model"
        )
        mock_custom_model.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value=["access_key", "secret_key", "session_token"],
    )
    def test_create_connector_titan_open_source(
        self,
        mock_get_password,
        mock_get_model_details,
    ):
        """Test creating a Bedrock connector with Titan embedding model in open-source service"""
        # Create model with open-source service type
        open_source_model = BedrockModel(
            opensearch_domain_region=self.region, service_type="open-source"
        )
        mock_get_model_details.return_value = "2"
        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Titan Text Embedding",
        )

        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock", "open-source", "Titan Text Embedding"
        )
        # Verify that create_connector was called instead of create_connector_with_role
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "",
        )
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
        )
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1", "", "test_prefix"])
    def test_create_connector_default_region(self, mock_input):
        """Test creating a Bedrock connector with default region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your AWS region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-west-2")

    @patch("builtins.input", side_effect=["2", "us-east-1", "test_prefix"])
    def test_create_connector_custom_region(self, mock_input):
        """Test creating a Bedrock connector with a custom region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your AWS region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-east-1")

    @patch("builtins.input", side_effect=["9", "", "test_prefix", "test_model_arn"])
    def test_enter_custom_model_arn(self, mock_input):
        """Test creating a Bedrock connector with a custom model ARN"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_body=self.connector_body,
        )
        mock_input.assert_any_call("Enter your custom model ARN: ")
        connector_role_inline_policy = (
            self.mock_helper.create_connector_with_role.call_args[0][0]
        )
        self.assertEqual(
            connector_role_inline_policy["Statement"][0]["Resource"], "test_model_arn"
        )

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating a Bedrock connector with an invalid model choice"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "",
        )
        mock_custom_model.return_value = self.connector_body
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once()

    def test_create_connector_failure(self):
        """Test creating a Bedrock connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None, None
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Anthropic Claude v2",
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
