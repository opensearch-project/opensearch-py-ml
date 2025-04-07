# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel import BedrockModel


class TestBedrockModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.bedrock_model = BedrockModel(opensearch_domain_region=self.region)
        self.connector_role_prefix = "test_role"

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_cohere(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Bedrock connector with Cohere embedding model"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Cohere embedding model",
        )

        # Verify method cals
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock", "amazon-opensearch-service", "Cohere embedding model"
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_titan(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Bedrock connector with Titan embedding model"""
        # Setup mocks
        mock_get_model_details.return_value = "2"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Titan embedding model",
        )

        # Verify method cals
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock", "amazon-opensearch-service", "Titan embedding model"
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_custom_model(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Bedrock connector with custom model"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )
        custom_payload = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            model_arn="test-model-arn",
            connector_body=custom_payload,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Bedrock", "amazon-opensearch-service", "Custom model"
        )
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
        )
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1", "test_prefix", ""])
    def test_create_connector_default_region(self, mock_input):
        """Test creating a Bedrock connector with default region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your Bedrock region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-west-2")

    @patch("builtins.input", side_effect=["1", "test_prefix", "us-east-1"])
    def test_create_connector_custom_region(self, mock_input):
        """Test creating a Bedrock connector with a custom region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your Bedrock region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-east-1")

    @patch("builtins.input", side_effect=["3", "test_prefix", "test-model-arn"])
    def test_enter_custom_model_arn(self, mock_input):
        """Test creating a Bedrock connector with a custom model ARN"""
        payload = {"test": "payload"}
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_body=payload,
        )
        mock_input.assert_any_call("Enter your custom model ARN: ")
        connector_role_inline_policy = (
            self.mock_helper.create_connector_with_role.call_args[0][0]
        )
        self.assertEqual(
            connector_role_inline_policy["Statement"][0]["Resource"], "test-model-arn"
        )

    @patch("builtins.input")
    def test_input_custom_model_details(self, mock_input):
        """Test create_connector for input_custom_model_details method"""
        mock_input.side_effect = [
            '{"name": "test-model",',
            '"description": "test description",',
            '"parameters": {"param": "value"}}',
            "",
        ]
        result = self.bedrock_model.input_custom_model_details()
        expected_result = {
            "name": "test-model",
            "description": "test description",
            "parameters": {"param": "value"},
        }
        self.assertEqual(result, expected_result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice(self, mock_print, mock_get_model_details):
        """Test creating a Bedrock connector with an invalid model choice"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            model_arn="test-arn",
            connector_body={"name": "test-model"},
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )

    def test_create_connector_failure(self):
        """Test creating a Bedrock connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None
        result = self.bedrock_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Cohere embedding model",
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
