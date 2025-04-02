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

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel.uuid")
    def test_create_bedrock_connector_cohere(self, mock_uuid):
        """Test creating a Bedrock connector with Cohere embedding model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Cohere embedding model",
        )

        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_role.assert_called_once()
        call_args = self.mock_helper.create_connector_with_role.call_args[0]

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_bedrock_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_bedrock_connector_create_12345678",
        )

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(
            connector_payload["name"], "Amazon Bedrock Cohere Connector: embedding v3"
        )
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)
        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel.uuid")
    def test_create_bedrock_connector_titan(self, mock_uuid):
        """Test creating a Bedrock connector with Titan embedding model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Titan embedding model",
        )

        call_args = self.mock_helper.create_connector_with_role.call_args[0]
        connector_payload = call_args[3]
        self.assertEqual(
            connector_payload["name"], "Amazon Bedrock Connector: titan embedding v1"
        )

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_bedrock_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_bedrock_connector_create_12345678",
        )
        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel.uuid")
    def test_create_bedrock_connector_custom(self, mock_uuid):
        """Test creating a Bedrock connector with custom model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        custom_arn = "arn:aws:bedrock:region:account:model/custom-model"
        connector_payload = {
            "name": "Custom Bedrock Connector",
            "description": "Test custom connector",
            "version": 1,
            "protocol": "aws_sigv4",
            "parameters": {"region": self.region, "service_name": "bedrock"},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://bedrock-runtime.region.amazonaws.com/model/custom-model/invoke",
                    "headers": {
                        "content-type": "application/json",
                        "x-amz-content-sha256": "required",
                    },
                }
            ],
        }

        result = self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Custom model",
            model_arn=custom_arn,
            connector_payload=connector_payload,
        )
        call_args = self.mock_helper.create_connector_with_role.call_args[0]
        inline_policy = call_args[0]
        self.assertEqual(inline_policy["Statement"][0]["Resource"], custom_arn)

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_bedrock_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_bedrock_connector_create_12345678",
        )
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_bedrock_connector_select_model_interactive(self, mock_input):
        """Test create_bedrock_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        result = self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
        )
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["test_prefix"])
    def test_valid_connector_role_prefix(self, mock_input):
        """Test creating a Bedrock connector with a valid connector role prefix"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            region=self.region,
            model_name="Cohere embedding model",
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        _, connector_role_name, create_connector_role_name, _ = create_connector_calls[
            0
        ][0]
        self.assertTrue(
            connector_role_name.startswith("test_prefix_bedrock_connector_")
        )
        self.assertTrue(
            create_connector_role_name.startswith(
                "test_prefix_bedrock_connector_create_"
            )
        )

    @patch("builtins.input", side_effect=[""])
    def test_invalid_connector_role_prefix(self, mock_input):
        """Test creating a Bedrock connector with an invalid connector role prefix"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        with self.assertRaises(ValueError) as context:
            self.bedrock_model.create_bedrock_connector(
                helper=self.mock_helper,
                save_config_method=self.mock_save_config,
                region=self.region,
                model_name="Cohere embedding model",
            )
        self.assertEqual(
            str(context.exception), "Connector role prefix cannot be empty."
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")

    @patch("builtins.input", side_effect=["1", "test_prefix", ""])
    def test_create_bedrock_connector_default_region(self, mock_input):
        """Test creating a Bedrock connector with default region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your Bedrock region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_payload = create_connector_calls[0][0]
        self.assertEqual(connector_payload["parameters"]["region"], "us-west-2")

    @patch("builtins.input", side_effect=["1", "test_prefix", "us-east-1"])
    def test_create_bedrock_connector_custom_region(self, mock_input):
        """Test creating a Bedrock connector with a custom region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
        )
        mock_input.assert_any_call(f"Enter your Bedrock region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_payload = create_connector_calls[0][0]
        self.assertEqual(connector_payload["parameters"]["region"], "us-east-1")

    @patch("builtins.input", side_effect=["3", "test_prefix", "test-model-arn"])
    def test_enter_custom_model_arn(self, mock_input):
        """Test creating a Bedrock connector with a custom model ARN"""
        payload = {"test": "payload"}
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_payload=payload,
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
        """Test create_bedrock_connector for input_custom_model_details method"""
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

    @patch("builtins.print")
    @patch("builtins.input")
    def test_create_bedrock_connector_invalid_choice(self, mock_input, mock_print):
        """Test creating a Bedrock connector with an invalid model choice"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        mock_input.side_effect = ['{"name": "test-model"}', ""]
        self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            model_arn="test-arn",
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        self.mock_helper.create_connector_with_role.assert_called_once()

    def test_create_bedrock_connector_failure(self):
        """Test creating a Bedrock connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None
        result = self.bedrock_model.create_bedrock_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Cohere embedding model",
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
