# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.CohereModel import CohereModel


class TestCohereModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.cohere_model = CohereModel()
        self.connector_role_prefix = "test_role"
        self.api_key = "test_api_key"
        self.secret_name = "test_secret_name"

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.CohereModel.uuid")
    def test_create_cohere_connector_embedding(self, mock_uuid):
        """Test creating a Cohere connector with Embedding model"""
        # Mock UUID to return a consistent value
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.cohere\\.ai/.*$"
                ]
            }
        }
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_with(
            body=settings_body
        )

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_secret.assert_called_once()
        call_args = self.mock_helper.create_connector_with_secret.call_args[0]

        # Verify secret name and value
        expected_secret_name = f"{self.secret_name}_12345678"
        expected_secret_value = {"cohere_api_key": self.api_key}
        self.assertEqual(call_args[0], expected_secret_name)
        self.assertEqual(call_args[1], expected_secret_value)

        # Verify role names
        expected_role_name = f"{self.connector_role_prefix}_cohere_connector_12345678"
        expected_create_role_name = (
            f"{self.connector_role_prefix}_cohere_connector_create_12345678"
        )
        self.assertEqual(call_args[2], expected_role_name)
        self.assertEqual(call_args[3], expected_create_role_name)

        # Verify connector payload
        connector_payload = call_args[4]
        self.assertEqual(connector_payload["name"], "Cohere Embedding Model Connector")
        self.assertEqual(connector_payload["protocol"], "http")
        self.assertEqual(connector_payload["parameters"]["model"], "embed-english-v3.0")
        self.assertEqual(
            connector_payload["actions"][0]["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.CohereModel.uuid")
    def test_create_cohere_connector_custom(self, mock_uuid):
        """Test creating a Cohere connector with custom model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        # Create a sample custom connector payload
        custom_payload = {
            "name": "Custom Cohere Connector",
            "description": "Test custom connector",
            "version": "1",
            "protocol": "http",
            "parameters": {
                "model": "custom-model",
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.cohere.ai/v1/custom",
                    "headers": {
                        "Authorization": "${auth}",
                    },
                }
            ],
        }

        result = self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Custom model",
            api_key=self.api_key,
            connector_payload=custom_payload,
            secret_name=self.secret_name,
        )

        call_args = self.mock_helper.create_connector_with_secret.call_args[0]
        connector_payload = call_args[4]
        self.assertEqual(connector_payload["name"], "Custom Cohere Connector")
        self.assertEqual(
            connector_payload["actions"][0]["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_cohere_connector_select_model_interactive(self, mock_input):
        """Test create_cohere_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )

        result = self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        self.mock_helper.create_connector_with_secret.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input")
    def test_cohere_api_key(self, mock_input):
        """Test create_cohere getting Cohere API key with asterisk masking"""
        mock_input.return_value = "test-api-key-123"
        self.mock_helper.get_password_with_asterisks.return_value = "test-api-key-123"
        api_key = self.mock_helper.get_password_with_asterisks(
            "Enter your Cohere API key: "
        )
        self.mock_helper.get_password_with_asterisks.assert_called_once_with(
            "Enter your Cohere API key: "
        )
        self.assertEqual(api_key, "test-api-key-123")

    @patch("builtins.input", side_effect=["test_prefix"])
    def test_valid_connector_role_prefix(self, mock_input):
        """Test creating a Cohere connector with a valid connector role prefix"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            api_key=self.api_key,
            model_name="Embedding model",
            secret_name=self.secret_name,
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_secret.call_args_list
        )
        _, _, connector_role_name, create_connector_role_name, _ = (
            create_connector_calls[0][0]
        )
        self.assertTrue(connector_role_name.startswith("test_prefix_cohere_connector_"))
        self.assertTrue(
            create_connector_role_name.startswith(
                "test_prefix_cohere_connector_create_"
            )
        )

    @patch("builtins.input", side_effect=[""])
    def test_invalid_connector_role_prefix(self, mock_input):
        """Test creating a Cohere connector with an invalid connector role prefix"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        with self.assertRaises(ValueError) as context:
            self.cohere_model.create_cohere_connector(
                helper=self.mock_helper,
                save_config_method=self.mock_save_config,
                api_key=self.api_key,
                model_name="Embedding model",
                secret_name=self.secret_name,
            )
        self.assertEqual(
            str(context.exception), "Connector role prefix cannot be empty."
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")

    @patch("builtins.input", side_effect=["test_secret"])
    def test_create_cohere_connector_secret_name(self, mock_input):
        """Test creating a Cohere connector when user provides a secret name through the prompt"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            api_key=self.api_key,
            model_name="Embedding model",
            connector_role_prefix=self.connector_role_prefix,
        )
        mock_input.assert_any_call("Enter a name for the AWS Secrets Manager secret: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_secret.call_args_list
        )
        secret_name, _, _, _, _ = create_connector_calls[0][0]
        self.assertTrue(secret_name.startswith("test_secret"))

    @patch("builtins.input")
    def test_input_custom_model_details(self, mock_input):
        """Test create_cohere_connector for input_custom_model_details method"""
        mock_input.side_effect = [
            '{"name": "test-model",',
            '"description": "test description",',
            '"parameters": {"param": "value"}}',
            "",
        ]
        result = self.cohere_model.input_custom_model_details()
        expected_result = {
            "name": "test-model",
            "description": "test description",
            "parameters": {"param": "value"},
        }
        self.assertEqual(result, expected_result)

    @patch("builtins.print")
    @patch("builtins.input")
    def test_create_cohere_connector_invalid_choice(self, mock_input, mock_print):
        """Test creating a Cohere connector with an invalid model choice"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        mock_input.side_effect = ['{"name": "test-model"}', ""]
        self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Invalid Model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        self.mock_helper.create_connector_with_secret.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.CohereModel.uuid")
    def test_create_cohere_connector_failure(self, mock_uuid):
        """Test creating a Cohere connector in failure scenario"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = None, None

        result = self.cohere_model.create_cohere_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
