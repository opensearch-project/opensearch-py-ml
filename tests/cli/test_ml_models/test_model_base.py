# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class TestModelBase(unittest.TestCase):

    def setUp(self):
        self.model_base = ModelBase()
        self.mock_helper = Mock()
        self.valid_json = """
{
    "name": "Test Model",
    "description": "Test connector",
    "version": "1",
    "protocol": "http",
    "parameters": {
        "model": "test-model"
    },
    "actions": [
        {
            "action_type": "predict",
            "method": "POST",
            "url": "https://api.test.com/v1/predict",
            "headers": {
                "Authorization": "${auth}"
            }
        }
    ]
}"""
        self.invalid_json = """
{
    "name": Invalid JSON,
    "missing": "quotes"
}"""

    def test_set_trusted_endpoint(self):
        """Test set_trusted_endpoint with valid input"""
        # Test data
        trusted_endpoint = "https://test-endpoint.com/*"
        expected_settings = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    trusted_endpoint
                ]
            }
        }

        # Execute
        self.model_base.set_trusted_endpoint(self.mock_helper, trusted_endpoint)

        # Verify
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once_with(
            body=expected_settings
        )

    @patch("uuid.uuid1")
    def test_create_connector_role_with_valid_inputs(self, mock_uuid):
        """Test create_connector_role with valid role prefix and model name"""
        # Mock UUID to return a fixed value
        mock_uuid_instance = Mock()
        mock_uuid_instance.configure_mock(__str__=lambda _: "123456")
        mock_uuid.return_value = mock_uuid_instance

        # Test inputs
        connector_role_prefix = "test-prefix"
        model_name = "model"

        # Execute
        connector_role_name, create_connector_role_name = (
            self.model_base.create_connector_role(connector_role_prefix, model_name)
        )

        # Verify
        expected_connector_role = "test-prefix-model-connector-123456"
        expected_create_role = "test-prefix-model-connector-create-123456"

        self.assertEqual(connector_role_name, expected_connector_role)
        self.assertEqual(create_connector_role_name, expected_create_role)
        mock_uuid.assert_called_once()

    @patch("builtins.input", return_value="input-prefix")
    @patch("uuid.uuid1")
    def test_create_connector_role_with_empty_prefix(self, mock_uuid, mock_input):
        """Test create_connector_role when prefix is empty and provided via input"""
        # Mock UUID to return a fixed value
        mock_uuid_instance = Mock()
        mock_uuid_instance.configure_mock(__str__=lambda _: "123456")
        mock_uuid.return_value = mock_uuid_instance

        # Test inputs
        connector_role_prefix = ""
        model_name = "model"

        # Execute
        connector_role_name, create_connector_role_name = (
            self.model_base.create_connector_role(connector_role_prefix, model_name)
        )

        # Verify
        expected_connector_role = "input-prefix-model-connector-123456"
        expected_create_role = "input-prefix-model-connector-create-123456"

        self.assertEqual(connector_role_name, expected_connector_role)
        self.assertEqual(create_connector_role_name, expected_create_role)
        mock_input.assert_called_once_with("Enter your connector role prefix: ")
        mock_uuid.assert_called_once()

    @patch("builtins.input", return_value="")
    def test_create_connector_role_with_empty_input(self, mock_input):
        """Test create_connector_role when both prefix and input are empty"""
        # Test inputs
        connector_role_prefix = ""
        model_name = "test_model"

        # Execute and verify exception
        with self.assertRaises(ValueError) as context:
            self.model_base.create_connector_role(connector_role_prefix, model_name)

        self.assertEqual(
            str(context.exception), "Connector role prefix cannot be empty."
        )
        mock_input.assert_called_once_with("Enter your connector role prefix: ")

    @patch("uuid.uuid1")
    def test_create_secret_name_with_valid_inputs(self, mock_uuid):
        """Test create_secret_name with all valid inputs provided"""
        # Mock UUID to return a fixed value
        mock_uuid_instance = Mock()
        mock_uuid_instance.configure_mock(__str__=lambda _: "1234")
        mock_uuid.return_value = mock_uuid_instance

        # Test inputs
        secret_name = "test-secret"
        model_name = "test_model"
        api_key = "test_api_key_123"

        # Execute
        result_secret_name, result_secret_value = self.model_base.create_secret_name(
            secret_name, model_name, api_key
        )

        # Verify
        expected_secret_name = "test-secret-1234"
        expected_secret_value = {"test_model_api_key": "test_api_key_123"}

        self.assertEqual(result_secret_name, expected_secret_name)
        self.assertEqual(result_secret_value, expected_secret_value)

    @patch("builtins.input", return_value="input_secret")
    @patch("uuid.uuid1")
    def test_create_secret_name_with_empty_secret_name(self, mock_uuid, mock_input):
        """Test create_secret_name when secret_name is empty and provided via input"""
        # Mock UUID to return a fixed value
        mock_uuid_instance = Mock()
        mock_uuid_instance.configure_mock(__str__=lambda _: "1234")
        mock_uuid.return_value = mock_uuid_instance

        # Test inputs
        secret_name = ""
        model_name = "test_model"
        api_key = "test_api_key_123"

        # Execute
        result_secret_name, result_secret_value = self.model_base.create_secret_name(
            secret_name, model_name, api_key
        )

        # Verify
        expected_secret_name = "input_secret-1234"
        expected_secret_value = {"test_model_api_key": "test_api_key_123"}

        self.assertEqual(result_secret_name, expected_secret_name)
        self.assertEqual(result_secret_value, expected_secret_value)
        mock_input.assert_called_once_with(
            "Enter a name for the AWS Secrets Manager secret: "
        )

    def test_set_api_key_with_provided_key(self):
        """Test set_api_key when API key is provided"""
        # Test inputs
        api_key = "test_api_key_123"
        model_name = "test_model"

        # Execute
        result = self.model_base.set_api_key(api_key, model_name)

        # Verify
        self.assertEqual(result, "test_api_key_123")

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks")
    def test_set_api_key_with_empty_key(self, mock_get_password):
        """Test set_api_key when API key is empty and provided via input"""
        # Setup mock
        mock_get_password.return_value = "test_api_key"

        # Test inputs
        api_key = ""
        model_name = "test_model"

        # Execute
        result = self.model_base.set_api_key(api_key, model_name)

        # Verify
        self.assertEqual(result, "test_api_key")
        mock_get_password.assert_called_once_with("Enter your test_model API key: ")

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ConnectorManager")
    def test_get_model_details_with_model_name(self, mock_connector_manager):
        """Test get_model_details when model_name is provided"""
        # Setup mock models
        mock_model1 = type("Model", (), {"id": "1", "name": "model1"})()
        mock_model2 = type("Model", (), {"id": "2", "name": "model2"})()
        available_models = [mock_model1, mock_model2]

        # Setup mock connector manager
        mock_instance = Mock()
        mock_instance.get_available_models.return_value = available_models
        mock_connector_manager.return_value = mock_instance

        # Test inputs
        connector_name = "test_connector"
        service_type = "test_service"
        model_name = "model2"

        # Execute
        result = self.model_base.get_model_details(
            connector_name, service_type, model_name
        )

        # Verify
        self.assertEqual(result, "2")
        mock_instance.get_available_models.assert_called_once_with(
            connector_name, service_type
        )

    @patch("builtins.input", return_value="2")
    @patch("builtins.print")
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ConnectorManager")
    def test_get_model_details_without_model_name(
        self, mock_connector_manager, mock_print, mock_input
    ):
        """Test get_model_details when model_name is not provided"""
        # Setup mock models
        mock_model1 = type("Model", (), {"id": "1", "name": "model1"})()
        mock_model2 = type("Model", (), {"id": "2", "name": "model2"})()
        available_models = [mock_model1, mock_model2]

        # Setup mock connector manager
        mock_instance = Mock()
        mock_instance.get_available_models.return_value = available_models
        mock_connector_manager.return_value = mock_instance

        # Test inputs
        connector_name = "test_connector"
        service_type = "test_service"

        # Execute
        result = self.model_base.get_model_details(connector_name, service_type)

        # Verify
        self.assertEqual(result, "2")
        mock_instance.get_available_models.assert_called_once_with(
            connector_name, service_type
        )
        mock_print.assert_any_call(
            "\nPlease select a model for the connector creation: "
        )
        mock_print.assert_any_call("1. model1")
        mock_print.assert_any_call("2. model2")
        mock_input.assert_called_once_with("Enter your choice (1-2): ")

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ConnectorManager")
    def test_get_model_details_no_models_found(self, mock_connector_manager):
        """Test get_model_details when no models are found"""
        # Setup mock connector manager
        mock_instance = Mock()
        mock_instance.get_available_models.return_value = []
        mock_connector_manager.return_value = mock_instance

        # Test inputs
        connector_name = "test_connector"
        service_type = "test_service"

        # Execute and verify exception
        with self.assertRaises(ValueError) as context:
            self.model_base.get_model_details(connector_name, service_type)

        self.assertEqual(
            str(context.exception), "No models found for connector: test_connector"
        )
        mock_instance.get_available_models.assert_called_once_with(
            connector_name, service_type
        )

    @patch("builtins.input", side_effect=["3", "invalid", "2"])
    @patch("builtins.print")
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ConnectorManager")
    def test_get_model_details_invalid_choices(
        self, mock_connector_manager, mock_print, mock_input
    ):
        """Test get_model_details with invalid choices before valid input"""
        # Setup mock models
        mock_model1 = type("Model", (), {"id": "1", "name": "model1"})()
        mock_model2 = type("Model", (), {"id": "2", "name": "model2"})()
        available_models = [mock_model1, mock_model2]

        # Setup mock connector manager
        mock_instance = Mock()
        mock_instance.get_available_models.return_value = available_models
        mock_connector_manager.return_value = mock_instance

        # Test inputs
        connector_name = "test_connector"
        service_type = "test_service"

        # Execute
        result = self.model_base.get_model_details(connector_name, service_type)

        # Verify
        self.assertEqual(result, "2")
        mock_instance.get_available_models.assert_called_once_with(
            connector_name, service_type
        )
        self.assertEqual(mock_input.call_count, 3)
        mock_print.assert_any_call("Invalid choice. Please enter a valid number.")

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_valid_json(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with valid JSON input"""
        json_lines = [
            line.strip() for line in self.valid_json.strip().split("\n") if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details()

        # Verify the result is parsed correctly
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test Model")
        self.assertEqual(result["protocol"], "http")
        self.assertEqual(result["actions"][0]["headers"]["Authorization"], "${auth}")

        # Verify prints were called
        mock_print.assert_any_call("Please enter your model details as a JSON object.")
        mock_console_print.assert_any_call("[bold]Amazon OpenSearch Service:[/bold]")

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_invalid_json(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with invalid JSON input"""
        json_lines = [
            line.strip()
            for line in self.invalid_json.strip().split("\n")
            if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details()

        # Verify the result is None for invalid JSON
        self.assertIsNone(result)

        # Verify error message was printed
        error_calls = [
            call
            for call in mock_print.call_args_list
            if call[0]
            and isinstance(call[0][0], str)
            and call[0][0].startswith("Invalid JSON input")
        ]

        self.assertTrue(error_calls)

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_external_true(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with external=True parameter"""
        json_lines = [
            line.strip() for line in self.valid_json.strip().split("\n") if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details(external=True)

        # Verify the result is parsed correctly
        self.assertIsNotNone(result)

        # Verify external-specific messages were printed
        mock_print.assert_any_call(
            f"{Fore.YELLOW}\nIMPORTANT: When customizing the connector configuration that requires API key authentication, ensure you include the following in the 'headers' section:"
        )
        mock_print.assert_any_call(
            f'{Fore.YELLOW}{Style.BRIGHT}"Authorization": "${{auth}}"'
        )


if __name__ == "__main__":
    unittest.main()
