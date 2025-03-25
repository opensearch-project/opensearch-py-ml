# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel import DeepSeekModel


class TestDeepSeekModel(unittest.TestCase):

    def setUp(self):
        self.service_type = "amazon-opensearch-service"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.deepseek_model = DeepSeekModel(service_type=self.service_type)

        self.connector_role_prefix = "test_role"
        self.api_key = "test_api_key"
        self.secret_name = "test_secret_name"

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel.uuid")
    def test_create_deepseek_connector_chat_model(self, mock_uuid):
        """Test creating a DeepSeek connector with Chat model."""
        # Mock UUID to return a consistent value
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.deepseek_model.create_deepseek_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="DeepSeek Chat model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()
        settings_body = {
            "persistent": {
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://api\\.deepseek\\.com/.*$"
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
        expected_secret_value = {"deepseek_api_key": self.api_key}
        self.assertEqual(call_args[0], expected_secret_name)
        self.assertEqual(call_args[1], expected_secret_value)

        # Verify role names
        expected_role_name = (
            f"{self.connector_role_prefix}_deepseek_chat_model_12345678"
        )
        expected_create_role_name = (
            f"{self.connector_role_prefix}_deepseek_chat_model_create_12345678"
        )
        self.assertEqual(call_args[2], expected_role_name)
        self.assertEqual(call_args[3], expected_create_role_name)

        # Verify connector payload
        connector_payload = call_args[4]
        self.assertEqual(connector_payload["name"], "DeepSeek Chat")
        self.assertEqual(connector_payload["protocol"], "http")
        self.assertEqual(connector_payload["parameters"]["model"], "deepseek-chat")
        self.assertEqual(
            connector_payload["actions"][0]["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel.uuid")
    def test_create_deepseek_connector_custom_model(self, mock_uuid):
        """Test creating a DeepSeek connector with Custom model."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        # Create a sample custom connector payload
        custom_payload = {
            "name": "Custom DeepSeek Connector",
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
                    "url": "https://api.deepseek.com/v1/custom",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "${auth}",
                    },
                }
            ],
        }

        result = self.deepseek_model.create_deepseek_connector(
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
        self.assertEqual(connector_payload["name"], "Custom DeepSeek Connector")
        self.assertEqual(
            connector_payload["actions"][0]["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )
        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel.uuid")
    def test_create_deepseek_connector_failure(self, mock_uuid):
        """Test connector creation failure."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_secret.return_value = None, None

        result = self.deepseek_model.create_deepseek_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="DeepSeek Chat model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        self.assertFalse(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.DeepSeekModel.uuid")
    def test_create_deepseek_connector_open_source(self, mock_uuid):
        """Test creating a DeepSeek connector for open-source service."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)

        # Create model with non-AWS service type
        open_source_model = DeepSeekModel(service_type="open-source")

        result = open_source_model.create_deepseek_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="DeepSeek Chat model",
            api_key=self.api_key,
        )

        # Verify that create_connector was called instead of create_connector_with_secret
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
