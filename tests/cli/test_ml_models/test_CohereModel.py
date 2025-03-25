# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

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
        """Test creating a Cohere connector with Embedding model."""
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
        """Test creating a Cohere connector with Custom model."""
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

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.CohereModel.uuid")
    def test_create_cohere_connector_failure(self, mock_uuid):
        """Test connector creation failure."""
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
