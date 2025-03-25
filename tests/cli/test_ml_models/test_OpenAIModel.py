# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.OpenAIModel import OpenAIModel


class TestOpenAIModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.service_type = "amazon-opensearch-service"
        self.openai_model = OpenAIModel(self.service_type)
        self.connector_role_prefix = "test_role"
        self.api_key = "test_api_key"
        self.secret_name = "test_secret_name"
        self.mock_helper.create_connector_with_secret.return_value = (
            "test-connector-id",
            "test-role-arn",
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.OpenAIModel.uuid")
    def test_create_openai_connector_embedding_model(self, mock_uuid):
        """Test creating an OpenAI connector with embedding model."""
        # Mock UUID to return a consistent value
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)

        result = self.openai_model.create_openai_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once_with(
            body={
                "persistent": {
                    "plugins.ml_commons.trusted_connector_endpoints_regex": [
                        "^https://api\\.openai\\.com/.*$"
                    ]
                }
            }
        )
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_with(
            body=settings_body
        )

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_secret.assert_called_once()
        call_args = self.mock_helper.create_connector_with_secret.call_args[0]

        # Verify secret name and value
        expected_secret_name = f"{self.secret_name}_12345678"
        expected_secret_value = {"openai_api_key": self.api_key}
        self.assertEqual(call_args[0], expected_secret_name)
        self.assertEqual(call_args[1], expected_secret_value)

        # Verify role names
        expected_role_name = (
            f"{self.connector_role_prefix}_openai_embedding_model_12345678"
        )
        expected_create_role_name = (
            f"{self.connector_role_prefix}_openai_embedding_model_create_12345678"
        )
        self.assertEqual(call_args[2], expected_role_name)
        self.assertEqual(call_args[3], expected_create_role_name)

        # Verify connector payload
        connector_payload = call_args[4]
        self.assertEqual(connector_payload["name"], "OpenAI embedding model connector")
        self.assertEqual(connector_payload["protocol"], "http")
        self.assertEqual(
            connector_payload["parameters"]["model"], "text-embedding-ada-002"
        )
        self.assertEqual(
            connector_payload["actions"][0]["headers"]["Authorization"],
            f"Bearer {self.api_key}",
        )

        self.assertTrue(result)


#     @patch("builtins.input", side_effect=["", "1", "1"])
#     def test_create_openai_connector_open_source_default(self, mock_input):
#         mock_helper = Mock()
#         mock_helper.create_connector.return_value = "test-connector-id"

#         mock_config = {}
#         mock_save_config = Mock()

#         self.openai_model.service_type = "open-source"
#         self.openai_model.create_openai_connector(
#             mock_helper, mock_config, mock_save_config
#         )

#         mock_helper.create_connector.assert_called_once()
#         mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

#     @patch(
#         "builtins.input",
#         side_effect=[
#             "",
#             "1",
#             "2",
#             '{"name": "Custom Model", "description": "Custom description"}',
#         ],
#     )
#     def test_create_openai_connector_open_source_custom(self, mock_input):
#         mock_helper = Mock()
#         mock_helper.create_connector.return_value = "test-connector-id"

#         mock_config = {}
#         mock_save_config = Mock()

#         self.openai_model.service_type = "open-source"
#         self.openai_model.create_openai_connector(
#             mock_helper, mock_config, mock_save_config
#         )

#         mock_helper.create_connector.assert_called_once()
#         mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

#     @patch("builtins.input", side_effect=["", "1", "1"])
#     def test_create_openai_connector_managed_default(self, mock_input):
#         mock_helper = Mock()
#         mock_helper.create_connector_with_secret.return_value = "test-connector-id"

#         mock_config = {}
#         mock_save_config = Mock()

#         self.openai_model.service_type = "amazon-opensearch-service"
#         self.openai_model.create_openai_connector(
#             mock_helper, mock_config, mock_save_config
#         )

#         mock_helper.create_connector_with_secret.assert_called_once()
#         mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})

#     @patch(
#         "builtins.input",
#         side_effect=[
#             "",
#             "1",
#             "2",
#             '{"name": "Custom Model", "description": "Custom description"}',
#         ],
#     )
#     def test_create_openai_connector_managed_custom(self, mock_input):
#         mock_helper = Mock()
#         mock_helper.create_connector_with_secret.return_value = "test-connector-id"

#         mock_config = {}
#         mock_save_config = Mock()

#         self.openai_model.service_type = "amazon-opensearch-service"
#         self.openai_model.create_openai_connector(
#             mock_helper, mock_config, mock_save_config
#         )

#         mock_helper.create_connector_with_secret.assert_called_once()
#         mock_save_config.assert_called_once_with({"connector_id": "test-connector-id"})


if __name__ == "__main__":
    unittest.main()


# class TestOpenAIModel(unittest.TestCase):
#     def setUp(self):
#         """Set up test fixtures."""
#         self.model = OpenAIModel(service_type="amazon-opensearch-service")
#         self.helper = MagicMock()
#         self.save_config_method = MagicMock()

#         # Common test values
#         self.api_key = "test-api-key"
#         self.connector_role_prefix = "test-prefix"
#         self.secret_name = "test-secret"
#         self.test_id = "12345678"

#         # Mock uuid
#         uuid.uuid1 = MagicMock(return_value=MagicMock(hex=self.test_id))

#     @patch("builtins.input")
#     def test_create_openai_connector_embedding_model(self, mock_input):
#         """Test creating an OpenAI connector for embedding model."""
#         # Setup
#         self.helper.create_connector_with_secret.return_value = ("connector-id", "role-arn")

#         # Test execution
#         result = self.model.create_openai_connector(
#             helper=self.helper,
#             save_config_method=self.save_config_method,
#             connector_role_prefix=self.connector_role_prefix,
#             model_name="Embedding model",
#             api_key=self.api_key,
#             secret_name=self.secret_name
#         )

#         # Assertions
#         self.assertTrue(result)

#         # Verify cluster settings were updated
#         self.helper.opensearch_client.cluster.put_settings.assert_called_once_with(
#             body={
#                 "persistent": {
#                     "plugins.ml_commons.trusted_connector_endpoints_regex": [
#                         "^https://api\\.openai\\.com/.*$"
#                     ]
#                 }
#             }
#         )

#         # Verify connector creation
#         expected_connector_payload = {
#             "name": "OpenAI embedding model connector",
#             "description": "Connector for OpenAI embedding model",
#             "version": "1.0",
#             "protocol": "http",
#             "parameters": {"model": "text-embedding-ada-002"},
#             "actions": [
#                 {
#                     "action_type": "predict",
#                     "method": "POST",
#                     "url": "https://api.openai.com/v1/embeddings",
#                     "headers": {
#                         "Authorization": f"Bearer {self.api_key}",
#                     },
#                     "request_body": '{ "input": ${parameters.input}, "model": "${parameters.model}" }',
#                     "pre_process_function": "connector.pre_process.openai.embedding",
#                     "post_process_function": "connector.post_process.openai.embedding",
#                 }
#             ],
#         }

#         self.helper.create_connector_with_secret.assert_called_once_with(
#             f"{self.secret_name}_{self.test_id[:8]}",
#             {"openai_api_key": self.api_key},
#             f"{self.connector_role_prefix}_openai_embedding_model_{self.test_id[:8]}",
#             f"{self.connector_role_prefix}_openai_embedding_model_create_{self.test_id[:8]}",
#             expected_connector_payload,
#             sleep_time_in_seconds=10,
#         )

#         # Verify config was saved
#         self.save_config_method.assert_called_once_with(
#             "connector-id",
#             self.helper.get_connector.return_value,
#             f"{self.connector_role_prefix}_openai_embedding_model_{self.test_id[:8]}",
#             f"{self.secret_name}_{self.test_id[:8]}",
#             "role-arn"
#         )

#     @patch("builtins.input")
#     def test_create_openai_connector_custom_model(self, mock_input):
#         """Test creating an OpenAI connector for custom model."""
#         # Setup
#         custom_payload = {
#             "name": "Custom Model",
#             "description": "Custom OpenAI model",
#             "version": "1.0",
#             "protocol": "http",
#             "parameters": {"model": "custom-model"},
#             "actions": [{"action_type": "predict"}]
#         }

#         self.helper.create_connector_with_secret.return_value = ("connector-id", "role-arn")

#         # Test execution
#         result = self.model.create_openai_connector(
#             helper=self.helper,
#             save_config_method=self.save_config_method,
#             connector_role_prefix=self.connector_role_prefix,
#             model_name="Custom model",
#             api_key=self.api_key,
#             connector_payload=custom_payload,
#             secret_name=self.secret_name
#         )

#         # Assertions
#         self.assertTrue(result)

#         # Verify connector creation with custom payload
#         self.helper.create_connector_with_secret.assert_called_once()
#         call_args = self.helper.create_connector_with_secret.call_args[0]
#         self.assertEqual(call_args[1]["openai_api_key"], self.api_key)

#     def test_create_openai_connector_missing_role_prefix(self):
#         """Test creating an OpenAI connector with missing role prefix."""
#         with self.assertRaises(ValueError) as context:
#             self.model.create_openai_connector(
#                 helper=self.helper,
#                 save_config_method=self.save_config_method,
#                 connector_role_prefix=None,
#                 model_name="Embedding model",
#                 api_key=self.api_key
#             )

#         self.assertEqual(str(context.exception), "Connector role prefix cannot be empty.")

#     @patch("builtins.input")
#     def test_create_openai_connector_failure(self, mock_input):
#         """Test handling connector creation failure."""
#         # Setup
#         self.helper.create_connector_with_secret.return_value = (None, None)

#         # Test execution
#         result = self.model.create_openai_connector(
#             helper=self.helper,
#             save_config_method=self.save_config_method,
#             connector_role_prefix=self.connector_role_prefix,
#             model_name="Embedding model",
#             api_key=self.api_key,
#             secret_name=self.secret_name
#         )

#         # Assertions
#         self.assertFalse(result)
