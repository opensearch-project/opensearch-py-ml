# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.BedrockModel import BedrockModel


class TestBedrockModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.bedrock_model = BedrockModel(opensearch_domain_region=self.region)
        self.connector_role_prefix = "test_role"

    @patch("uuid.uuid1")
    def test_create_bedrock_connector_cohere(self, mock_uuid):
        """Test creating a Bedrock connector with Cohere embedding model."""
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
        # self.assertEqual(call_args[1], f"{self.connector_role_prefix}_bedrock_connector_12345678")
        # self.assertEqual(call_args[2], f"{self.connector_role_prefix}_bedrock_connector_create_12345678")

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(
            connector_payload["name"], "Amazon Bedrock Cohere Connector: embedding v3"
        )
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)

        self.assertTrue(result)

    @patch("uuid.uuid1")
    def test_create_bedrock_connector_titan(self, mock_uuid):
        """Test creating a Bedrock connector with Titan embedding model."""
        # mock_uuid.retu rn_value = Mock(hex='12345678')

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
        self.assertTrue(result)

    @patch("uuid.uuid1")
    def test_create_bedrock_connector_custom(self, mock_uuid):
        """Test creating a Bedrock connector with custom model."""
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
        self.assertTrue(result)

    def test_create_bedrock_connector_failure(self):
        """Test connector creation failure scenarios."""
        # Test when create_connector_with_role fails
        self.mock_helper.create_connector_with_role.return_value = None

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
