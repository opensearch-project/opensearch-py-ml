# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel import (
    SageMakerModel,
)


class TestSageMakerModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.sagemaker_model = SageMakerModel(opensearch_domain_region=self.region)
        self.connector_role_prefix = "test_role"
        self.connector_endpoint_arn = "test_arn"
        self.connector_endpoint_url = "test_url"

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_embedding(self, mock_uuid):
        """Test creating a SageMaker connector with embedding model."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_role.assert_called_once()
        call_args = self.mock_helper.create_connector_with_role.call_args[0]

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(
            connector_payload["name"], "SageMaker Embedding Model Connector"
        )
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_deepseek(self, mock_uuid):
        """Test creating a SageMaker connector with DeepSeek R1 model."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="DeepSeek R1 model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_role.assert_called_once()
        call_args = self.mock_helper.create_connector_with_role.call_args[0]

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(connector_payload["name"], "DeepSeek R1 model connector")
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_custom(self, mock_uuid):
        """Test creating a SageMaker connector with custom model."""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        connector_payload = {
            "name": "Custom SageMaker Connector",
            "description": "Test custom connector",
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": {"service_name": "sagemaker", "region": self.region},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": self.connector_endpoint_url,
                    "headers": {"Content-Type": "application/json"},
                    "request_body": "${parameters.input}",
                }
            ],
        }

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
            connector_payload=connector_payload,
        )

        call_args = self.mock_helper.create_connector_with_role.call_args[0]
        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )
        self.assertTrue(result)

    def test_create_sagemaker_connector_failure(self):
        """Test connector creation failure scenarios."""
        # Test when create_connector_with_role fails
        self.mock_helper.create_connector_with_role.return_value = None, None

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
