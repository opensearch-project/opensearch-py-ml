# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.cli.ml_models.custom_connector import CustomConnector


class TestCustomConnector(unittest.TestCase):
    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.service_type = CustomConnector.AMAZON_OPENSEARCH_SERVICE
        self.custom_connector = CustomConnector(service_type=self.service_type)
        self.connector_role_prefix = "test_role"
        self.connector_secret_name = "test_secret_name"
        self.connector_role_inline_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": ["*"]}
            ],
        }
        self.api_key = "test_api_key"
        self.connector_body = {
            "name": "Test Model",
            "description": "Test description",
            "version": "1",
            "parameters": {"api_name": "test_api_name"},
        }

    def test_create_connector_with_policy(self):
        """Test creating custom connector with inline policy"""
        # Setup
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            None,
        )

        # Execute
        result = self.custom_connector.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            connector_role_inline_policy=self.connector_role_inline_policy,
            required_policy=True,
            required_secret=False,
            connector_body=self.connector_body,
        )

        # Verify method calls
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    def test_create_connector_with_secret(self):
        """Test creating custom connector with secret"""
        # Setup
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test_secret_arn",
        )

        # Execute
        result = self.custom_connector.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            connector_secret_name=self.connector_secret_name,
            required_policy=False,
            required_secret=True,
            api_key=self.api_key,
            connector_body=self.connector_body,
        )

        # Verify method calls
        self.mock_helper.create_connector_with_secret.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["yes", '{"test": "policy"}', ""])
    def test_create_connector_with_policy_input(self, mock_input):
        """Test creating custom connector with policy from user input"""
        # Setup
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            None,
        )

        # Execute
        result = self.custom_connector.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            required_secret=False,
            connector_body=self.connector_body,
        )

        # Verify method calls
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["yes", "test-secret"])
    def test_create_connector_with_secret_input(self, mock_input):
        """Test creating custom connector with secret configuration from user input"""
        # Setup
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test-secret-arn",
        )

        # Execute
        result = self.custom_connector.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            required_policy=False,
            api_key=self.api_key,
            connector_body=self.connector_body,
        )

        # Verify method calls
        self.mock_helper.create_connector_with_secret.assert_called_once()
        self.assertTrue(result)

    def test_create_connector_open_source(self):
        """Test creating custom connector in open-source service"""
        # Create model with open-source service type
        open_source_model = CustomConnector(service_type=CustomConnector.OPEN_SOURCE)

        # Execute
        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_body=self.connector_body,
        )

        # Verify method call
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    def test_create_connector_failure(self):
        """Test custom connector creation failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None, None
        result = self.custom_connector.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            required_policy=True,
            required_secret=False,
            connector_role_inline_policy=self.connector_role_inline_policy,
            connector_body=self.connector_body,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
